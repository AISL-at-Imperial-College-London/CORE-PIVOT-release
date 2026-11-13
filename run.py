"""Reviewer entry point: train a method, or evaluate frozen selected weights.

Examples (run from the repository root):
  python run.py train iql --seed 123 --device cuda
  python run.py train hoflon --all-seeds --dry-run
  python run.py evaluate iql --seed 123
  python run.py evaluate core-pivot --source trained --condition all

This launcher uses only the standard library. Help and dry runs never import
learners, instantiate optimizers, download files, or start subprocesses.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
METHODS = ('bc', 'easybcq', 'hoflon', 'core-pivot', 'td3-bc', 'iql', 'cql', 'best-pd')
SEEDS = (123, 124, 125, 126, 127)
UPDATES = 500000
REPLAY_FIELDS = ('observations', 'next_observations', 'actions', 'next_actions',
                 'rewards', 'terminals', 'timeouts')


@dataclass
class Job:
    label: str
    command: list[str]
    output: Path
    # Selected downloads are checked without deserializing any checkpoint.
    hashes: list[tuple[Path, str]] = field(default_factory=list)
    # Native checkpoint identity comes from hash-linked run/step sidecars.
    native: list[tuple[Path, str, int, int]] = field(default_factory=list)


@dataclass
class Plan:
    jobs: list[Job] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise ValueError(f'Missing file: {path}') from exc


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rooted(path: Path | str, root: Path) -> Path:
    return (root / path).resolve()


def family(method: str) -> str:
    return 'qbeta' if method in ('easybcq', 'hoflon') else method


def run_dir(runs: Path, algorithm: str, seed: int) -> Path:
    return runs / f'{algorithm}_s{seed}'


def native_path(runs: Path, algorithm: str, seed: int, step: int) -> Path:
    return run_dir(runs, algorithm, seed) / 'checkpoints' / f'step{step:09d}.pt'


def train_spec(root: Path, algorithm: str):
    spec = read_json(root / 'configs' / 'train' / f'{algorithm}.json')
    if spec['algorithm'] != algorithm or spec['steps'] != UPDATES:
        raise ValueError(f'{algorithm}: expected the full 500k paper configuration')
    return spec


def verify_native(path: Path, algorithm: str, seed: int, step: int, root: Path):
    """Verify provenance without torch.load or optimizer construction."""
    if not path.is_file():
        raise ValueError(f'Missing trained checkpoint: {path}. Train it explicitly first.')
    proof = read_json(path.with_suffix('.json'))
    run_path = path.parent.parent / 'run.json'
    record = read_json(run_path)
    identity = record['identity']
    expected = {'algorithm': algorithm, 'seed': seed,
                'config': train_spec(root, algorithm)['config'], 'smoke_only': False}
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Native run identity/settings differ from the paper configuration: {run_path}')
    if (proof.get('step') != step or proof.get('model_sha256') != sha256(path)
            or proof.get('run_sha256') != sha256(run_path)):
        raise ValueError(f'Native checkpoint/run commitment mismatch: {path}')
    return identity


def reuse_completed(runs: Path, algorithm: str, seed: int, data: Path, root: Path) -> bool:
    output = run_dir(runs, algorithm, seed)
    if not output.exists():
        return False
    marker = output / f'completed_step{UPDATES:09d}.json'
    if not marker.is_file():
        raise ValueError(f'Existing/incomplete run will not be overwritten: {output}. '
                         'Use scripts/train.py --resume explicitly, or choose another --runs-dir.')
    complete = read_json(marker)
    if complete.get('training_complete') is not True or complete.get('step') != UPDATES:
        raise ValueError(f'Invalid completion record: {marker}')
    path = native_path(runs, algorithm, seed, UPDATES)
    identity = verify_native(path, algorithm, seed, UPDATES, root)
    # Ensure a shared Q fit is useful to both named deployment methods.
    selected = read_json(root / 'configs' / 'experiments.json')['selected']
    steps = {selected[m]['checkpoint'] for m in METHODS
             if m != 'best-pd' and family(m) == algorithm}
    for step in sorted(steps - {UPDATES}):
        verify_native(native_path(runs, algorithm, seed, step), algorithm, seed, step, root)
    for name, key in [('manifest.json', 'data_manifest_sha256'),
                      ('normalization.json', 'normalization_sha256')]:
        if not (data / name).is_file() or identity.get(key) != sha256(data / name):
            raise ValueError(f'Cannot reuse {output}: prepared data identity differs or is missing ({data / name}).')
    if algorithm == 'core-pivot':
        bc = native_path(runs, 'bc', 123, UPDATES)
        if not bc.is_file() or identity.get('bc_checkpoint_sha256') != sha256(bc):
            raise ValueError(f'Cannot reuse {output}: shared seed-123 BC identity differs.')
    return True


def build_training(args, root: Path) -> Plan:
    plan, seen = Plan(), set()
    runs, data = rooted(args.runs_dir, root), rooted(args.data_dir, root)
    methods = METHODS if args.algorithm == 'all' else [args.algorithm]
    seeds = SEEDS if args.all_seeds else [args.seed]

    def add(algorithm, seed):
        if (algorithm, seed) in seen:
            return
        seen.add((algorithm, seed))
        train_spec(root, algorithm)
        if algorithm == 'core-pivot':
            add('bc', 123)
        if reuse_completed(runs, algorithm, seed, data, root):
            plan.notes.append(f'Reuse verified completed fit: {algorithm}, seed {seed}.')
            return
        output = run_dir(runs, algorithm, seed)
        command = [sys.executable, 'scripts/train.py', '--algorithm', algorithm,
                   '--data-dir', str(data), '--seed', str(seed), '--device', args.device,
                   '--output', str(output), '--steps', str(UPDATES)]
        if algorithm == 'core-pivot':
            command += ['--bc-checkpoint', str(native_path(runs, 'bc', 123, UPDATES))]
        plan.jobs.append(Job(f'Train {algorithm}, seed {seed}: {UPDATES:,} updates', command, output))

    for method in methods:
        if method == 'best-pd':
            plan.notes.append('Best-PD is a fixed controller: no training or checkpoint is required.')
            continue
        for seed in seeds:
            if method in ('easybcq', 'hoflon'):
                add('bc', seed)
            add(family(method), seed)
    if any(m in ('easybcq', 'hoflon') for m in methods):
        plan.notes.append('EasyBCQ and HOFLON share one Q_beta fit per seed; their BC500k proposals are seed-matched.')
    if 'core-pivot' in methods:
        plan.notes.append('CORE-PIVOT uses the same BC500k seed-123 proposal for every critic seed.')
    return plan


def selected_entries(manifest: Path):
    if not manifest.is_file():
        raise ValueError(f'Missing selected-weight manifest: {manifest}. Restore it from the release repository.')
    document = read_json(manifest)
    if document.get('schema_version') != 1 or not isinstance(document.get('checkpoints'), list):
        raise ValueError(f'Unsupported selected-weight manifest: {manifest}')
    entries = {}
    for entry in document['checkpoints']:
        key = (entry['method'], entry['seed'])
        if key in entries:
            raise ValueError(f'Duplicate selected checkpoint identity: {key}')
        entries[key] = entry
    return entries


def manifest_file(record, manifest: Path) -> tuple[Path, str]:
    relative = Path(record['path'])
    path = (manifest.parent / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(manifest.parent.resolve()):
        raise ValueError(f'Manifest checkpoint path must stay inside its directory: {relative}')
    digest = record['sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError(f'Invalid manifest SHA256: {relative}')
    return path, digest


def conditions(root: Path) -> list[str]:
    robustness = read_json(root / 'configs' / 'experiments.json')['evaluation']['robustness']
    return [f'{family_name}_{value:g}'
            for family_name, key in [('severity', 'random_walk_peaks'), ('step', 'step_peaks'),
                                     ('reversing', 'reversing_peaks')]
            for value in robustness[key]]


def build_evaluation(args, root: Path) -> Plan:
    if args.split == 'test' and not args.allow_test:
        raise ValueError('Test evaluation requires --allow-test after freezing checkpoints/settings.')
    if args.condition and args.condition not in conditions(root) + ['all']:
        raise ValueError('Unknown robustness condition. Choose: ' + ', '.join(conditions(root) + ['all']))
    if args.winds is not None and args.winds < 1:
        raise ValueError('--winds must be positive (diagnostic prefix only).')
    plan = Plan(notes=['Evaluation is CPU-only and never trains or downloads weights.'])
    methods = METHODS if args.algorithm == 'all' else [args.algorithm]
    seeds = SEEDS if args.all_seeds else [args.seed]
    runs = rooted(args.runs_dir, root)
    manifest = rooted(args.manifest, root)
    entries = (selected_entries(manifest) if args.source == 'selected'
               and any(m != 'best-pd' for m in methods) else {})
    settings = read_json(root / 'configs' / 'experiments.json')['selected']
    tag = args.condition or args.split
    if args.winds is not None:
        tag += f'_winds{args.winds}'
    if args.trajectories:
        tag += '_trajectories'
    output_dir = rooted(args.output_dir, root) if args.output_dir else runs / 'evaluations' / args.source
    for method in methods:
        for seed in ([123] if method == 'best-pd' else seeds):
            stem = method if method == 'best-pd' else f'{method}_s{seed}'
            output = rooted(args.output, root) if args.output else output_dir / f'{stem}_{tag}.json'
            command = [sys.executable, 'scripts/evaluate.py', '--method', method,
                       '--seed', str(seed), '--output', str(output)]
            job = Job(f'Evaluate {stem}: {tag} ({args.source})', command, output)
            if method != 'best-pd':
                step = settings[method]['checkpoint']
                if args.source == 'selected':
                    entry = entries.get((method, seed))
                    if entry is None or entry.get('step') != step:
                        raise ValueError(f'Manifest lacks the fixed selected {method} seed-{seed} step-{step} checkpoint.')
                    checkpoint, digest = manifest_file(entry, manifest)
                    job.hashes.append((checkpoint, digest))
                    command += ['--checkpoint-sha256', digest]
                    bc_record = entry.get('bc_checkpoint')
                    if method in ('easybcq', 'hoflon') or bc_record is not None:
                        expected_seed = 123 if method == 'core-pivot' else seed
                        if not bc_record or bc_record.get('seed') != expected_seed or bc_record.get('step') != UPDATES:
                            raise ValueError(f'{method} requires the correct frozen BC500k proposal (seed {expected_seed}).')
                        bc, bc_digest = manifest_file(bc_record, manifest)
                        job.hashes.append((bc, bc_digest))
                        command += ['--bc-checkpoint', str(bc)]
                else:
                    checkpoint = native_path(runs, family(method), seed, step)
                    job.native.append((checkpoint, family(method), seed, step))
                    if method in ('easybcq', 'hoflon'):
                        bc = native_path(runs, 'bc', seed, UPDATES)
                        job.native.append((bc, 'bc', seed, UPDATES))
                        command += ['--bc-checkpoint', str(bc)]
                    elif method == 'core-pivot':
                        # The policy embeds its proposal; verify the training source too.
                        bc = native_path(runs, 'bc', 123, UPDATES)
                        job.native.append((bc, 'bc', 123, UPDATES))
                command += ['--checkpoint', str(checkpoint), '--expected-step', str(step)]
            if method in ('easybcq', 'hoflon'):
                command += ['--candidates', str(settings[method]['candidates'])]
            if method == 'hoflon':
                command += ['--radius', str(settings[method]['radius']),
                            '--iterations', str(settings[method]['iterations'])]
            command += ['--condition', args.condition] if args.condition else ['--split', args.split]
            if args.allow_test:
                command.append('--allow-test')
            if args.winds is not None:
                command += ['--winds', str(args.winds)]
            if args.trajectories:
                command.append('--trajectories')
            plan.jobs.append(job)
    if args.output and len(plan.jobs) != 1:
        raise ValueError('--output names one result only; use --output-dir for multiple methods/seeds.')
    return plan


def preflight(plan: Plan, args, root: Path):
    """Check the entire batch before starting its first subprocess."""
    checked = set()
    for job in plan.jobs:
        if job.output.exists():
            raise ValueError(f'Refusing to overwrite existing output: {job.output}')
        if args.command == 'evaluate' and args.trajectories:
            folder = job.output.parent / (job.output.stem + '_trajectories')
            if folder.exists():
                raise ValueError(f'Refusing to overwrite existing trajectories: {folder}')
        for path, digest in job.hashes:
            if (path, digest) in checked:
                continue
            if not path.is_file():
                raise ValueError(f'Missing selected weights: {path}. Run python scripts/download_checkpoints.py first.')
            if sha256(path) != digest:
                raise ValueError(f'Selected checkpoint SHA256 mismatch: {path}')
            checked.add((path, digest))
        identities = {}
        for path, algorithm, seed, step in job.native:
            identities[algorithm] = verify_native(path, algorithm, seed, step, root)
        if 'core-pivot' in identities:
            bc = next(path for path, algorithm, _, _ in job.native if algorithm == 'bc')
            if identities['core-pivot'].get('bc_checkpoint_sha256') != sha256(bc):
                raise ValueError('CORE-PIVOT training proposal differs from the shared seed-123 BC checkpoint.')
    if args.command == 'train' and plan.jobs:
        data = rooted(args.data_dir, root)
        required = ['manifest.json', 'normalization.json'] + [f'{name}.npy' for name in REPLAY_FIELDS]
        for name in required:
            if not (data / name).is_file():
                raise ValueError(f'Missing prepared training data: {data / name}. '
                                 f'Run python scripts/prepare_data.py --output "{data}" explicitly first.')


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'evaluate'):
        part = subparsers.add_parser(name, help=('Fit full 500k paper settings' if name == 'train'
                                               else 'Evaluate frozen selected checkpoints; never train'))
        part.add_argument('algorithm', choices=(*METHODS, 'all'))
        group = part.add_mutually_exclusive_group()
        group.add_argument('--seed', type=int, default=123, help='Training seed (default: 123)')
        group.add_argument('--all-seeds', action='store_true', help='Run seeds 123-127 sequentially; Best-PD runs once')
        part.add_argument('--runs-dir', type=Path, default=Path('runs'),
                          help='Native fits at <dir>/<family>_s<seed> (default: runs)')
        part.add_argument('--dry-run', action='store_true', help='Print the plan only; never execute/download/create outputs')
        if name == 'train':
            part.add_argument('--data-dir', type=Path, default=Path('data/paper'))
            part.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
        else:
            part.add_argument('--source', choices=('selected', 'trained'), default='selected',
                              help='Released weights (default) or native fits from run.py train')
            part.add_argument('--manifest', type=Path, default=Path('checkpoints/selected/manifest.json'))
            group = part.add_mutually_exclusive_group()
            group.add_argument('--split', choices=('validation', 'test'), default='validation')
            group.add_argument('--condition', help='One robustness condition (e.g. reversing_3), or all 17: all')
            part.add_argument('--allow-test', action='store_true', help='Explicitly permit held-out test access after freezing selection')
            part.add_argument('--winds', type=int, help='Diagnostic prefix per condition; omit for complete evaluation')
            part.add_argument('--trajectories', action='store_true', help='Save per-episode state/action/wind arrays')
            group = part.add_mutually_exclusive_group()
            group.add_argument('--output', type=Path, help='One new JSON result file')
            group.add_argument('--output-dir', type=Path, help='Batch output directory (default: runs/evaluations/<source>)')
    return parser


def main(argv=None, *, root: Path | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    root = ROOT if root is None else Path(root).resolve()
    try:
        plan = build_training(args, root) if args.command == 'train' else build_evaluation(args, root)
        for note in plan.notes:
            print(f'# {note}')
        print(f'# {"DRY RUN - no work will execute" if args.dry_run else "Plan"}: {len(plan.jobs)} command(s).')
        for job in plan.jobs:
            print(f'# {job.label}')
            print(subprocess.list2cmdline(job.command) if os.name == 'nt' else shlex.join(job.command))
        if args.dry_run:
            return 0
        preflight(plan, args, root)
        for job in plan.jobs:
            print(f'\n{job.label}', flush=True)
            subprocess.run(job.command, cwd=root, check=True)
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    except subprocess.CalledProcessError as exc:
        print(f'Command stopped with exit code {exc.returncode}; remaining commands were not launched.', file=sys.stderr)
        return exc.returncode or 1
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
