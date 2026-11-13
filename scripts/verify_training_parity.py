"""Compare migrated numerical kernels with an independently imported archive.

Uses two fresh processes, identical initial seeds and paired minibatches. This
is a same-runtime migration check, NOT a claim of historical GPU bitwise replay.
An archive is deliberately external to the release, avoiding duplicate kernels.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def worker(args):
    # Only the requested source tree can supply algorithm/training imports.
    sys.path.insert(0, str(args.source_root.resolve()))
    import numpy as np
    import torch
    from algorithms.corl_baselines import CORLCQL, CORLCQLConfig, CORLIQL, CORLIQLConfig, CORLTD3BC, CORLTD3BCConfig
    from algorithms.truncated_bc import TruncatedBC, TruncatedBCConfig
    from algorithms.qbeta_recorded_action_20261017 import RecordedActionQBeta
    from algorithms.one_step_easy_bcq import OneStepEasyBCQConfig
    from algorithms.core_pivot import COREPIVOTConfig
    from core_pivot_truncated_20261019 import DensityProposal, TruncatedCOREPIVOT
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    classes = dict(bc=(TruncatedBC, TruncatedBCConfig), qbeta=(RecordedActionQBeta, OneStepEasyBCQConfig),
        **{'core-pivot': (TruncatedCOREPIVOT, COREPIVOTConfig), 'td3-bc': (CORLTD3BC, CORLTD3BCConfig)},
        iql=(CORLIQL, CORLIQLConfig), cql=(CORLCQL, CORLCQLConfig))
    spec = json.loads((ROOT / 'configs/train' / f'{args.algorithm}.json').read_text())
    if args.data_dir:
        norm = json.loads((args.data_dir / 'normalization.json').read_text())
        arrays = {name: np.load(args.data_dir / f'{name}.npy', mmap_mode='r', allow_pickle=False)
                  for name in ('observations', 'next_observations', 'actions', 'next_actions', 'rewards', 'terminals', 'timeouts')}
        mean, scale = norm['mean'], norm['scale']
    else:
        # Unit-test fixture only: never reported as performance evidence.
        rng = np.random.default_rng(90210)
        n = 4096
        arrays = {k: rng.normal(size=(n, 25)).astype('float32') for k in ('observations', 'next_observations')}
        arrays.update({k: rng.uniform(-.9, .9, (n, 3)).astype('float32') for k in ('actions', 'next_actions')})
        arrays.update(rewards=-rng.uniform(0, 2, (n, 1)).astype('float32'),
                      terminals=np.zeros((n, 1), bool), timeouts=np.zeros((n, 1), bool))
        arrays['terminals'][::750] = True
        mean, scale = np.zeros(25), np.ones(25)
    proposal = None
    if args.algorithm == 'core-pivot':
        if args.bc_checkpoint:
            saved = torch.load(args.bc_checkpoint, map_location='cpu', weights_only=True)
            saved = saved.get('trainer', saved)
            model = saved['modules']['policy'] if 'modules' in saved else saved['model']
            proposal = DensityProposal(model['normalize.mean'], model['normalize.scale'], saved['config']['hidden_size'])
            proposal.load_state_dict(model)
        else:
            if args.data_dir:
                raise ValueError('Real-replay CORE parity requires a fitted --bc-checkpoint')
            torch.manual_seed(7001)
            proposal = DensityProposal(mean, scale, 1024)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cls, cfg = classes[args.algorithm]
    learner = cls.__new__(cls)
    learner._initialize(cfg(**spec['config']), mean, scale, args.seed, 'cpu', proposal)
    sampler = torch.Generator().manual_seed(args.seed + (20000 if args.algorithm == 'qbeta' else 10000))
    initial = {k: v.clone() for k, v in learner.modules[next(iter(learner.modules))].state_dict().items()}
    logs = []
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        ids = torch.randint(len(arrays['observations']), (spec['config']['batch_size'],), generator=sampler).numpy()
        batch = {key: torch.from_numpy(np.array(value[ids], copy=True)) for key, value in arrays.items()}
        batch['bootstrap_mask'] = (~batch['terminals']).float()
        metrics = learner.update(batch)
        if not np.isfinite(list(metrics.values())).all():
            raise ValueError('Nonfinite training metric')
        logs.append(metrics)
    # Exact same-device save/load and next update, including targets and optimizers.
    output = args.output
    native = output.with_suffix('.native.pt')
    learner.save(native)
    numpy_state = np.random.get_state()
    boundary = dict(replay_rng=sampler.get_state().clone(), global_rng=dict(
        torch=torch.get_rng_state().clone(), python=random.getstate(),
        numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]), cuda=[]))
    before_rng = torch.get_rng_state().clone()
    restored = cls.load(native, device='cpu')
    torch.set_rng_state(before_rng)
    ids = torch.randint(len(arrays['observations']), (spec['config']['batch_size'],), generator=sampler).numpy()
    batch = {key: torch.from_numpy(np.array(value[ids], copy=True)) for key, value in arrays.items()}
    batch['bootstrap_mask'] = (~batch['terminals']).float()
    before_rng = torch.get_rng_state().clone()
    expected = learner.update(batch)
    next_rng = torch.get_rng_state().clone()
    torch.set_rng_state(before_rng)
    actual = restored.update(batch)
    equal(expected, actual)
    equal(learner.state_dict(), restored.state_dict())
    equal(torch.get_rng_state(), next_rng)
    with output.open('xb') as stream:
        torch.save(dict(state=learner.state_dict(), metrics=logs, boundary=boundary, replay_rng=sampler.get_state(),
                        global_rng=next_rng, initial_probe=initial, resume_exact=True), stream)
    print(json.dumps(dict(algorithm=args.algorithm, steps=args.steps, seconds=time.monotonic() - started,
                         source=str(args.source_root), resume_exact=True)), flush=True)


def equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        if not torch.equal(a, b):
            raise AssertionError('Tensor mismatch')
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b, (a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--bc-checkpoint', type=Path)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--verify-cli', action='store_true', help='Run the actual training CLI as the release arm (requires real replay)')
    parser.add_argument('--algorithms', nargs='+', default=['bc', 'qbeta', 'core-pivot', 'td3-bc', 'iql', 'cql'])
    parser.add_argument('--output', type=Path, default=ROOT / 'verification' / 'training_parity')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--source-root', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--algorithm', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('Steps must be positive')
    if args.worker:
        return worker(args)
    if not args.archive_root:
        parser.error('--archive-root must identify the independently retained historical source tree')
    if args.verify_cli and not args.data_dir:
        parser.error('--verify-cli requires --data-dir')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, steps=args.steps, seed=args.seed,
                  real_replay=args.data_dir is not None,
                  claim='Same-runtime numerical-kernel migration and next-update resume parity, not historical GPU bitwise reproduction',
                  records=[])
    import torch
    for algorithm in args.algorithms:
        outputs = []
        for name, root in [('archive', args.archive_root), ('release', ROOT)]:
            path = args.output / f'{algorithm}_{name}.pt'
            if name == 'release' and args.verify_cli:
                directory = args.output / f'{algorithm}_cli'
                command = [sys.executable, '-B', str(ROOT / 'scripts/train.py'), '--algorithm', algorithm,
                           '--data-dir', str(args.data_dir), '--steps', str(args.steps), '--seed', str(args.seed),
                           '--output', str(directory), '--log-every', '1']
                if algorithm == 'core-pivot':
                    command.extend(['--bc-checkpoint', str(args.bc_checkpoint)])
                # Preserve a complete log without flooding the terminal with 1000 lines.
                with (args.output / f'{algorithm}_cli.stdout.log').open('x') as stream:
                    subprocess.run(command, check=True, stdout=stream)
                reference = torch.load(outputs[0], map_location='cpu', weights_only=True)
                baseline = torch.load(outputs[0].with_suffix('.native.pt'), map_location='cpu', weights_only=True)
                actual_path = directory / 'checkpoints' / f'step{args.steps:09d}.pt'
                actual = torch.load(actual_path, map_location='cpu', weights_only=True)
                aux = torch.load(actual_path.with_suffix('.state.pt'), map_location='cpu', weights_only=True)
                equal(baseline, actual)
                equal(reference['boundary']['replay_rng'], aux['replay_rng'])
                equal(reference['boundary']['global_rng'], aux['global_rng'])
                logs = [json.loads(line) for line in (directory / 'training.jsonl').read_text().splitlines()]
                for log in logs:
                    del log['elapsed_seconds']
                equal(reference['metrics'], logs)
                resume_command = list(command)
                resume_command[resume_command.index('--steps') + 1] = str(args.steps + 1)
                resume_command.extend(['--resume', str(actual_path)])
                with (args.output / f'{algorithm}_cli.resume.stdout.log').open('x') as stream:
                    subprocess.run(resume_command, check=True, stdout=stream)
                resumed = torch.load(directory / 'checkpoints' / f'step{args.steps+1:09d}.pt', map_location='cpu', weights_only=True)
                equal(reference['state'], resumed)
                print(json.dumps(dict(algorithm=algorithm, actual_cli_exact=True, updates=args.steps)), flush=True)
                outputs.append(None)
                continue
            command = [sys.executable, '-B', __file__, '--worker', '--source-root', str(root),
                       '--algorithm', algorithm, '--steps', str(args.steps), '--seed', str(args.seed), '--output', str(path)]
            if args.data_dir:
                command.extend(['--data-dir', str(args.data_dir)])
            if args.bc_checkpoint:
                command.extend(['--bc-checkpoint', str(args.bc_checkpoint)])
            subprocess.run(command, check=True)
            outputs.append(path)
        if not args.verify_cli:
            a, b = [torch.load(p, map_location='cpu', weights_only=True) for p in outputs]
            equal(a, b)
        record = dict(algorithm=algorithm, updates=args.steps,
                      all_logged_metrics_exact=True, all_parameters_targets_optimizers_exact=True,
                      replay_private_global_rng_exact=True, exact_next_update_after_resume=True)
        record['actual_training_cli_verified'] = args.verify_cli
        report['records'].append(record)
        print(json.dumps(record), flush=True)
    report['passed'] = True
    with (args.output / 'report.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)


if __name__ == '__main__':
    main()
