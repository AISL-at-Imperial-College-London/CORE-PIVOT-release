"""Train the paper kernels on the prepared, fixed offline replay.

Example: python scripts/train.py --algorithm iql --data-dir data/paper --seed 123
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from datasets.replay import FIELDS, ReplayBuffer
from training.factory import (
    LEARNERS, configure, cpu, fresh, global_rng, load_initial,
    proposal_from_checkpoint, restore_global_rng, restore_native,
)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def save(learner, sampler, output, run_hash):
    folder = output / 'checkpoints'
    folder.mkdir(exist_ok=True)
    path = folder / f'step{learner.step:09d}.pt'
    state_path = path.with_suffix('.state.pt')
    for target, state in ((path, cpu(learner.state_dict())), (state_path, dict(
            step=learner.step, replay_rng=sampler.get_state(), global_rng=global_rng()))):
        with target.open('xb') as stream:
            torch.save(state, stream)
    write_json(path.with_suffix('.json'), dict(step=learner.step,
        model_sha256=sha(path), state_sha256=sha(state_path), run_sha256=run_hash))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', choices=LEARNERS, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path, help='JSON with algorithm/config/steps/checkpoint_every')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=JSON',
                        help='Override one learner hyperparameter, e.g. conservative_weight=0')
    parser.add_argument('--steps', type=int, help='Total learner updates, default 500000')
    parser.add_argument('--checkpoint-every', type=int)
    parser.add_argument('--log-every', type=int, default=1000)
    parser.add_argument('--bc-checkpoint', type=Path, help='CORE: shared seed-123 BC500k checkpoint')
    parser.add_argument('--initial-checkpoint', type=Path, help='Optional archived zero-step initial state')
    parser.add_argument('--resume', type=Path, help='Resume a native checkpoint from this output folder')
    parser.add_argument('--smoke-data', action='store_true', help='Permit a small replay; not a paper run')
    args = parser.parse_args()
    if args.resume and args.initial_checkpoint:
        parser.error('--resume and --initial-checkpoint are mutually exclusive')
    spec = json.loads((args.config or ROOT / 'configs/train' / f'{args.algorithm}.json').read_text())
    if spec['algorithm'] != args.algorithm:
        parser.error('Configuration algorithm differs')
    config = spec['config']
    for override in args.set:
        name, value = override.split('=', 1)
        if name not in config:
            parser.error(f'Unknown learner option: {name}')
        config[name] = json.loads(value)
    steps = args.steps if args.steps is not None else spec['steps']
    frequency = args.checkpoint_every or spec['checkpoint_every']
    if steps < 1 or frequency < 1 or args.log_every < 1:
        parser.error('Steps and checkpoint interval must be positive')
    configure(args.device)
    manifest_path = args.data_dir / 'manifest.json'
    normalization_path = args.data_dir / 'normalization.json'
    manifest = json.loads(manifest_path.read_text())
    normalization = json.loads(normalization_path.read_text())
    arrays = {name: np.load(args.data_dir / f'{name}.npy', mmap_mode='r', allow_pickle=False)
              for name in FIELDS}
    replay = ReplayBuffer(arrays, manifest)
    if len(replay) != 24000000 and not args.smoke_data:
        parser.error('Paper training requires 24 million transitions; use --smoke-data only for tests')
    # Data preparation writes the historical augmented view. Never add noise here.
    proposal = None
    if args.algorithm == 'core-pivot':
        if not args.bc_checkpoint:
            parser.error('CORE-PIVOT requires --bc-checkpoint (shared seed-123 BC500k)')
        proposal = proposal_from_checkpoint(args.bc_checkpoint, density=True)
    elif args.bc_checkpoint:
        parser.error('Only CORE training consumes a BC proposal; Q_beta fitting does not')
    learner, sampler = fresh(args.algorithm, config, normalization['mean'], normalization['scale'],
                             args.seed, args.device, proposal)
    if args.initial_checkpoint:
        load_initial(learner, sampler, args.initial_checkpoint)
    output = args.output or ROOT / 'runs' / args.algorithm / f'seed{args.seed}'
    identity = dict(algorithm=args.algorithm, config=config, seed=args.seed,
                    data_manifest_sha256=sha(manifest_path), normalization_sha256=sha(normalization_path),
                    bc_checkpoint_sha256=sha(args.bc_checkpoint) if args.bc_checkpoint else None,
                    initial_checkpoint_sha256=sha(args.initial_checkpoint) if args.initial_checkpoint else None,
                    replay_sampler_seed=args.seed + (20000 if args.algorithm == 'qbeta' else 10000),
                    smoke_only=args.smoke_data)
    if args.resume:
        if args.resume.resolve().parent.parent != output.resolve():
            parser.error('Resume checkpoint must belong to --output')
        old_run = json.loads((output / 'run.json').read_text())
        # Initial-state provenance is inherited, not reconstructed at resume.
        identity['initial_checkpoint_sha256'] = old_run['identity']['initial_checkpoint_sha256']
        if old_run['identity'] != identity:
            parser.error('Resume configuration/data/seed/proposal identity differs')
        commitment = json.loads(args.resume.with_suffix('.json').read_text())
        state_path = args.resume.with_suffix('.state.pt')
        if sha(args.resume) != commitment['model_sha256'] or sha(state_path) != commitment['state_sha256']:
            parser.error('Checkpoint commitment failed')
        restore_native(learner, torch.load(args.resume, map_location='cpu', weights_only=True))
        aux = torch.load(state_path, map_location='cpu', weights_only=True)
        sampler.set_state(aux['replay_rng'])
        restore_global_rng(aux['global_rng'])
    else:
        output.mkdir(parents=True, exist_ok=False)
        kernels = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'algorithms').rglob('*.py'))}
        kernels['core_pivot_truncated_20261019.py'] = sha(ROOT / 'core_pivot_truncated_20261019.py')
        write_json(output / 'run.json', dict(identity=identity, requested_updates=steps,
            checkpoint_every=frequency, kernel_sha256=kernels,
            runtime=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                         device=args.device, gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None),
            historical_bitwise_reproduction_claimed=False,
            note='Numerical kernels and sampler protocol retained; exact hardware/runtime agreement is separate.'))
    run_hash = sha(output / 'run.json')
    if learner.step == 0 and not args.resume:
        save(learner, sampler, output, run_hash)
    if steps <= learner.step:
        parser.error('Requested total steps must exceed the restored step')
    start = time.monotonic()
    with (output / 'training.jsonl').open('a', encoding='utf-8') as log:
        for step in range(learner.step + 1, steps + 1):
            metrics = learner.update(replay.sample(config['batch_size'], sampler, device=args.device))
            if learner.step != step or not np.isfinite(list(metrics.values())).all():
                raise RuntimeError('Invalid update counter or nonfinite training metrics')
            if step == 1 or step % args.log_every == 0 or step == steps:
                record = dict(metrics, elapsed_seconds=time.monotonic() - start)
                log.write(json.dumps(record, allow_nan=False) + '\n')
                log.flush()
                print(json.dumps(record), flush=True)
            if step % frequency == 0 or step == steps:
                save(learner, sampler, output, run_hash)
    write_json(output / f'completed_step{steps:09d}.json', dict(training_complete=True,
        step=steps, elapsed_seconds=time.monotonic() - start, validation_performed=False,
        note='Evaluate saved checkpoints separately; no test-based checkpoint selection.'))


if __name__ == '__main__':
    main()
