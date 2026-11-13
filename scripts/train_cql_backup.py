"""Paired actor-action versus recorded-action Bellman backups, same CQL actor.

Run each arm with the same seed and data; a fresh output directory is mandatory.
Both arms consume identical random draws, including the unused actor successor
draw in the SARSA arm. This is not a warm-start from a selected trained model.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from ablations.cql_backup import BackupCQL
from algorithms.corl_baselines import CORLCQLConfig
from datasets.replay import ReplayBuffer, FIELDS
from training.factory import configure, restore_native, restore_global_rng
from scripts.train import sha, save, write_json


def runtime_record(device):
    """Record numerical execution settings so a paired run can be checked."""
    return dict(python=platform.python_version(), torch=str(torch.__version__),
        numpy=np.__version__, platform=platform.system(), device=device,
        gpu=torch.cuda.get_device_name() if device == 'cuda' else None,
        gpu_capability=list(torch.cuda.get_device_capability()) if device == 'cuda' else None,
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        allow_tf32_cudnn=torch.backends.cudnn.allow_tf32, torch_threads=torch.get_num_threads())


def initialize(normalization, config, seed, backup, device='cpu', initial=None):
    learner = BackupCQL.__new__(BackupCQL)
    learner._initialize(CORLCQLConfig(**config), normalization['mean'],
                        normalization['scale'], seed, device, None)
    sampler = torch.Generator().manual_seed(seed + 10000)
    if initial is not None:
        record = torch.load(initial, map_location='cpu', weights_only=True)
        state = record.get('learner', record)
        if state['step'] != 0 or record.get('seed', seed) != seed:
            raise ValueError('An archival initial state must be step zero with the same seed.')
        if record.get('variant', 'cql_actor') not in ('cql_actor', 'cql_sarsa'):
            raise ValueError('This runner accepts CQL initial states, not CORE-PIVOT states.')
        if not (np.array_equal(state['mean'], normalization['mean']) and
                np.array_equal(state['scale'], normalization['scale'])):
            raise ValueError('Initial-state normalization differs from the requested data.')
        if any(optimizer['state'] for optimizer in state['optimizers'].values()):
            raise ValueError('A fresh archival initial state must have empty optimizer histories.')
        restore_native(learner, state)
        if 'sampler' in record:
            sampler.set_state(record['sampler'])
        elif 'auxiliary' in record:
            sampler.set_state(record['auxiliary']['replay_rng'])
        if not torch.equal(sampler.get_state(), torch.Generator().manual_seed(seed + 10000).get_state()):
            raise ValueError('Initial-state replay stream differs from the paired sampling protocol.')
    learner.backup = backup
    return learner, sampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup', choices=('actor', 'sarsa'), required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial-checkpoint', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--steps', type=int, default=500000)
    parser.add_argument('--checkpoint-every', type=int, default=10000)
    parser.add_argument('--log-every', type=int, default=1000)
    parser.add_argument('--smoke-data', action='store_true')
    args = parser.parse_args()
    if args.initial_checkpoint and args.resume:
        parser.error('Initial and resume checkpoints are mutually exclusive.')
    if min(args.steps, args.checkpoint_every, args.log_every) < 1:
        parser.error('Update counts must be positive.')
    configure(args.device)
    spec = json.loads((ROOT / 'configs' / f'backup_{args.backup}.json').read_text())
    norm = json.loads((args.data_dir / 'normalization.json').read_text())
    manifest = json.loads((args.data_dir / 'manifest.json').read_text())
    replay = ReplayBuffer({name: np.load(args.data_dir / f'{name}.npy', mmap_mode='r', allow_pickle=False)
                           for name in FIELDS}, manifest)
    if len(replay) != 24000000 and not args.smoke_data:
        parser.error('Expected full paper data; --smoke-data is testing only.')
    augmentation = manifest.get('previous_action_augmentation', {})
    if not args.smoke_data and (augmentation.get('protocol') != 'state_consistent_previous_action_v1'
                               or augmentation.get('std_data_scaled') != .01):
        parser.error('Paper backup fits require the fixed historical .01 augmented replay.')
    learner, sampler = initialize(norm, spec['config'], args.seed, args.backup,
                                  args.device, args.initial_checkpoint)
    identity = dict(algorithm='cql-backup', backup=args.backup, seed=args.seed,
        config=spec['config'], data_manifest_sha256=sha(args.data_dir / 'manifest.json'),
        normalization_sha256=sha(args.data_dir / 'normalization.json'),
        initial_sha256=sha(args.initial_checkpoint) if args.initial_checkpoint else None,
        sampler_seed=args.seed + 10000, per_update_torch_seed='seed*1000003+step',
        actor_entropy=.1, target_entropy=False, smoke_only=args.smoke_data)
    output = args.output
    if args.resume:
        if args.resume.resolve().parent.parent != output.resolve():
            parser.error('Resume checkpoint must belong to the output directory.')
        old = json.loads((output / 'run.json').read_text())
        current_runtime = runtime_record(args.device)
        if any(current_runtime.get(key) != value for key, value in old['runtime'].items()):
            parser.error('Resume runtime differs; preserve this run and use its original runtime.')
        identity['initial_sha256'] = old['identity']['initial_sha256']
        if old['identity'] != identity:
            parser.error('Resume identity mismatch.')
        commitment = json.loads(args.resume.with_suffix('.json').read_text())
        auxpath = args.resume.with_suffix('.state.pt')
        if commitment['run_sha256'] != sha(output / 'run.json'):
            parser.error('Checkpoint belongs to a different run manifest.')
        if any(sha(ROOT / name) != digest for name, digest in old['kernel_sha256'].items()):
            parser.error('Backup kernels changed since this run was started.')
        if sha(args.resume) != commitment['model_sha256'] or sha(auxpath) != commitment['state_sha256']:
            parser.error('Checkpoint hash mismatch.')
        learner = BackupCQL.load(args.resume, args.device)
        if learner.backup != args.backup or learner.seed != args.seed:
            parser.error('Restored backup/seed mismatch.')
        aux = torch.load(auxpath, map_location='cpu', weights_only=True)
        sampler.set_state(aux['replay_rng'])
        restore_global_rng(aux['global_rng'])
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'run.json', dict(identity=identity, requested_updates=args.steps,
            checkpoint_every=args.checkpoint_every, runtime=runtime_record(args.device),
            kernel_sha256={name: sha(ROOT / name) for name in
                ('ablations/cql_backup.py', 'algorithms/corl_vendor/cql.py', 'algorithms/corl_baselines.py')},
            historical_cuda_bitwise_reproduction_claimed=False))
        save(learner, sampler, output, sha(output / 'run.json'))
    if args.steps <= learner.step:
        parser.error('Total updates must exceed the restored step.')
    started = time.monotonic()
    with (output / 'training.jsonl').open('a') as log:
        while learner.step < args.steps:
            metrics = learner.update(replay.sample(spec['config']['batch_size'], sampler, device=args.device))
            if not np.isfinite(list(metrics.values())).all():
                raise RuntimeError('Nonfinite training metrics.')
            if learner.step == 1 or learner.step % args.log_every == 0 or learner.step == args.steps:
                row = dict(metrics, elapsed_seconds=time.monotonic() - started)
                log.write(json.dumps(row, allow_nan=False) + '\n'); log.flush()
                print(json.dumps(row), flush=True)
            if learner.step % args.checkpoint_every == 0 or learner.step == args.steps:
                save(learner, sampler, output, sha(output / 'run.json'))
    write_json(output / f'completed_step{learner.step:09d}.json', dict(training_complete=True,
        step=learner.step, elapsed_seconds=time.monotonic() - started, validation_performed=False))


if __name__ == '__main__':
    main()
