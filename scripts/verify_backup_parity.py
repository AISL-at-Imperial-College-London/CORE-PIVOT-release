"""Compare portable CQL backup arms with an archived reference kernel on CPU.

This verifies implementation parity on recorded minibatches. It does not claim
bitwise agreement with historical CUDA training on a different machine.
"""
import argparse
import importlib.util
import inspect
import json
from pathlib import Path
import platform
import sys
import time
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from ablations.cql_backup import BackupContinuousCQL
from datasets.replay import ReplayBuffer, FIELDS
from scripts.train_cql_backup import initialize
from scripts.train import sha, write_json
from training.factory import configure, fresh, restore_native, cpu


def assert_equal(first, second):
    """Compare complete learner state without importing training infrastructure."""
    if isinstance(first, torch.Tensor):
        if not torch.equal(first, second):
            raise AssertionError('Tensor differs.')
    elif isinstance(first, dict):
        if first.keys() != second.keys():
            raise AssertionError('Dictionary keys differ.')
        for name in first:
            assert_equal(first[name], second[name])
    elif isinstance(first, (list, tuple)):
        if len(first) != len(second):
            raise AssertionError('Sequence lengths differ.')
        for a, b in zip(first, second):
            assert_equal(a, b)
    elif first != second:
        raise AssertionError('Scalar differs.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('A positive number of updates is required.')
    configure('cpu')
    reference_path = args.reference_root / 'algorithms/corl_vendor/cql.py'
    spec = importlib.util.spec_from_file_location('archived_backup_kernel', reference_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    # The isolated override is exactly the historical loss, not a rewrite.
    original_source = inspect.getsource(module.ContinuousCQL._q_loss)
    portable_source = inspect.getsource(BackupContinuousCQL._q_loss)
    if original_source != portable_source:
        raise ValueError('Portable and archived loss sources differ.')
    historical_lines = reference_path.read_text().splitlines()
    intervention = next(i for i, line in enumerate(historical_lines) if 'Controlled ablation' in line)
    del historical_lines[intervention:intervention + 4]
    if historical_lines != (ROOT / 'algorithms/corl_vendor/cql.py').read_text().splitlines():
        raise ValueError('The historical vendor differs by more than the four-line backup intervention.')
    config = json.loads((ROOT / 'configs/backup_actor.json').read_text())['config']
    norm = json.loads((args.data_dir / 'normalization.json').read_text())
    manifest = json.loads((args.data_dir / 'manifest.json').read_text())
    replay = ReplayBuffer({name: np.load(args.data_dir / f'{name}.npy', mmap_mode='r', allow_pickle=False)
                           for name in FIELDS}, manifest)
    initial_path = args.reference_root / 'data' / f'initial_cql_s{args.seed}.pt'
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    rows = []
    for backup in ('actor', 'sarsa'):
        started = time.monotonic()
        portable, p_rng = initialize(norm, config, args.seed, backup, initial=initial_path)
        reference, r_rng = fresh('cql', config, norm['mean'], norm['scale'], args.seed)
        restore_native(reference, initial['learner'])
        r_rng.set_state(initial['sampler'])
        reference.trainer._q_loss = MethodType(module.ContinuousCQL._q_loss, reference.trainer)
        for step in range(args.steps):
            pb, rb = replay.sample(config['batch_size'], p_rng), replay.sample(config['batch_size'], r_rng)
            assert_equal(pb, rb)
            torch.manual_seed(args.seed * 1000003 + step)
            reference.trainer.recorded_next_actions = rb['next_actions'] if backup == 'sarsa' else None
            expected = reference.update(rb)
            expected_rng = torch.get_rng_state().clone()
            actual = portable.update(pb)
            assert_equal(expected, actual)
            assert_equal(expected_rng, torch.get_rng_state())
            if (step + 1) % 100 == 0:
                print(f'{backup}: {step + 1}/{args.steps} exact updates', flush=True)
        for name in ('modules', 'optimizers', 'rng'):
            assert_equal(cpu(reference.state_dict()[name]), cpu(portable.state_dict()[name]))
        assert_equal(r_rng.get_state(), p_rng.get_state())
        rows.append(dict(backup=backup, updates=args.steps, metrics_bitwise_equal=True,
            model_optimizer_rng_bitwise_equal=True, replay_rng_equal=True,
            elapsed_seconds=time.monotonic() - started))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, dict(passed=True, seed=args.seed, rows=len(replay), arms=rows,
        config=config, actor_entropy=.1, target_entropy=False,
        source_loss_exact=True, vendor_diff_is_only_four_line_backup_intervention=True,
        reference_kernel_sha256=sha(reference_path),
        initial_state_sha256=sha(initial_path), data_manifest_sha256=sha(args.data_dir / 'manifest.json'),
        runtime=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__, device='cpu'),
        claim='Same-runtime archived-kernel equivalence, not historical CUDA bitwise replay.'))
    print(json.dumps(dict(passed=True, report=str(args.output)), indent=2))


if __name__ == '__main__':
    main()
