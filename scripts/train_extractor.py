"""Train only a fresh extraction policy; never update the source IQL values."""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from extraction.frozen_iql import ExtractionConfig, FrozenIQLExtractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Frozen IQL bundle (export_iql.py)')
    parser.add_argument('--data', type=Path, required=True, help='Prepared training replay directory')
    parser.add_argument('--out', type=Path, required=True, help='New output directory; never overwritten')
    parser.add_argument('--kind', choices=['awr', 'ddpg_bc'], required=True)
    parser.add_argument('--coefficient', type=float, help='Default: AWR beta=5, DDPG+BC weight=1')
    parser.add_argument('--seed', type=int, choices=range(123, 128), default=123)
    parser.add_argument('--updates', type=int, default=50000)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--smoke', action='store_true', help='Permit a small replay; never a paper reproduction')
    parser.add_argument('--resume', type=Path, help='Resume into a NEW directory, preserving prior artifacts')
    args = parser.parse_args()
    if args.updates < 1 or args.checkpoint_every < 1:
        parser.error('updates and checkpoint-every must be positive')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    observations = np.load(args.data / 'observations.npy', mmap_mode='r')
    actions = np.load(args.data / 'actions.npy', mmap_mode='r')
    if observations.shape[1:] != (25,) or actions.shape != (len(observations), 3):
        raise ValueError('Expected 25D observations and normalized 3D actions')
    if not args.smoke and len(observations) != 24000000:
        raise ValueError('Paper extraction requires all 24M training transitions; use --smoke for tests')
    source = torch.load(args.source, map_location='cpu', weights_only=True)
    coefficient = args.coefficient if args.coefficient is not None else (5. if args.kind == 'awr' else 1.)
    config = ExtractionConfig(seed=args.seed, kind=args.kind, coefficient=coefficient)
    fit = FrozenIQLExtractor(source, config, args.device)
    if args.resume:
        fit.restore(torch.load(args.resume, map_location=args.device, weights_only=True))
    else:
        fit.estimate_q_scale(observations, actions)
    if fit.step >= args.updates:
        raise ValueError('Requested endpoint is not beyond the resume checkpoint')
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoints = args.out / 'checkpoints'
    checkpoints.mkdir()
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    runtime = dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                   device=args.device, gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None,
                   source_sha256=source_hash, frozen_hash=fit.initial_frozen,
                   config=config.__dict__, smoke=args.smoke, rows=len(observations), qscale=fit.qscale,
                   source_step=source['step'], updates=args.updates, checkpoint_every=args.checkpoint_every,
                   critic_updates=0, value_updates=0, fixed_sigma=1., test_accessed=False)
    (args.out / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
    torch.save(fit.snapshot(), checkpoints / f'step{fit.step:06d}.pt')
    start = time.monotonic()
    with (args.out / 'metrics.jsonl').open('x') as log:
        while fit.step < args.updates:
            metrics = fit.update(*fit.batch(observations, actions))
            if fit.step % 100 == 0 or fit.step == args.updates:
                record = dict(step=fit.step, elapsed_seconds=time.monotonic() - start, **metrics)
                log.write(json.dumps(record, allow_nan=False) + '\n')
                log.flush()
                print(json.dumps(record), flush=True)
            if fit.step % args.checkpoint_every == 0 or fit.step == args.updates:
                fit.assert_frozen()
                torch.save(fit.snapshot(), checkpoints / f'step{fit.step:06d}.pt')
    fit.assert_frozen()
    report = dict(passed=True, updates=fit.step, critic_updates=0, value_updates=0,
                  elapsed_seconds=time.monotonic() - start, source_sha256=source_hash,
                  frozen_hash=fit.initial_frozen, smoke=args.smoke)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
