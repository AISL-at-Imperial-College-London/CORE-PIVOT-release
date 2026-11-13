"""Evaluate frozen policies on a named split or robustness condition (CPU only).

Examples:
  python scripts/evaluate.py --method best-pd --split validation --output runs/pd.json
  python scripts/evaluate.py --method hoflon --checkpoint runs/qbeta/step000310000.pt \
      --bc-checkpoint runs/bc/step000500000.pt --seed 123 --split validation --output runs/hf.json
  python scripts/evaluate.py --method core-pivot --checkpoint model.pt --seed 123 \
      --condition reversing_3 --trajectories --output runs/core_reversal.json

Use --split test --allow-test explicitly only after settings/checkpoints are frozen.
Outputs are exclusive: existing files are never overwritten.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '1')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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
import scipy
from evaluation.protocol import robustness_conditions
from evaluation.policies import load_model, episode_policy
from evaluation.identity import checkpoint_identity
from evaluation.rollout import rollout


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['bc', 'easybcq', 'hoflon', 'core-pivot', 'td3-bc', 'iql', 'cql', 'sarsa-cql', 'awr', 'ddpg-bc', 'best-pd'], required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--bc-checkpoint', type=Path)
    parser.add_argument('--source-critic', type=Path, help='export_iql.py bundle used to fit AWR/DDPG+BC')
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--expected-step', type=int)
    parser.add_argument('--seed', type=int, default=123, help='Training seed metadata; checkpoint identity must be verified separately')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--split', choices=['validation', 'test'], default='validation')
    group.add_argument('--condition', choices=[c['id'] for c in robustness_conditions()] + ['all'])
    parser.add_argument('--allow-test', action='store_true')
    parser.add_argument('--winds', type=int, help='Diagnostic prefix only; omitted means the complete condition')
    parser.add_argument('--candidates', type=int, default=100, help='EasyBCQ/HOFLON candidate count; CORE uses saved config')
    parser.add_argument('--radius', default='0.1', help='HOFLON normalised radius or box; ignored by other methods')
    parser.add_argument('--iterations', type=int, default=40, help='HOFLON SLSQP budget; CORE uses saved config')
    parser.add_argument('--trajectories', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.split == 'test' and not args.allow_test:
        parser.error('Test evaluation requires explicit --allow-test after freezing the selection')
    if args.method != 'best-pd' and not args.checkpoint:
        parser.error('--checkpoint is required for learned methods')
    if args.output.exists():
        raise FileExistsError(args.output)
    radius = None if args.radius == 'box' else float(args.radius)
    if args.candidates < 1 or args.iterations < 1 or (radius is not None and radius < 0):
        parser.error('Search budget and radius must be valid')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = None
    policy_identity = dict(policy_config={}, checkpoint_seed=None,
                           seed_verified=False, seed_evidence='fixed PD; no training seed')
    identities = {}
    for name in ('checkpoint', 'bc_checkpoint', 'source_critic'):
        path = getattr(args, name)
        if path:
            identities[name] = {'file': path.name, 'sha256': sha(path)}
    if args.checkpoint_sha256 and identities['checkpoint']['sha256'] != args.checkpoint_sha256:
        raise ValueError('Checkpoint SHA256 mismatch')
    if args.method != 'best-pd':
        policy_identity = checkpoint_identity(args.checkpoint, args.seed)
        model = load_model(args.method, args.checkpoint, args.bc_checkpoint, args.expected_step, args.source_critic)
        if args.method in ('awr', 'ddpg-bc') and model.fit.config.seed != args.seed:
            raise ValueError('--seed does not match the extraction checkpoint')
    split = json.loads((ROOT / 'configs/dataset_split.json').read_text())['splits']
    if args.condition:
        seeds = split['validation']['ordinary'] + split['validation']['stronger']
        cases = [(c, seeds) for c in robustness_conditions() if args.condition == 'all' or c['id'] == args.condition]
    else:
        cases = [(dict(id=args.split + '_' + name, family='split', stronger=stronger), split[args.split][name])
                 for name, stronger in [('ordinary', False), ('stronger', True)]]
    if args.winds is not None:
        if args.winds < 1:
            parser.error('--winds must be positive')
        cases = [(c, seeds[:args.winds]) for c, seeds in cases]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows, start = [], time.monotonic()
    for condition, seeds in cases:
        for wind_seed in seeds:
            policy = None if model is None else episode_policy(model, args.method, wind_seed, args.candidates, radius, args.iterations)
            row, data = rollout(policy, condition, wind_seed, record=args.trajectories)
            if data is not None:
                folder = args.output.parent / (args.output.stem + '_trajectories')
                folder.mkdir(exist_ok=True)
                path = folder / f"{condition['id']}_wind{wind_seed}.npz"
                with path.open('xb') as stream:
                    np.savez_compressed(stream, **data)
                row['trajectory'] = str(path.relative_to(args.output.parent))
                row['trajectory_sha256'] = sha(path)
            rows.append(row)
            print(f"{condition['id']} wind {wind_seed}: {row['episode_cost']:.6g}", flush=True)
    search = None
    if args.method in ('easybcq', 'hoflon'):
        search = dict(candidates=args.candidates, radius=0 if args.method == 'easybcq' else radius,
                      iterations=0 if args.method == 'easybcq' else args.iterations,
                      rng='checkpoint state; reset each episode')
    elif args.method == 'core-pivot':
        search = dict(saved_config=vars(model.config), constraint='actuator box only',
                      rng='9001 + wind_seed; reset each episode')
    result = dict(passed=True, method=args.method, seed=None if model is None else args.seed,
        checkpoint_step=None if model is None else model.step, inputs=identities,
        conditions=[c for c, _ in cases], rows=rows, network_updates=0,
        evaluation_batch='new_reproduction_run_not_pooled_with_paper_exports',
        partial_diagnostic=args.winds is not None, held_out_test_accessed=args.split == 'test' and not args.condition,
        search=search, **policy_identity,
        runtime=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                     scipy=scipy.__version__, device='cpu', seconds=time.monotonic() - start),
        reproducibility_note='Original HPC solver runtime: Python3.12.3, torch2.9.1+cu130, numpy2.5.1, scipy1.18.0. Cross-platform SLSQP equality is not guaranteed.')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
