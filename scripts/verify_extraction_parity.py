"""Compare migrated extraction against an independently archived worker.

This optional developer audit needs the original experiment directory. It is
not a claim of bitwise replay of a historical GPU run. The portable smoke tests
do not require that private archive.
"""
import argparse
import ast
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from algorithms.corl_vendor.iql import GaussianPolicy, TwinQ, ValueFunction
from extraction.frozen_iql import ExtractionConfig, FrozenIQLExtractor, module_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--updates', type=int, default=1000)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source_file = args.archive / 'experiment.py'
    manifest = json.loads((args.archive / 'manifest.json').read_text())
    tree = ast.parse(source_file.read_text())
    kept = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and node.name in ('Fit', 'statehash')]
    namespace = dict(np=np, torch=torch, copy=copy, hashlib=hashlib,
                     GaussianPolicy=GaussianPolicy, TwinQ=TwinQ, ValueFunction=ValueFunction,
                     R=args.archive, M=manifest,
                     sha=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest())
    exec(compile(ast.Module(body=kept, type_ignores=[]), str(source_file), 'exec'), namespace)
    obs = np.load(args.data / 'observations.npy', mmap_mode='r')
    actions = np.load(args.data / 'actions.npy', mmap_mode='r')
    model = args.archive / manifest['models']['123']['path']
    source = torch.load(model, map_location='cpu', weights_only=True)
    results = []
    for kind, coefficient in [('awr', 5.), ('ddpg_bc', 1.)]:
        config = ExtractionConfig(kind=kind, coefficient=coefficient)
        portable = FrozenIQLExtractor(source, config)
        reference = namespace['Fit'](dict(seed=123, kind=kind, coefficient=coefficient))
        assert module_hash({'actor': portable.actor}) == module_hash({'actor': reference.actor})
        portable.estimate_q_scale(obs, actions)
        reference.qscale = portable.qscale
        start = time.monotonic()
        for _ in range(args.updates):
            batch = portable.batch(obs, actions)
            other = reference.batch(obs, actions)
            assert all(torch.equal(a, b) for a, b in zip(batch, other))
            got, expected = portable.update(*batch), reference.update(*other)
            assert got == expected, (kind, portable.step, got, expected)
            assert module_hash({'actor': portable.actor}) == module_hash({'actor': reference.actor})
        portable.assert_frozen()
        results.append(dict(kind=kind, updates=args.updates, exact_metrics=True, exact_actor=True,
                            shared_batches=True, frozen_values=True,
                            elapsed_seconds=time.monotonic() - start,
                            actor_sha256=module_hash({'actor': portable.actor})))
        print(kind, args.updates, 'updates match exactly', flush=True)
    report = dict(passed=True, scope='CPU implementation migration parity, not historical CUDA replay',
                  source_worker_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
                  source_model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                  torch=str(torch.__version__), numpy=np.__version__, results=results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
