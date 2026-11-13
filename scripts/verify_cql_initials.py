"""Compare all five fresh paired initializations with retained historical files.

Historical initial checkpoints are optional external provenance, not bundled
weights or a dependency of fresh reproduction. This is a CPU initialization
check, not a claim that historical CUDA training has been rerun.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.train import sha, write_json
from scripts.train_cql_backup import initialize, runtime_record
from scripts.verify_backup_parity import assert_equal
from training.factory import configure, cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True,
                        help='Retained packet containing data/initial_cql_s123.pt ... s127.pt')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configure('cpu')
    normalization = json.loads((args.data_dir / 'normalization.json').read_text())
    config = json.loads((ROOT / 'configs/backup_actor.json').read_text())['config']
    rows = []
    for seed in range(123, 128):
        path = args.reference_root / 'data' / f'initial_cql_s{seed}.pt'
        archived = torch.load(path, map_location='cpu', weights_only=True)
        if archived['step'] != 0 or archived['seed'] != seed:
            raise ValueError('Historical initial-state identity mismatch.')
        for arm in ('actor', 'sarsa'):
            model, sampler = initialize(normalization, config, seed, arm)
            fresh = cpu(model.state_dict())
            for name in ('config', 'mean', 'scale', 'step', 'modules', 'optimizers', 'rng', 'scheduler'):
                assert_equal(fresh[name], archived['learner'][name])
            assert_equal(sampler.get_state(), archived['sampler'])
        rows.append(dict(seed=seed, both_arms_bitwise_equal_to_archived_initial=True,
            model_optimizer_rng_and_normalization_equal=True,
            initial_file_sha256=sha(path)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, dict(passed=True, seeds=rows, runtime=runtime_record('cpu'),
        normalization_sha256=sha(args.data_dir / 'normalization.json'),
        note='Fresh CPU initialization parity; no 500k training or cross-device equality claimed.'))
    print(f'All five historical initializations matched: {args.output}')


if __name__ == '__main__':
    main()
