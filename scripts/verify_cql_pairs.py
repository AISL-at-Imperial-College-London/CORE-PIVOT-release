"""Fail closed unless saved actor/SARSA runs form a matched five-seed ablation.

This verifies initial tensors, settings, runtime, checkpoint commitments and
random streams. It does not require trained critics/actors to remain equal:
their divergence is an outcome of changing the Bellman successor action.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.train import sha, write_json
from scripts.verify_backup_parity import assert_equal


def load_committed(folder, step):
    checkpoint = folder / 'checkpoints' / f'step{step:09d}.pt'
    auxiliary = checkpoint.with_suffix('.state.pt')
    proof = json.loads(checkpoint.with_suffix('.json').read_text())
    if (proof['step'] != step or proof['model_sha256'] != sha(checkpoint)
            or proof['state_sha256'] != sha(auxiliary)
            or proof['run_sha256'] != sha(folder / 'run.json')):
        raise ValueError(f'Invalid checkpoint commitment: {checkpoint}')
    model = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = torch.load(auxiliary, map_location='cpu', weights_only=True)
    if model['step'] != step or state['step'] != step:
        raise ValueError('Checkpoint counter mismatch.')
    return model, state


def verify_pair(actor_folder, sarsa_folder, seed, steps=500000, checkpoint_every=10000):
    folders = [Path(actor_folder), Path(sarsa_folder)]
    runs = [json.loads((folder / 'run.json').read_text()) for folder in folders]
    expected_config = json.loads((ROOT / 'configs/backup_actor.json').read_text())['config']
    identities = []
    for folder, run, arm in zip(folders, runs, ('actor', 'sarsa')):
        identity = dict(run['identity'])
        if (identity.pop('backup') != arm or identity['seed'] != seed
                or identity['algorithm'] != 'cql-backup' or identity['config'] != expected_config
                or identity['actor_entropy'] != .1 or identity['target_entropy'] is not False
                or identity['sampler_seed'] != seed + 10000
                or identity['per_update_torch_seed'] != 'seed*1000003+step'):
            raise ValueError(f'Wrong paired-run identity: {folder}')
        identities.append(identity)
        completed = json.loads((folder / f'completed_step{steps:09d}.json').read_text())
        if not completed['training_complete'] or completed['step'] != steps:
            raise ValueError(f'Incomplete run: {folder}')
        for name, digest in run['kernel_sha256'].items():
            if sha(ROOT / name) != digest:
                raise ValueError(f'Run code differs from this checkout: {name}')
    if identities[0] != identities[1] or runs[0]['kernel_sha256'] != runs[1]['kernel_sha256']:
        raise ValueError('Pair differs in settings, data, initial source or kernels.')
    if runs[0]['runtime'] != runs[1]['runtime']:
        raise ValueError('Pair has different numerical runtimes or GPU models.')
    checkpoints = list(range(0, steps + 1, checkpoint_every))
    if checkpoints[-1] != steps:
        checkpoints.append(steps)
    for step in checkpoints:
        (actor, a_aux), (sarsa, s_aux) = [load_committed(folder, step) for folder in folders]
        if actor['backup'] != 'actor' or sarsa['backup'] != 'sarsa':
            raise ValueError('Checkpoint has the wrong backup arm.')
        if actor['seed'] != seed or sarsa['seed'] != seed:
            raise ValueError('Checkpoint has the wrong training seed.')
        assert_equal(a_aux['replay_rng'], s_aux['replay_rng'])
        assert_equal(a_aux['global_rng']['torch'], s_aux['global_rng']['torch'])
        assert_equal(a_aux['global_rng']['cuda'], s_aux['global_rng']['cuda'])
        if step == 0:
            actor = {key: value for key, value in actor.items() if key != 'backup'}
            sarsa = {key: value for key, value in sarsa.items() if key != 'backup'}
            assert_equal(actor, sarsa)
    return dict(seed=seed, passed=True, checkpoints=len(checkpoints), updates=steps,
        identical_initial_state=True, matched_replay_and_torch_streams=True,
        runtime=runs[0]['runtime'], config=expected_config,
        smoke_only=identities[0]['smoke_only'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=500000)
    parser.add_argument('--checkpoint-every', type=int, default=10000)
    args = parser.parse_args()
    if min(args.steps, args.checkpoint_every) < 1:
        parser.error('Update counts must be positive.')
    rows = [verify_pair(args.runs / 'actor' / f'seed{seed}', args.runs / 'sarsa' / f'seed{seed}',
                        seed, args.steps, args.checkpoint_every) for seed in range(123, 128)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, dict(passed=True, pairs=rows, test_access=False,
        historical_cuda_bitwise_reproduction_claimed=False,
        note='Backup-only intervention; adaptive multipliers and trained policies may diverge.'))
    print(f'All five saved pairs passed: {args.output}')


if __name__ == '__main__':
    main()
