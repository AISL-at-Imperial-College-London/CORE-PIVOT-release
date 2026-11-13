"""Print the complete paired CQL ablation plan; never execute or submit jobs."""
import argparse
import json
from pathlib import Path
import shlex


def build_plan(data='data/train', runs='runs/cql-paired', device='cuda'):
    jobs = []
    training_ids = []
    for backup in ('actor', 'sarsa'):
        for seed in range(123, 128):
            identifier = f'{backup}_{seed}'
            training_ids.append(identifier)
            jobs.append(dict(id=identifier, stage='train', dependencies=[], command=[
                'python', 'scripts/train_cql_backup.py', '--backup', backup,
                '--data-dir', data, '--seed', str(seed), '--device', device,
                '--steps', '500000', '--checkpoint-every', '10000',
                '--output', f'{runs}/{backup}/seed{seed}']))
    jobs.append(dict(id='verify_pairs', stage='audit', dependencies=training_ids, command=[
        'python', 'scripts/verify_cql_pairs.py', '--runs', runs,
        '--output', f'{runs}/pair_audit.json']))
    evaluation_ids = []
    for backup, method in (('actor', 'cql'), ('sarsa', 'sarsa-cql')):
        for seed in range(123, 128):
            for step in range(0, 500001, 10000):
                identifier = f'{backup}_{seed}_step{step}'
                evaluation_ids.append(identifier)
                jobs.append(dict(id=identifier, stage='evaluate', dependencies=['verify_pairs'], command=[
                    'python', 'scripts/evaluate.py', '--method', method, '--checkpoint',
                    f'{runs}/{backup}/seed{seed}/checkpoints/step{step:09d}.pt',
                    '--expected-step', str(step), '--seed', str(seed), '--split', 'validation',
                    '--output', f'{runs}/validation/{backup}_seed{seed}_step{step:09d}.json']))
    jobs.append(dict(id='summarize', stage='summary', dependencies=evaluation_ids, command=[
        'python', 'scripts/summarize_evaluations.py', '--input', f'{runs}/validation',
        '--output', f'{runs}/validation_summary.json', '--plot']))
    return dict(execution=False, training_runs=10, validation_tasks=510,
        validation_episodes=10200, test_access=False, jobs=jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/train')
    parser.add_argument('--runs', default='runs/cql-paired')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--stage', choices=('all', 'train', 'audit', 'evaluate', 'summary'), default='all')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    plan = build_plan(Path(args.data).as_posix(), Path(args.runs).as_posix(), args.device)
    if args.stage != 'all':
        plan['jobs'] = [job for job in plan['jobs'] if job['stage'] == args.stage]
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print('# Plan only: 10 fits, 510 validation tasks, 10,200 episodes; no test access.')
        print('# Keep each actor/SARSA seed pair on the same GPU model and numerical runtime.')
        for job in plan['jobs']:
            print(shlex.join(job['command']))


if __name__ == '__main__':
    main()
