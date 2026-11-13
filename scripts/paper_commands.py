"""Print all 30 selected-setting training commands; never launch or submit jobs."""
import argparse
import json
import shlex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/train')
    parser.add_argument('--runs', default='runs')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--json', action='store_true', help='Argument lists and dependencies for a scheduler')
    args = parser.parse_args()
    jobs = []
    for algorithm in ['bc', 'qbeta', 'td3-bc', 'iql', 'cql', 'core-pivot']:
        for seed in range(123, 128):
            command = ['python', 'scripts/train.py', '--algorithm', algorithm,
                       '--data-dir', args.data, '--seed', str(seed), '--device', args.device,
                       '--output', f'{args.runs}/{algorithm}/seed{seed}']
            dependencies = []
            if algorithm == 'core-pivot':
                command += ['--bc-checkpoint', f'{args.runs}/bc/seed123/checkpoints/step000500000.pt']
                dependencies = ['bc_123']
            jobs.append(dict(id=f'{algorithm}_{seed}', dependencies=dependencies, command=command))
    if args.json:
        print(json.dumps(dict(execution=False, jobs=jobs), indent=2))
    else:
        print('# Command plan only. No jobs submitted or started.')
        for job in jobs:
            if job['dependencies']:
                print('# After: ' + ', '.join(job['dependencies']))
            print(shlex.join(job['command']))


if __name__ == '__main__':
    main()
