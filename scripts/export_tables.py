"""Export held-out test and robustness tables from the verified paper archive.

This redraws archived numerical results; it does not train or evaluate policies.
Blank fields denote unavailable statistics, not zero variability.
"""
import argparse
import csv
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.plot_paper import LABELS, ORDER, verify

SEEDS = tuple(range(123, 128))
FIELDS = ['method', 'label', 'condition', 'disturbance_family', 'amplitude_m_s2',
          'mean', 'sample_sd', 'minimum', 'maximum', 'variability',
          'n_training_seeds', 'n_episodes', *[f'seed_{seed}' for seed in SEEDS]]


def moments(values):
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError('Expected nonempty finite costs')
    return dict(mean=statistics.mean(values),
                sample_sd=statistics.stdev(values) if len(values) > 1 else None,
                minimum=min(values), maximum=max(values))


def held_out_rows(data):
    source = data['original_test']['rows']
    records = []
    for stronger in (False, True):
        for method in [name for name in ORDER if name != 'sarsa-cql']:
            rows = [row for row in source
                    if row['method'] == method and row['stronger'] == stronger]
            if not rows:
                raise ValueError(f'Missing held-out costs: {method}/{stronger}')
            record = dict(method=method, label=LABELS[method],
                          condition='test_stronger' if stronger else 'test_ordinary',
                          disturbance_family='random_walk', amplitude_m_s2=None,
                          n_episodes=len(rows))
            if method == 'pd11':
                values = [float(row['cost']) for row in rows]
                if len({row['wind_seed'] for row in rows}) != len(rows):
                    raise ValueError('Fixed PD costs must contain unique winds')
                record.update(n_training_seeds=0, variability='wind_realisations_fixed_PD')
            else:
                if {row['seed'] for row in rows} != set(SEEDS):
                    raise ValueError(f'Incomplete five-seed test group: {method}')
                groups = [[row for row in rows if row['seed'] == seed] for seed in SEEDS]
                winds = [tuple(sorted(row['wind_seed'] for row in group)) for group in groups]
                if len(set(winds)) != 1 or len(set(winds[0])) != len(winds[0]):
                    raise ValueError('Test seed groups need matched unique winds')
                values = [statistics.mean(float(row['cost']) for row in group) for group in groups]
                record.update(n_training_seeds=5, variability='wind_averaged_training_seed_means')
                record.update({f'seed_{seed}': value for seed, value in zip(SEEDS, values)})
            record.update(moments(values))
            records.append(record)
    return records


def robustness_rows(data):
    records = []
    conditions = data['robustness']['conditions']
    for condition in sorted(conditions, key=lambda name: (['severity', 'step', 'reversing'].index(name.split('_')[0]), float(name.split('_')[1]))):
        family, amplitude = condition.split('_')
        for method in ORDER:
            source = conditions[condition][method]
            record = dict(method=method, label=LABELS[method], condition=condition,
                          disturbance_family='random_walk' if family == 'severity' else family,
                          amplitude_m_s2=float(amplitude), n_episodes=None)
            if method == 'pd11':
                # Only the wind-averaged PD mean survives in this compact archive.
                record.update(mean=float(source['mean']), sample_sd=None,
                              minimum=None, maximum=None, n_training_seeds=0,
                              variability='wind_realisations_fixed_PD_dispersion_unavailable')
            else:
                seed_means = source['seed_means']
                if set(seed_means) != {str(seed) for seed in SEEDS}:
                    raise ValueError(f'Incomplete five-seed robustness group: {method}/{condition}')
                values = [float(seed_means[str(seed)]) for seed in SEEDS]
                stats = moments(values)
                if not math.isclose(stats['mean'], source['mean'], rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError('Archived robustness mean differs from seed means')
                record.update(stats, n_training_seeds=5,
                              variability='wind_averaged_training_seed_means')
                record.update({f'seed_{seed}': value for seed, value in zip(SEEDS, values)})
            records.append(record)
    return records


def export_tables(data, output):
    # Validate both tables before touching generated outputs.
    tables = {'held_out_test.csv': held_out_rows(data),
              'robustness_analysis.csv': robustness_rows(data)}
    tables['table1_selected.csv'] = tables['held_out_test.csv'] + [
        r for r in tables['robustness_analysis.csv']
        if r['amplitude_m_s2'] in (1.5, 2.5, 4.) and r['method'] != 'sarsa-cql']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, records in tables.items():
        with (output/name).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(records)
        print(f'{len(records)} rows written to {output/name}')
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'figures/reproduced')
    args = parser.parse_args()
    export_tables(verify(), args.output)
    from scripts.plot_supplement import export_trajectory_table
    export_trajectory_table(args.output)


if __name__ == '__main__':
    main()
