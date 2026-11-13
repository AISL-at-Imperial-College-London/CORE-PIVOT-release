"""Aggregate new evaluator outputs without mixing them with archived paper data.

python scripts/summarize_evaluations.py --input runs/evaluations --output runs/summary.json
Files may cover multiple methods/checkpoints. Five matched seeds are required
by default. Use --allow-partial for diagnostic runs, never paper statistics.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.identity import register_configuration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--plot', action='store_true', help='Plot matched validation checkpoint curves as SVG/PDF/PNG')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    grouped = defaultdict(dict)
    configurations = {}
    sources = []
    for path in sorted(args.input.rglob('*.json')):
        record = json.loads(path.read_text())
        if record.get('evaluation_batch') != 'new_reproduction_run_not_pooled_with_paper_exports':
            continue
        if not record.get('passed') or record.get('network_updates') != 0:
            raise ValueError(f'Failed or non-frozen evaluation: {path}')
        if record['partial_diagnostic'] and not args.allow_partial:
            raise ValueError('Partial wind evaluations require --allow-partial')
        register_configuration(record, configurations)
        if record['method'] != 'best-pd' and not record.get('seed_verified') and not args.allow_partial:
            raise ValueError('Seed provenance is unverified; retain run sidecars or use --allow-partial for diagnostics')
        sources.append(str(path.relative_to(args.input)))
        conditions = sorted({r['condition'] for r in record['rows']})
        for condition in conditions:
            rows = [r for r in record['rows'] if r['condition'] == condition]
            if not np.isfinite([r['episode_cost'] for r in rows]).all():
                raise ValueError('Nonfinite episode cost')
            key = (record['method'], record['checkpoint_step'], condition)
            seed = record['seed']
            if seed in grouped[key]:
                raise ValueError(f'Duplicate method/step/condition/seed: {key}, {seed}')
            winds = [(r['wind_seed'], r['wind_sha256']) for r in rows]
            if len(set(winds)) != len(winds):
                raise ValueError('Duplicate wind in one seed')
            grouped[key][seed] = dict(mean=float(np.mean([r['episode_cost'] for r in rows])),
                                     costs=[r['episode_cost'] for r in rows], winds=sorted(winds))
    if not grouped:
        raise ValueError('No compatible evaluation files found')
    summaries = []
    for (method, step, condition), seeds in sorted(grouped.items(), key=lambda item: str(item[0])):
        fixed_pd = method == 'best-pd'
        expected = {None} if fixed_pd else set(range(123, 128))
        if set(seeds) != expected and not args.allow_partial:
            raise ValueError(f'{method}/{step}/{condition}: incomplete five-seed group')
        wind_sets = {tuple(r['winds']) for r in seeds.values()}
        if len(wind_sets) != 1:
            raise ValueError(f'{method}/{step}/{condition}: wind realisations differ across seeds')
        values = next(iter(seeds.values()))['costs'] if fixed_pd else [r['mean'] for r in seeds.values()]
        summaries.append(dict(method=method, step=step, condition=condition,
            seed_means={str(seed): r['mean'] for seed, r in seeds.items()},
            mean=float(np.mean(values)), sample_sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
            minimum=float(np.min(values)), maximum=float(np.max(values)),
            variability='across winds of one fixed PD' if fixed_pd else 'across wind-averaged training seed means',
            matched_winds=len(next(iter(wind_sets)))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(dict(passed=True, partial_diagnostic=args.allow_partial,
                       batch='new_reproduction_evaluations_only', sources=sources,
                       configurations={method: json.loads(value) for method, value in configurations.items()},
                       summaries=summaries), stream, indent=2)
    if args.plot:
        plot_curves(summaries, args.output.parent / args.output.stem)
    print(f'{len(summaries)} summaries written to {args.output}')


def plot_curves(summaries, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from plot_paper import COLORS, LABELS, style
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    available = False
    for condition, ax in zip(['validation_ordinary', 'validation_stronger'], axes):
        style(ax)
        for method in sorted({r['method'] for r in summaries if r['condition'] == condition}):
            rows = sorted([r for r in summaries if r['condition'] == condition and r['method'] == method], key=lambda r: r['step'] or 0)
            if method == 'best-pd':
                ax.axhline(rows[0]['mean'], color=COLORS['pd11'], ls='--', label=LABELS['pd11'])
                continue
            available = True
            x = np.asarray([r['step'] for r in rows]) / 1000
            color = COLORS.get(method)
            ax.plot(x, [r['mean'] for r in rows], color=color, label=LABELS.get(method, method), lw=1.3)
            ax.fill_between(x, [r['minimum'] for r in rows], [r['maximum'] for r in rows], color=color, alpha=.13, lw=0)
        ax.set_title(condition.removeprefix('validation_').capitalize() + ' winds')
        ax.set_xlabel('Learner / extraction updates (×1000)')
    if not available:
        plt.close(fig)
        print('No learned validation curves present; summary was saved without a plot.')
        return
    axes[0].set_ylabel('Episode cost')
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    for extension in ['svg', 'pdf', 'png']:
        fig.savefig(output.with_suffix('.' + extension), bbox_inches='tight', dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    main()
