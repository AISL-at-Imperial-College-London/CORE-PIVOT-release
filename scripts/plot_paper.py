"""Recreate compact scientific plots from bundled, hash-verified paper data.

No model checkpoints, training dataset, personal paths or new rollouts required.
python scripts/plot_paper.py --figure all --output figures/reproduced
SVG, PDF and PNG are emitted. Main plots are restyled from original numerical
arrays, not claimed to be pixel-identical copies of the paper's PDF artwork.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

COLORS = {'bc': '#777777', 'easybcq': '#0072B2', 'hoflon': '#8063AE',
          'core-pivot': '#008080', 'td3-bc': '#B8860B', 'iql': '#CC79A7',
          'cql': '#D55E00', 'sarsa-cql': '#49516F', 'pd11': '#333333'}
LABELS = {'bc': 'BC', 'easybcq': 'EasyBCQ', 'hoflon': 'HOFLON-EasyBCQ',
          'core-pivot': 'CORE-PIVOT', 'td3-bc': 'TD3-BC', 'iql': 'IQL',
          'cql': 'CQL', 'sarsa-cql': 'SARSA-CQL', 'pd11': 'Best PD baseline in training'}
ORDER = ['bc', 'easybcq', 'hoflon', 'core-pivot', 'td3-bc', 'iql', 'cql', 'sarsa-cql', 'pd11']


def verify():
    directory = ROOT / 'results/paper_data'
    proof = json.loads((directory / 'provenance.json').read_text())
    for name, expected in proof['generated_files'].items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Bundled-data hash mismatch: {name}')
    paired = proof['paired_cql_update']
    if hashlib.sha256((ROOT / paired['audit']).read_bytes()).hexdigest() != paired['audit_sha256']:
        raise ValueError('Paired CQL audit hash mismatch')
    return json.loads((directory / 'paper_results.json').read_text())


def band(ax, x, values, color, label=None, linestyle='-', marker=None):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(x) or not np.isfinite(values).all():
        raise ValueError('Expected seed-by-checkpoint finite costs')
    ax.plot(x, values.mean(0), color=color, label=label, lw=1.25, ls=linestyle,
            marker=marker, markersize=3, markerfacecolor='white' if marker else color)
    if len(values) > 1:
        ax.fill_between(x, values.min(0), values.max(0), color=color, alpha=.13, lw=0)


def style(ax, log=True):
    if log:
        ax.set_yscale('log')
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis='y', color='#bbbbbb', alpha=.28, lw=.45)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.3, length=2.5, pad=1.5)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)


def save(fig, output, name):
    for ext in ['svg', 'pdf', 'png']:
        fig.savefig(output / f'{name}.{ext}', dpi=230, bbox_inches='tight', pad_inches=.03)
    plt.close(fig)
    remove_legacy_outputs(output, name, ('svg', 'pdf', 'png'))


def remove_legacy_outputs(output, replacement, extensions):
    """Retire only known generated names, after every replacement exists."""
    legacy = {
        'held_out_test_performance': 'original_test_violins',
        'held_out_test_summary': 'original_test_summary',
        'robustness_analysis': 'robustness',
    }.get(replacement)
    if legacy is None:
        return
    if not all((output / f'{replacement}.{ext}').is_file() for ext in extensions):
        return
    for ext in extensions:
        path = output / f'{legacy}.{ext}'
        # Do not recurse, follow links, or touch similarly named user files.
        if path.is_file() and not path.is_symlink():
            path.unlink()
            print(f'Removed superseded generated output: {path}')


def learning(data, output, fullrange=False):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 2.65), sharex=True, sharey='row')
    groups = [('Candidate-based', ['EasyBCQ', 'HOFLON-EasyBCQ']),
              ('Multi-step improvement', ['TD3-BC', 'IQL', 'CQL']),
              ('Conservative one-step', ['CORE-PIVOT'])]
    keys = {'EasyBCQ': 'easybcq', 'HOFLON-EasyBCQ': 'hoflon', 'TD3-BC': 'td3-bc',
            'IQL': 'iql', 'CQL': 'cql', 'CORE-PIVOT': 'core-pivot'}
    for column, (title, names) in enumerate(groups):
        axes[0, column].set_title(title, fontweight='bold', fontsize=7.5)
        for condition in range(2):
            ax = axes[condition, column]
            style(ax)
            for name in names:
                curve = data['learning_curves'][name]
                band(ax, np.asarray(curve['steps']) / 1000, np.asarray(curve['costs'])[:, :, condition], COLORS[keys[name]], LABELS[keys[name]])
                if not fullrange:
                    cap = [400, 2000][condition]
                    above = np.asarray(curve['costs'])[:, :, condition].max(0) > cap
                    ax.scatter(np.asarray(curve['steps'])[above] / 1000, np.full(above.sum(), cap / 1.05), marker='^', s=9, color=COLORS[keys[name]], zorder=6)
            ax.axhline(data['pd_validation_means'][condition], color=COLORS['pd11'], ls='--', lw=.9)
            ax.set_xlim(0, 500)
            ax.set_xticks([0, 250, 500])
            if not fullrange:
                ax.set_ylim(*[(25, 400), (40, 2000)][condition])
            # Out-of-window seed maxima are marked, with an uncropped companion.
        axes[0, column].legend(fontsize=5.7, frameon=False, loc='upper right', handlelength=1.5)
    axes[0, 0].set_ylabel('Ordinary winds\nEpisode cost')
    axes[1, 0].set_ylabel('Stronger winds\nEpisode cost')
    fig.supxlabel('Training updates (×1000)', fontsize=8, y=.015)
    fig.subplots_adjust(left=.08, right=.99, bottom=.19, top=.88, wspace=.15, hspace=.16)
    save(fig, output, 'learning_curves_fullrange' if fullrange else 'learning_curves')
    if not fullrange:
        learning(data, output, fullrange=True)


def robustness(data, output):
    from evaluation.protocol import disturbance
    with np.load(ROOT / 'results/paper_data/benchmark_data.npz') as z:
        train = z['training_winds']
    split = json.loads((ROOT / 'configs/dataset_split.json').read_text())['splits']['validation']
    seeds = split['ordinary'] + split['stronger']
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.0), gridspec_kw={'height_ratios': [.8, 1]})
    time = np.arange(750) * .02
    families = [('severity', 'Random walk'), ('step', 'Step gust'), ('reversing', 'Reversing gust')]
    for col, (family, title) in enumerate(families):
        ax = axes[0, col]
        style(ax, False)
        ax.set_title(title, fontsize=8, fontweight='bold')
        if family == 'severity':
            ax.axhspan(-1, 1, color='#999999', alpha=.16, lw=0)
            for trace in train:
                for k in range(3):
                    ax.plot(time, trace[:, k], color='#888888', alpha=.035, lw=.3)
        for k, color in enumerate(['#0072B2', '#D55E00', '#009E73']):
            for seed in seeds:
                wind = disturbance(dict(family=family, amplitude=2.5, onset=4., reversal=9.), seed)
                highlighted = seed == (51048 if family == 'severity' else 51091)
                ax.plot(time, wind[:, k], color=color, alpha=1 if highlighted else .13,
                        lw=1.05 if highlighted else .35)
        ax.set(xlim=(0, 15), ylim=(-2.7, 2.7), xticks=[0, 5, 10, 15], yticks=[-2.5, 0, 2.5])
        ax.set_xlabel('Time (s)', labelpad=1)
        lower = axes[1, col]
        style(lower)
        conditions = sorted([c for c in data['robustness']['conditions'] if c.startswith(family + '_')], key=lambda c: float(c.split('_')[1]))
        x = [float(c.split('_')[1]) for c in conditions]
        for method in ORDER:
            stats = [data['robustness']['conditions'][c][method] for c in conditions]
            y = [[s['mean'] for s in stats]] if method == 'pd11' else [[s['seed_means'][str(seed)] for s in stats] for seed in range(123, 128)]
            band(lower, x, y, COLORS[method], LABELS[method], '--' if method == 'pd11' else '-', 'o' if method == 'sarsa-cql' else None)
        lower.set_xlabel('Peak component A (m/s²)', labelpad=1)
        lower.set_xticks(x)
    axes[0, 0].set_ylabel('Wind (m/s²)')
    axes[1, 0].set_ylabel('Episode cost')
    handles = [Line2D([], [], color=c, label=l) for c, l in zip(['#0072B2', '#D55E00', '#009E73'], ['$w_x$', '$w_y$', '$w_z$'])]
    axes[0, 1].legend(handles=handles, ncol=3, loc='lower left', fontsize=6, frameon=False, handlelength=1.1, columnspacing=.6)
    h, l = axes[1, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc='lower center', frameon=False, fontsize=6, handlelength=1.6, columnspacing=.9)
    fig.subplots_adjust(left=.08, right=.99, bottom=.24, top=.91, wspace=.24, hspace=.49)
    save(fig, output, 'robustness_analysis')


def ablations(data, output, fullrange=False):
    a = data['ablations']
    fig = plt.figure(figsize=(7.2, 2.75))
    outer = fig.add_gridspec(2, 2, left=.065, right=.99, bottom=.12, top=.9, hspace=.68, wspace=.18)
    for i, (key, title) in enumerate([('radius', 'HOFLON-EasyBCQ'), ('core', 'CORE-PIVOT'), ('iql', 'Frozen-IQL extraction'), ('cql', 'CQL backups')]):
        grid = outer[i // 2, i % 2].subgridspec(1, 2, wspace=.24)
        axs = [fig.add_subplot(grid[0, c]) for c in range(2)]
        bounds = outer[i // 2, i % 2].get_position(fig)
        fig.text((bounds.x0 + bounds.x1) / 2, bounds.y1 + .052, f'({chr(97 + i)}) {title}', ha='center', fontsize=7.5, weight='bold')
        xlabel = {'radius': 'Radius ρ / box only', 'core': 'Critic updates (×1000)',
                  'iql': 'Extraction updates (×1000)', 'cql': 'Training updates (×1000)'}[key]
        fig.text((bounds.x0 + bounds.x1) / 2, bounds.y0 - .061, xlabel, ha='center', va='top', fontsize=6.5)
        for condition, ax in enumerate(axs):
            style(ax)
            ax.set_title(['Ordinary', 'Stronger'][condition], fontsize=6.2, pad=2)
            if key == 'radius':
                band(ax, range(5), np.asarray(a[key]['costs'])[:, :, condition], COLORS['hoflon'], marker='o')
                ax.set_xticks(range(5), a[key]['labels'])
            else:
                specs = {'core': [('0.01', '#008080', 'λ=0.01', '-'), ('0', '#9E1B32', 'λ=0', '--')],
                         'iql': [('AWR', '#CC79A7', 'AWR', '-'), ('DDPG+BC', '#0072B2', 'DDPG+BC', '-')],
                         'cql': [('CQL', '#D55E00', 'CQL', '--'), ('SARSA-CQL', '#49516F', 'SARSA-CQL', '-')]}[key]
                for name, color, label, ls in specs:
                    curve = a[key]['curves'][name]
                    steps, costs = np.asarray(curve['steps']), np.asarray(curve['costs'])[:, :, condition]
                    # The compact CQL window begins at 10k; its full-range
                    # companion retains the shared untrained step-zero policy.
                    mask = ablation_step_mask(key, steps, fullrange)
                    band(ax, steps[mask] / 1000, costs[:, mask], color, label, ls)
                    if key == 'iql' and not fullrange:
                        above = costs.max(0) > 2000
                        ax.scatter(steps[above] / 1000, np.full(above.sum(), 1800), marker='^', s=10, color=color, zorder=6)
                ax.set_xlim(0, 50 if key == 'iql' else 500)
                ax.set_xticks([0, 25, 50] if key == 'iql' else [0, 250, 500])
                if condition == 0:
                    ax.legend(fontsize=5.5, frameon=False, handlelength=1.3, loc='upper right')
        if key == 'iql':
            for ax in axs:
                if fullrange:
                    ax.set_yticks([100, 10000, 1000000, 100000000])
                else:
                    ax.set_ylim(12, 2000)
            if not fullrange:
                axs[0].set_yticks([20, 40, 100, 500, 1000], ['20', '40', '100', '500', '1000'])
                axs[1].set_yticks([20, 50, 100, 500, 1000], ['20', '50', '100', '500', '1000'])
        elif key == 'cql' and not fullrange:
            for ax in axs:
                ax.set_ylim(20, 1000)
                ax.set_yticks([40, 60, 100, 300, 1000], ['40', '60', '100', '300', '1000'])
    save(fig, output, 'four_ablations_fullrange' if fullrange else 'four_ablations')


def ablation_step_mask(key, steps, fullrange):
    """Only the compact CQL panel omits the shared untrained checkpoint."""
    steps = np.asarray(steps)
    return steps > 0 if key == 'cql' and not fullrange else np.ones(len(steps), dtype=bool)


def test_violins(data, output):
    rows = data['original_test']['rows']
    methods = [x for x in ORDER if x != 'sarsa-cql']
    fig, axes = plt.subplots(2, 1, figsize=(6.75, 6.6))
    summary = {}
    for strong, ax in enumerate(axes):
        costs = [[r['cost'] for r in rows if r['method'] == m and r['stronger'] == bool(strong)] for m in methods]
        parts = ax.violinplot([np.log10(x) for x in costs], widths=.76,
                              showextrema=False, bw_method=.35, points=200)
        for body, method in zip(parts['bodies'], methods):
            body.set_facecolor(COLORS[method]); body.set_edgecolor(COLORS[method]); body.set_alpha(.35)
        for j, (method, values) in enumerate(zip(methods, costs)):
            q1, med, q3 = np.log10(np.percentile(values, [25, 50, 75]))
            jitter = np.random.default_rng(870+j).uniform(-.14, .14, len(values))
            ax.scatter(j+1+jitter, np.log10(values), s=2.6, color=COLORS[method],
                       alpha=.33, linewidths=0, zorder=2)
            ax.plot([j + 1, j + 1], [q1, q3], color='white', lw=2.3, zorder=4)
            ax.scatter(j + 1, med, color='white', edgecolor=COLORS[method], s=10, zorder=5)
            seed_means = [np.mean([r['cost'] for r in rows if r['method'] == method and r['seed'] == seed and r['stronger'] == bool(strong)]) for seed in range(123, 128)] if method != 'pd11' else values
            if method != 'pd11':
                ax.scatter(j+1+np.linspace(-.16,.16,5), np.log10(seed_means),
                           marker='D', s=9, facecolors=COLORS[method], edgecolors='black',
                           linewidths=.25, zorder=6)
            summary.setdefault(method, {})[['ordinary', 'stronger'][strong]] = dict(mean=float(np.mean(seed_means)), sample_sd=float(np.std(seed_means, ddof=1)), sd_population='wind means across training seeds' if method != 'pd11' else 'wind realisations of one fixed controller')
        ax.set_xticks(range(1, 9), ['BC', 'EasyBCQ', 'HOFLON–\nEasyBCQ', 'CORE-\nPIVOT', 'TD3-BC', 'IQL', 'CQL', 'Best PD\nbaseline'])
        ax.set_title(['(a) Ordinary winds', '(b) Stronger winds'][strong], fontsize=10, weight='bold', loc='left')
        low = min(min(np.log10(v)) for v in costs)-.1
        high = max(max(np.log10(v)) for v in costs)+.1
        ticks = [v for v in [20,30,50,100,200,300,500,1000,2000,5000,10000] if low <= np.log10(v) <= high]
        if len(ticks)>5: ticks=ticks[::2]
        ax.set_yticks(np.log10(ticks), [str(v) for v in ticks])
        ax.set_ylim(low,high)
        style(ax, False)
        ax.set_ylabel('Episode cost')
        ax.tick_params(axis='x', length=0, labelsize=8)
        for label, method in zip(ax.get_xticklabels(), methods): label.set_color(COLORS[method])
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color='gray',markersize=3,label='Individual episode'),
                        Line2D([],[],marker='o',ls='',color='gray',markerfacecolor='white',markersize=4,label='Median / IQR'),
                        Line2D([],[],marker='D',ls='',color='black',markerfacecolor='gray',markersize=4,label='Seed mean (learned methods)')],
               loc='lower center',ncol=3,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.11, right=.99, bottom=.12, top=.96, hspace=.32)
    save(fig, output, 'held_out_test_performance')
    (output / 'held_out_test_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    remove_legacy_outputs(output, 'held_out_test_summary', ('json',))


def dataset(output):
    with np.load(ROOT / 'results/paper_data/benchmark_data.npz') as z:
        costs, kp, kd, winds, time = [z[k].copy() for k in ['costs', 'kp', 'kd', 'training_winds', 'training_time']]
    means = costs.mean(1); order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(3.5, 2.8)); style(ax, False)
    dots = ax.scatter(kp, kd, c=means, cmap='cividis', norm=LogNorm(), s=8, linewidths=0)
    ax.scatter(kp[order[:10]], kd[order[:10]], facecolor='none', edgecolor='#8063AE', s=22, linewidth=.7)
    ax.scatter(kp[order[0]], kd[order[0]], marker='*', color='#192F4C', edgecolor='white', s=35, linewidth=.4)
    ax.set(xlabel='$k_p$', ylabel='$k_d$', xticks=[1,2,3], yticks=[.5,1,1.5])
    bar = fig.colorbar(dots, ax=ax, fraction=.035, pad=.025)
    bar.set_ticks([70,150,430], labels=['70','150','430'])
    bar.ax.minorticks_off()
    bar.ax.tick_params(labelsize=6.5)
    save(fig, output, 'pd_gains')
    colors = ['#0072B2', '#D55E00', '#009E73']
    fig, ax = plt.subplots(figsize=(3.5, 2.8)); style(ax, False)
    values = [costs.ravel(), costs[order[:10]].ravel(), costs[order[0]]]
    parts = ax.violinplot([np.log10(x) for x in values], widths=.75, showextrema=False)
    for body, color in zip(parts['bodies'], colors): body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(.4)
    for j, v in enumerate(values,1):
        q1,med,q3=np.log10(np.percentile(v,[25,50,75]))
        ax.plot([j,j],np.log10([v.min(),v.max()]),color='#29465b',lw=.7)
        ax.plot([j,j],[q1,q3],color='#29465b',lw=3)
        ax.scatter(j,med,color='white',edgecolor='#29465b',s=18,zorder=3)
    ax.set_xticks([1, 2, 3], ['All PDs', 'Top 10', 'Best PD'])
    ax.set_yticks(np.log10([50, 100, 200, 800]), ['50', '100', '200', '800'])
    ax.set_ylabel('Episode cost')
    save(fig, output, 'cost_diversity')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', choices=['all', 'learning', 'robustness', 'ablations', 'test', 'dataset'], default='all')
    parser.add_argument('--output', type=Path, default=ROOT / 'figures/reproduced')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 7, 'axes.labelsize': 7,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    data = verify()
    for name, function in [('learning', learning), ('robustness', robustness), ('ablations', ablations), ('test', test_violins)]:
        if args.figure in ('all', name): function(data, args.output)
    if args.figure in ('all', 'dataset'): dataset(args.output)
    print(f'Plots written to {args.output.resolve()}')


if __name__ == '__main__':
    main()
