"""Regenerate current supplementary Figures 9 and 12--19 from archived arrays.

No checkpoints, full replay buffer, training or evaluation are required.
The numerical data are hash-verified before plotting. Fonts may vary by system.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/paper_data'
METHODS = [('pd11', 'Best PD baseline', '#333333'), ('bc', 'BC', '#777777'),
           ('easybcq', 'EasyBCQ', '#0072B2'), ('hoflon', 'HOFLON-EasyBCQ', '#8063AE'),
           ('core-pivot', 'CORE-PIVOT', '#008080'), ('td3-bc', 'TD3-BC', '#B8860B'),
           ('iql', 'IQL', '#CC79A7'), ('cql', 'CQL', '#D55E00')]
CONDITIONS = {'test': ['test_ordinary', 'test_stronger'],
              'random_walk': ['severity_2', 'severity_2.5'],
              'step': ['step_1.5', 'step_2', 'step_3'],
              'reversing': ['reversing_1.5', 'reversing_2', 'reversing_3']}


def verify(names=None):
    proof = json.loads((DATA / 'supplement_provenance.json').read_text())
    for name, expected in proof['generated_files'].items():
        if names is not None and name not in names:
            continue
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Archived plotting data hash mismatch: {name}')


def configure():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 9, 'axes.linewidth': 1.5, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none'})


def save(fig, output, name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(output / f'{name}.{ext}', dpi=240, bbox_inches='tight', pad_inches=.03)
    plt.close(fig)
    print(f'Generated {name} (PDF, SVG, PNG)')


def density(output):
    verify(['state_action_histogram.npz'])
    with np.load(DATA / 'state_action_histogram.npz', allow_pickle=False) as z:
        full, best, xe, ue = (z[k] for k in ('full_counts', 'best_counts', 'x_edges', 'u_edges'))
    assert full.sum() == 24000000 and best.sum() == 120000
    p_all, p_best = full / full.sum(), best / best.sum()
    ordered = np.sort(p_best[p_best > 0])[::-1]
    level = ordered[np.searchsorted(np.cumsum(ordered), .9)]
    cmap = LinearSegmentedColormap.from_list('recorded_density', ['#f1f7f6', '#90c5bf', '#459C99', '#126b65'])
    cmap.set_bad('white')
    fig, ax = plt.subplots(figsize=(4.2, 3.7))
    mesh = ax.pcolormesh(xe, ue, np.ma.masked_where(p_all.T == 0, p_all.T),
                         cmap=cmap, norm=LogNorm(vmin=1/full.sum(), vmax=1e-2), rasterized=True)
    ax.contour((xe[1:]+xe[:-1])/2, (ue[1:]+ue[:-1])/2, p_best.T,
               levels=[level], colors=['#243C59'], linewidths=1)
    ax.set(xlabel='$e_x$ (m)', ylabel='$u_x$ (m s$^{-2}$)', xlim=(xe[0], xe[-1]), ylim=(ue[0], ue[-1]))
    ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.legend(handles=[Line2D([], [], color='#243C59', label='Best PD baseline: 90% mass')],
              loc='lower right', frameon=False, fontsize=8)
    bar = fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=.2, fraction=.05)
    bar.set_ticks([1e-6, 1e-4, 1e-2]); bar.set_label('Probability per bin')
    save(fig, output, 'state_action_density_only')


def tuning(output, selected=None):
    verify(['tuning_arrays.npz', 'tuning_settings.json'])
    meta = json.loads((DATA / 'tuning_settings.json').read_text())
    panels = [('TD3-BC', 'TD3-BC', r'$\alpha$'), ('EasyBCQ', 'EasyBCQ', '$M$'),
              ('Hoflon', 'HOFLON-EasyBCQ', r'$\rho$'), ('CORE-PIVOT', 'CORE-PIVOT', r'$\lambda$'),
              ('IQL 0.9', 'IQL: expectile 0.90', r'$\beta$'), ('IQL 0.98', 'IQL: expectile 0.98', r'$\beta$'),
              ('CQL fixed', 'CQL: fixed weight', r'$\alpha_{\mathrm{CQL}}$'), ('CQL adaptive', 'CQL: adaptive gap', 'Gap')]
    groups = {}
    with np.load(DATA / 'tuning_arrays.npz', allow_pickle=False) as z:
        for key, _, _ in panels:
            rows = sorted((r for r in meta if r['group'] == key), key=lambda r: r['parameter'])
            groups[key] = []
            for j, row in enumerate(rows):
                prefix = key.replace(' ', '_')+f'_{j}'
                steps, costs = z[prefix+'_steps'], z[prefix+'_costs']
                assert costs.shape == (2, len(steps), 2) and np.isfinite(costs).all() and (costs > 0).all()
                np.testing.assert_allclose(costs[:, list(steps).index(row['selected_common_step'])], row['per_seed_costs'], rtol=1e-12)
                groups[key].append((row['parameter'], steps, costs))
    colors = ['#0072B2', '#009E73', '#D55E00', '#8063AE', '#B8860B', '#555555']
    styles = ['-', '--', '-.', ':', (0, (5, 1, 1, 1)), (0, (2, 1))]
    markers = ['o', 's', '^', 'D', 'v', 'X']
    title_colors = ['#B8860B', '#0072B2', '#8063AE', '#008080', '#CC79A7', '#CC79A7', '#D55E00', '#D55E00']
    for view, limits, ticks in [('full', (20, 2e8), [1e2, 1e4, 1e6, 1e8]),
                                ('zoom', (20, 2000), [20, 50, 100, 200, 500, 1000, 2000])]:
        for wi, wind in enumerate(('ordinary', 'stronger')):
            if selected is not None and (wind, view) != selected:
                continue
            fig, axes = plt.subplots(2, 4, figsize=(13.8, 6.0), sharex=True, sharey=True)
            fig.subplots_adjust(left=.065, right=.99, bottom=.14, top=.94, wspace=.19, hspace=.7)
            for pi, (ax, (key, title, symbol)) in enumerate(zip(axes.flat, panels)):
                for j, (param, steps, costs) in enumerate(groups[key]):
                    x, y = steps/1000, costs[:, :, wi]
                    ax.fill_between(x, y.min(0), y.max(0), color=colors[j], alpha=.13, lw=0)
                    ax.plot(x, y.mean(0), color=colors[j], lw=1.5, ls=styles[j], marker=markers[j],
                            ms=3, markevery=max(1, len(x)//8), label=f'{symbol} = {param:g}')
                    if view == 'zoom':
                        over = y.max(0) > limits[1]
                        ax.scatter(x[over], np.full(over.sum(), limits[1]), marker='^', s=17,
                                   edgecolors=colors[j], facecolors='white', linewidths=.7, clip_on=False, zorder=5)
                ax.set(yscale='log', ylim=limits, xlim=(0, 500), xticks=[0, 125, 250, 375, 500])
                ax.tick_params(labelbottom=True, labelsize=8, pad=2)
                ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(NullLocator())
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}' if v < 10000 else rf'$10^{{{int(np.log10(v))}}}$'))
                ax.set_title(f'({chr(97+pi)}) {title}', loc='left', fontsize=10.5,
                             fontweight='bold', color=title_colors[pi], pad=8)
                ax.grid(axis='y', color='#D6D6D6', lw=.55, alpha=.7)
                ax.legend(loc='upper center', bbox_to_anchor=(.5, -.15), ncol=2 if len(groups[key])<5 else 3,
                          fontsize=8, frameon=False, handlelength=2, handletextpad=.4, columnspacing=.8)
            fig.supylabel('Validation episode cost', x=.009, fontsize=11)
            fig.supxlabel('Training updates (×1000)', y=-.028, fontsize=11)
            save(fig, output, f'hpo_shared_{wind}_{view}_clean')


def clip(xyz, lo, hi):
    """Clip line segments to the fixed view; mark the first exit, not a failure."""
    a, delta = xyz[:-1], np.diff(xyz, axis=0)
    enter, leave, valid = np.zeros(len(a)), np.ones(len(a)), np.ones(len(a), dtype=bool)
    for k in range(3):
        moving = np.abs(delta[:, k]) > 1e-14
        valid &= moving | ((a[:, k]>=lo[k]) & (a[:, k]<=hi[k]))
        t0, t1 = np.full(len(a), -np.inf), np.full(len(a), np.inf)
        u, v = (lo[k]-a[moving, k])/delta[moving, k], (hi[k]-a[moving, k])/delta[moving, k]
        t0[moving], t1[moving] = np.minimum(u, v), np.maximum(u, v)
        enter, leave = np.maximum(enter, t0), np.minimum(leave, t1)
    valid &= enter <= leave
    seg = np.stack([a[valid]+enter[valid, None]*delta[valid], a[valid]+leave[valid, None]*delta[valid]], axis=1)
    inside = ((xyz>=lo) & (xyz<=hi)).all(1)
    exits = np.flatnonzero(inside[:-1] & ~inside[1:])
    marker = None if not len(exits) else a[exits[0]]+leave[exits[0]]*delta[exits[0]]
    lines = []
    if len(seg):
        cuts = np.r_[0, np.flatnonzero(np.any(np.abs(seg[1:, 0]-seg[:-1, 1])>1e-10, axis=1))+1, len(seg)]
        for left, right in zip(cuts[:-1], cuts[1:]):
            lines.append(np.concatenate([seg[left:right, 0], seg[right-1:right, 1]], axis=0))
    return lines, marker


def trajectories(output, family=None):
    for family in ([family] if family else CONDITIONS):
        name = f'trajectories_{family}.npz'; verify([name])
        cids = CONDITIONS[family]
        # Four algorithms per block match the enlarged supplementary layout.
        order = [1, 2, 3, 4, 0, 6, 7, 5] if family == 'test' else list(range(8))
        nr = len(cids)*2
        fig, axes = plt.subplots(nr, 4, figsize=(9.4, 1.45*nr+.8), squeeze=False,
                                  subplot_kw={'projection': '3d', 'computed_zorder': False})
        fig.subplots_adjust(left=.09, right=.995, bottom=.065, top=.95, wspace=.025, hspace=.15)
        with np.load(DATA / name, allow_pickle=False) as z:
            ref = z['reference']; lo, hi = ref.min(0)-.6, ref.max(0)+.6
            for block in range(2):
                for r, cid in enumerate(cids):
                    for col, index in enumerate(order[4*block:4*block+4]):
                        alg, label, color = METHODS[index]; ax = axes[block*len(cids)+r, col]
                        prefix = alg+'__'+cid
                        paths, costs = z[prefix+'__position'], z[prefix+'__cost']
                        winds = 15 if cid=='test_ordinary' else 5 if cid=='test_stronger' else 20
                        assert len(paths) == winds*(1 if alg=='pd11' else 5)
                        segments, marks = [], []
                        for xyz in paths:
                            lines, marker = clip(xyz, lo, hi); segments.extend(lines)
                            if marker is not None: marks.append(marker)
                        collection = Line3DCollection(segments, colors=color, linewidths=.55, alpha=.35, zorder=2)
                        ax.add_collection3d(collection)
                        # Dense traces are rasterised at 240 dpi; text/reference stay vector.
                        ax.set_rasterization_zorder(3)
                        ax.plot(*ref.T, color='#183B65', lw=1.15, ls='--', zorder=4)
                        ax.scatter(*ref[0], s=10, c='black', depthshade=False, zorder=5)
                        ax.scatter(*ref[-1], s=24, c='black', marker='*', depthshade=False, zorder=5)
                        if marks: ax.scatter(*np.array(marks).T, s=5, c='#444444', marker='x', linewidths=.45, alpha=.45, depthshade=False, zorder=6)
                        origin = np.array([hi[0], hi[1], lo[2]])
                        for k in range(3):
                            end = origin.copy(); end[k] = lo[k] if k<2 else hi[k]
                            ax.plot(*np.array([origin, end]).T, color='#555555', lw=.55, zorder=0)
                            end[k] += -.18 if k<2 else .15
                            if k==1: end[2]+=.5
                            ax.text(*end, 'xyz'[k], fontsize=6.5, color='black', ha='center', va='center', zorder=7)
                        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
                        ax.set_proj_type('ortho'); ax.set_box_aspect(hi-lo, zoom=.82)
                        ax.view_init(elev=30, azim=-55); ax.set_axis_off()
                        if r==0: ax.set_title(label, fontsize=8, color=color, weight='bold', pad=3)
                        bb = ax.get_position(original=True)
                        fig.text(bb.x0+.025*bb.width, bb.y0+.045*bb.height, f'J = {costs.mean():,.1f}', fontsize=7)
                        fig.add_artist(Rectangle((bb.x0, bb.y0), bb.width, bb.height, transform=fig.transFigure, fill=False, lw=1.4))
                    bb = axes[block*len(cids)+r, 0].get_position(original=True)
                    label = {'test_ordinary': 'Ordinary', 'test_stronger': 'Stronger'}.get(cid, 'A = '+cid.split('_')[-1])
                    fig.text(bb.x0-.012, (bb.y0+bb.y1)/2, label, ha='right', va='center', fontsize=8, rotation=90)
        fig.legend([Line2D([], [], color='#777777'), Line2D([], [], color='#183B65', ls='--'), Line2D([], [], color='black', marker='x', ls='none')],
                   ['All winds and seeds', 'Reference', 'First exit from view'], loc='lower center', ncol=3, frameon=False, fontsize=8)
        save(fig, output, family+'_trajectories_light')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', choices=['all', 'density', 'tuning', 'trajectories'], default='all')
    parser.add_argument('--output', type=Path, default=ROOT/'figures/reproduced')
    args = parser.parse_args(); configure()
    if args.figure in ('all', 'density'): density(args.output)
    if args.figure in ('all', 'tuning'): tuning(args.output)
    if args.figure in ('all', 'trajectories'): trajectories(args.output)


def reproduce_figure(number, output):
    """Render one figure using the numbering of the current manuscript."""
    configure()
    if number == 9:
        return density(output)
    if 12 <= number <= 15:
        return tuning(output, [('ordinary', 'full'), ('stronger', 'full'),
                              ('ordinary', 'zoom'), ('stronger', 'zoom')][number-12])
    if 16 <= number <= 19:
        return trajectories(output, ['test', 'random_walk', 'step', 'reversing'][number-16])
    raise ValueError(f'Figure {number} is not handled by plot_supplement')


def export_trajectory_table(output):
    """Table 5 uses the trajectory-recording batch, not the held-out scalar batch."""
    verify(['trajectory_costs.json'])
    rows = json.loads((DATA/'trajectory_costs.json').read_text())
    table = []
    for condition in sorted({r['condition'] for r in rows}):
        for method, label, _ in METHODS:
            group = [r for r in rows if r['condition']==condition and r['method']==method]
            seeds = sorted({r['seed'] for r in group}, key=str)
            seed_means = [float(np.mean([r['cost'] for r in group if r['seed']==seed])) for seed in seeds]
            assert len(seeds) == (1 if method=='pd11' else 5)
            values = [r['cost'] for r in group] if method=='pd11' else seed_means
            table.append(dict(method=label, condition=condition, mean=float(np.mean(values)),
                              sample_sd=float(np.std(values, ddof=1)), minimum=float(np.min(values)),
                              maximum=float(np.max(values)), n_episodes=len(group),
                              variability='wind_realisations' if method=='pd11' else 'wind_averaged_seed_means'))
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    target = output/'trajectory_batch_summary.csv'
    with target.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    return target


if __name__ == '__main__':
    main()
