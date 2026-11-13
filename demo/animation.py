"""Closer shared view, larger outlined bodies, and synchronous tracking error."""
import argparse
import json
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import PchipInterpolator
from demo import playback as base
from demo.drone import LatexStyleDrone

ROOT, FPS = base.ROOT, base.FPS
OUT = ROOT/'outputs/demo'
FIGSIZE = (16, 10)
RESOLUTION = (1920, 1200)
ZOOM, BODY_SCALE, ARROW_LENGTH = 1.62, .64, .82
ERROR_LIMIT = 2.25


def errors_at(data, reference, times):
    return np.linalg.norm(data['position'](times)-reference(times), axis=-1)


def build_scene(index, fps=FPS):
    groups, lo, hi, maxwind, records = base.load_data()
    group = groups[index]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': '#ECF2F8', 'font.size': 10,
                         'axes.labelcolor': '#C5D2DD', 'xtick.color': '#9DB0C0', 'ytick.color': '#9DB0C0'})
    fig = plt.figure(figsize=FIGSIZE, facecolor='black')
    fig.text(.035, .962, base.CONDITIONS[index][1], fontsize=21, weight='bold')
    clock = fig.text(.965, .962, '', fontsize=17, ha='right')
    eventlabel = fig.text(.72, .926, '', fontsize=11, ha='center', color='#F2B878')
    speedlabel = fig.text(.965, .926, '', fontsize=10, ha='right', color='#A9B9C7')
    times, events = base.frame_times(index, fps=fps)
    ref = group['pd11']['reference_position']
    reference = PchipInterpolator(group['pd11']['time'], ref, axis=0)
    anchors = np.array([[x, y, z] for z in [.20, 1.55]
                        for x in np.linspace(lo[0]+1, hi[0]-1, 3)
                        for y in np.linspace(lo[1]+1, hi[1]-1, 3)])
    windmap = base.matplotlib.colormaps['cool']
    windnorm = base.matplotlib.colors.Normalize(0, np.ceil(maxwind*2)/2)
    panels = []
    for j, (alg, label, color) in enumerate(base.METHODS):
        left = .018+(j % 2)*.50
        bottom = .555 if j < 2 else .205
        ax = fig.add_axes([left, bottom, .464, .34], projection='3d',
                          computed_zorder=False, facecolor='black')
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
        ax.set_box_aspect(hi-lo, zoom=ZOOM)
        ax.set_proj_type('ortho')
        ax.view_init(24, -53)
        ax.set_axis_off()
        label_y = bottom+(.333 if j < 2 else .320)
        fig.text(left+.025, label_y, label, color=color, fontsize=15, weight='bold')
        value = fig.text(left+.425, label_y, '', color=color, fontsize=10, ha='right')
        # Only the decorative floor/back wall is cropped to a panel window.
        # The recorded path and drone are not clipped or independently fitted.
        window_bottom = bottom+.006 if j < 2 else bottom-.012
        window = Rectangle((left+.012, window_bottom), .44, bottom+.312-window_bottom, transform=fig.transFigure)
        grid = []
        for x in np.arange(np.ceil(lo[0]), hi[0], 1):
            grid.extend([[[x, lo[1], lo[2]], [x, hi[1], lo[2]]],
                         [[x, hi[1], lo[2]], [x, hi[1], hi[2]]]])
        for y in np.arange(np.ceil(lo[1]), hi[1], 1):
            grid.append([[lo[0], y, lo[2]], [hi[0], y, lo[2]]])
        for z in np.arange(np.ceil(lo[2]), hi[2], .5):
            grid.append([[lo[0], hi[1], z], [hi[0], hi[1], z]])
        gridartist = Line3DCollection(grid, colors='#476075', linewidths=.7, alpha=.55, zorder=-2)
        ax.add_collection3d(gridartist, autolim=False)
        gridartist.set_clip_path(window)
        corner = np.array([lo[0]+.15, lo[1]+.15, lo[2]])
        for axis, name in zip(np.eye(3), ['x', 'y', 'z']):
            end = corner+axis*.7
            ax.plot(*np.array([corner, end]).T, color='#A1B4C3', lw=1)
            ax.text(*(end+axis*.13), name, color='#C5D1DC', fontsize=10)
        for z in [0, 1, 2]:
            ax.text(lo[0]+.12, hi[1], z, f'{z} m', color='#9DB1C1', fontsize=8)
        ax.plot(*ref.T, '--', color='#A1B7CE', lw=1.4, alpha=.8, zorder=1)
        ax.scatter(*ref[-1], marker='*', s=30, color='#A1B7CE', zorder=1)
        d = group[alg]
        ax.plot(*d['state'][:, :3].T, color=color, alpha=.15, lw=.9, zorder=1)
        trail, = ax.plot([], [], [], color=color, lw=2.5, zorder=4)
        shadow, = ax.plot([], [], [], color=color, lw=1, alpha=.20, zorder=0)
        tether, = ax.plot([], [], [], color=color, lw=.9, ls=':', alpha=.45, zorder=1)
        field = Line3DCollection([], linewidths=1.25, zorder=0)
        ax.add_collection3d(field, autolim=False)
        drone = LatexStyleDrone(ax, color, scale=BODY_SCALE, finish='dark')
        drone.edges.set_linewidth(1.35)
        drone.rims.set_linewidth(1.0)
        for artist in list(ax.lines)+list(ax.collections):
            if artist is not gridartist:
                artist.set_clip_on(False)
        panels.append((d, trail, shadow, tether, drone, field, value))
    # Wind scale moved to the header to leave the bottom for a shared time trace.
    cax = fig.add_axes([.225, .929, .17, .007])
    cb = fig.colorbar(base.matplotlib.cm.ScalarMappable(norm=windnorm, cmap=windmap), cax=cax,
                      orientation='horizontal', ticks=np.arange(0, windnorm.vmax+.1, .5))
    cb.ax.tick_params(labelsize=8, colors='#AFC0CE', length=2, pad=1)
    fig.text(.215, .929, 'Wind |w| (m/s²)', ha='right', fontsize=10, color='#CAD6E0')
    windvalue = fig.text(.41, .929, '', fontsize=10, color='#CAD6E0')
    fig.text(.525, .929, 'Matched seed 123', fontsize=9, color='#72889A', ha='center')
    fig.add_artist(Line2D([.50, .50], [.202, .895], transform=fig.transFigure, color='#25333E', lw=.8))
    fig.add_artist(Line2D([.045, .955], [.549, .549], transform=fig.transFigure, color='#25333E', lw=.8))

    errorax = fig.add_axes([.075, .057, .89, .104], facecolor='black')
    errorax.set_label('tracking_error')
    errorax.set(xlim=(0, 15), ylim=(0, ERROR_LIMIT), ylabel='Error (m)', xlabel='Physical time (s)')
    errorax.set_xticks([0, 3, 6, 9, 12, 15])
    errorax.set_yticks([0, 1, 2])
    errorax.spines[['top', 'right']].set_visible(False)
    for spine in errorax.spines.values():
        spine.set_color('#405364')
    errorax.tick_params(labelsize=9, pad=2)
    errorax.grid(axis='y', color='#314455', lw=.6, alpha=.5)
    fig.text(.075, .176, 'Position tracking error', fontsize=10, color='#CFD9E2', weight='bold')
    fig.legend(handles=[Line2D([0], [0], color=c, lw=2, label=label) for _, label, c in base.METHODS],
               bbox_to_anchor=(.965, .169), loc='lower right', ncol=4, frameon=False, fontsize=9,
               handlelength=1.9, columnspacing=1.6, borderpad=0)
    cursor = errorax.axvline(0, color='#DDE5ED', lw=1, alpha=.7)
    for event in events:
        errorax.axvline(event, color='#A89A89', ls=':', lw=1)
        errorax.text(event+.09, ERROR_LIMIT-.10, 'Gust onset' if event == 4 else 'Reversal',
                     color='#BCAE9D', fontsize=8, va='top')
    errorartists = []
    for alg, _, color in base.METHODS:
        line, = errorax.plot([], [], color=color, lw=1.9)
        point, = errorax.plot([], [], 'o', color=color, ms=4)
        line.set_gid('error_'+alg)
        point.set_gid('error_point_'+alg)
        errorartists.append((group[alg], line, point))

    def update(frame):
        t = float(times[frame])
        clock.set_text(f't = {t:05.2f} s')
        speedlabel.set_text('End of rollout' if t == 15 else f'{base.playback_rate(t, events):.1f}× playback  |  {fps} fps')
        eventlabel.set_text('WIND REVERSES' if index == 3 and 9 <= t < 10.2
                            else 'GUST BEGINS' if index >= 2 and 4 <= t < 5.2 else '')
        wind = base.wind_at(group['pd11'], t)
        magnitude = np.linalg.norm(wind)
        segments = base.field_segments(anchors, wind, length=ARROW_LENGTH)
        rgba = (*windmap(windnorm(magnitude))[:3], .66)
        windvalue.set_text(f'{magnitude:.2f}' if magnitude > 1e-10 else 'Calm')
        for d, trail, shadow, tether, drone, field, value in panels:
            position = d['position'](t)
            yaw = np.interp(t, d['time'], d['heading'])
            drone.set_pose(position, yaw, rotor_phase=t*2*np.pi*2.5)
            tt = np.r_[np.arange(max(0, t-1.2), t, .02), t]
            recent = d['position'](tt)
            trail.set_data_3d(*recent.T)
            shadow.set_data_3d(recent[:, 0], recent[:, 1], np.full(len(recent), lo[2]))
            tether.set_data_3d([position[0]]*2, [position[1]]*2, [lo[2], position[2]])
            field.set_segments(segments)
            field.set_color(rgba)
            value.set_text(f'Error  {float(errors_at(d, reference, t)):.2f} m')
        cursor.set_xdata([t, t])
        for d, line, point in errorartists:
            trace_times = np.r_[d['time'][d['time'] < t], t]
            error = errors_at(d, reference, trace_times)
            line.set_data(trace_times, error)
            point.set_data([t], [error[-1]])
        return []
    return fig, update, times, records


def render(index):
    base.OUT = OUT
    base.build_scene = build_scene
    return base.render(index)


def chapter_card(index):
    fig = plt.figure(figsize=FIGSIZE, facecolor='black')
    fig.text(.5, .55, base.CONDITIONS[index][1], ha='center', va='center', fontsize=29, color='#E5EDF5', weight='bold')
    fig.text(.5, .44, 'New matched rollout  |  t = 0 s', ha='center', fontsize=16, color='#8FA5B7')
    path = OUT/f'chapter_{index+1}.png'
    fig.savefig(path, dpi=120, facecolor='black')
    plt.close(fig)
    subprocess.run([base.imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loop', '1', '-framerate', str(FPS),
                    '-i', str(path), '-t', str(base.CARD_SECONDS), '-c:v', 'libx264', '-crf', '19',
                    '-pix_fmt', 'yuv420p', '-threads', '2', str(OUT/f'chapter_{index+1}.mp4')],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gif', action='store_true', help='Render the compact looping README animation; no training')
    parser.add_argument('--gif-output', type=Path, default=OUT/'drone_demo.gif')
    parser.add_argument('--gif-condition', choices=['ordinary', 'stronger', 'step', 'reversing'], default='reversing')
    args = parser.parse_args()
    if args.gif:
        from demo.web_preview import render_gif
        render_gif(args.gif_output, ['ordinary', 'stronger', 'step', 'reversing'].index(args.gif_condition))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate([11., 11., 4.5, 9.5]):
        fig, update, times, _ = build_scene(i)
        update(int(np.argmin(abs(times-t))))
        fig.savefig(OUT/f'preview_{i+1}.png', dpi=120, facecolor='black')
        plt.close(fig)
    previous = json.loads((ROOT/'assets/demo_trajectories/manifest.json').read_text())
    previous.update(resolution=RESOLUTION, camera_zoom=ZOOM, body_scale=BODY_SCALE, arrow_length=ARROW_LENGTH,
                    error_metric='Euclidean position error to the time-matched reference, metres; not episode cost',
                    error_limits=[0, ERROR_LIMIT], source_script=Path(__file__).name,
                    preserved='Same trajectories, shared xyz limits, camera angles, physical clock and gust timing')
    (OUT/'provenance.json').write_text(json.dumps(previous, indent=2))
    if args.preview:
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(render, range(4)))
    for i in range(4):
        chapter_card(i)
    files = [p for i in range(4) for p in [OUT/f'chapter_{i+1}.mp4', OUT/f'condition_{i+1}_120fps.mp4']]
    cmd = [base.imageio_ffmpeg.get_ffmpeg_exe(), '-y']
    for path in files:
        cmd += ['-i', str(path)]
    inputs = ''.join(f'[{i}:v]' for i in range(8))
    cmd += ['-filter_complex', inputs+'concat=n=8:v=1:a=0[v]', '-map', '[v]', '-c:v', 'libx264',
            '-crf', '19', '-preset', 'fast', '-threads', '4', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(OUT/'four_controllers_120fps.mp4')]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    subprocess.run([base.imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', str(OUT/'four_controllers_120fps.mp4'),
                    '-vf', 'fps=60', '-c:v', 'libx264', '-crf', '19', '-preset', 'fast', '-threads', '4',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT/'four_controllers_60fps.mp4')],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    print('Complete:', OUT, flush=True)


if __name__ == '__main__':
    main()
