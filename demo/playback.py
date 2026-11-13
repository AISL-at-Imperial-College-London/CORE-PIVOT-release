"""Portable utilities for the matched saved-rollout demonstration."""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/demo'
FPS = 120
CARD_SECONDS = .75
METHODS = [('cql', 'CQL', '#FF872C'), ('core-pivot', 'CORE-PIVOT', '#25D6BF'),
           ('td3-bc', 'TD3-BC', '#F0C34E'), ('pd11', 'Best PD baseline', '#DFE6EF')]
CONDITIONS = [('test_ordinary', 'Ordinary test winds'), ('test_stronger', 'Stronger test winds'),
              ('step_1.5', 'Step gust  |  A = 1.5 m/s²'),
              ('reversing_1.5', 'Reversing gust  |  A = 1.5 m/s²')]


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x*x*(3-2*x)


def playback_rate(t, events):
    slow = 0.
    for event in events:
        slow = max(slow, smoothstep((t-(event-.6))/.4)*(1-smoothstep((t-(event+.8))/.4)))
    return 1-.6*slow


@lru_cache(maxsize=1)
def load_data():
    directory = ROOT / 'assets/demo_trajectories'
    manifest = json.loads((directory/'manifest.json').read_text())
    groups = []
    for condition, _ in CONDITIONS:
        group = {}
        for algorithm, _, _ in METHODS:
            matches = [r for r in manifest['records'] if r['condition'] == condition and r['algorithm'] == algorithm]
            assert len(matches) == 1
            record = matches[0]
            path = directory/record['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
            with np.load(path, allow_pickle=False) as z:
                data = {k: z[k].copy() for k in z.files}
            np.testing.assert_allclose(data['instantaneous_cost'].sum(), record['episode_cost'], rtol=1e-10)
            data['position'] = PchipInterpolator(data['time'], data['state'][:, :3], axis=0)
            data['heading'] = heading_for(data['state'][:, 3:6])
            group[algorithm] = data
        for algorithm, _, _ in METHODS:
            for key in ['time', 'reference_position', 'disturbance']:
                np.testing.assert_array_equal(group[algorithm][key], group['pd11'][key])
        groups.append(group)
    xyz = np.concatenate([d['state'][:, :3] for g in groups for d in g.values()] + [groups[0]['pd11']['reference_position']])
    lo, hi = xyz.min(0)-.65, xyz.max(0)+.65
    maxwind = max(np.linalg.norm(g['pd11']['disturbance'], axis=1).max() for g in groups)
    return groups, lo, hi, maxwind, manifest['records']


class DrawnAggWriter(FFMpegWriter):
    """Use FuncAnimation's already-drawn Agg frame rather than draw twice."""
    def setup(self, fig, outfile, dpi=None):
        super().setup(fig, outfile, dpi=dpi)
        fig.set_dpi(self.dpi)

    def grab_frame(self, **savefig_kwargs):
        assert self.frame_format == 'rgba'
        assert tuple(self.fig.canvas.get_width_height()) == tuple(self.frame_size)
        self._proc.stdin.write(self.fig.canvas.buffer_rgba())


def heading_for(velocity):
    """Illustrative yaw, not recorded aircraft attitude."""
    v = velocity[:, :2]
    heading = np.zeros(len(v))
    moving = np.flatnonzero(np.linalg.norm(v, axis=1) > .08)
    if len(moving):
        heading[0] = np.arctan2(v[moving[0], 1], v[moving[0], 0])
    for k in range(1, len(v)):
        heading[k] = heading[k-1]
        if np.linalg.norm(v[k]) > .08:
            target = np.arctan2(v[k, 1], v[k, 0])
            delta = np.arctan2(np.sin(target-heading[k-1]), np.cos(target-heading[k-1]))
            heading[k] += np.clip(delta*(1-np.exp(-.02/.18)),
                                  -np.deg2rad(160)*.02, np.deg2rad(160)*.02)
    return heading


def frame_times(index, fps=FPS):
    events = [] if index < 2 else [4.] if index == 2 else [4., 9.]
    values, t = [0.] * round(.35*fps), 0.
    while t < 15:
        nxt = min(15., t+playback_rate(t, events)/fps)
        # Include the exact event time. Never interpolate wind across a jump.
        for event in events:
            if t < event <= nxt:
                nxt = event
                break
        values.append(nxt)
        t = nxt
    values.extend([15.]*round(.65*fps))
    return np.array(values), events


def wind_at(data, t):
    return data['disturbance'][min(int(t/.02+1e-8), 749)]


def field_segments(anchors, wind, length=.66):
    """Constant-length glyphs encode direction; colour encodes magnitude."""
    magnitude = np.linalg.norm(wind)
    if magnitude < 1e-10:
        return np.empty((0, 2, 3))
    unit = wind/magnitude
    side = np.cross(unit, [0., 0., 1.])
    if np.linalg.norm(side) < 1e-6:
        side = np.cross(unit, [0., 1., 0.])
    side /= np.linalg.norm(side)
    other = np.cross(unit, side)
    start, end = anchors-unit*length/2, anchors+unit*length/2
    segments = [np.stack([start, end], axis=1)]
    for cross in [side, -side, other, -other]:
        segments.append(np.stack([end, end-unit*.19+cross*.075], axis=1))
    return np.concatenate(segments)


def render(index):
    fig, update, times, _ = build_scene(index)
    matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    animation = FuncAnimation(fig, update, frames=len(times), interval=1000/FPS,
                               blit=False, cache_frame_data=False)
    animation.save(OUT/f'condition_{index+1}_120fps.mp4',
                   writer=DrawnAggWriter(fps=FPS, codec='libx264', extra_args=[
                       '-crf', '19', '-preset', 'fast', '-threads', '2', '-pix_fmt', 'yuv420p',
                       '-movflags', '+faststart']), dpi=120,
                   progress_callback=lambda k, n: print(f'Condition {index+1}: {k}/{n}', flush=True)
                   if k % 600 == 0 else None)
    plt.close(fig)
    return len(times)
