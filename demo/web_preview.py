"""Small GitHub-compatible animation of the same verified saved rollouts."""
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from demo import playback
from demo.animation import build_scene


def render_gif(output, condition=3, fps=20, width=800):
    """Render one complete matched condition, preserving event-aware timing.

    The web asset reduces only display resolution/frame rate, not the source
    trajectories, physical horizon, camera scale, or discontinuous wind values.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, update, times, records = build_scene(condition, fps=fps)
    ffmpeg = playback.imageio_ffmpeg.get_ffmpeg_exe()
    playback.matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg
    animation = FuncAnimation(fig, update, frames=len(times), interval=1000/fps,
                              blit=False, cache_frame_data=False)
    with tempfile.TemporaryDirectory(prefix='core_pivot_demo_') as directory:
        video = Path(directory)/'preview.mp4'
        animation.save(video, writer=playback.DrawnAggWriter(
            fps=fps, codec='libx264', extra_args=['-crf', '15', '-preset', 'fast',
            '-threads', '2', '-pix_fmt', 'yuv420p']), dpi=width/fig.get_figwidth(),
            progress_callback=lambda k, n: print(f'Web preview: {k}/{n}', flush=True)
            if k % 100 == 0 else None)
        plt.close(fig)
        # A global palette keeps the algorithm colours consistent across frames;
        # ordered dithering avoids noisy, flickering quantisation on the grid.
        filters = ('split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];'
                   '[s1][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle')
        subprocess.run([ffmpeg, '-y', '-i', str(video), '-filter_complex', filters,
                        '-loop', '0', str(output)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    condition_name = playback.CONDITIONS[condition][0]
    report = {
        'condition': condition_name,
        'description': 'Full saved reversing-gust rollout; web-resolution display' if condition == 3
                       else 'Full saved matched rollout; web-resolution display',
        'fps': fps, 'width': width, 'height': round(width*10/16),
        'frame_count': len(times), 'duration_seconds': len(times)/fps,
        'physical_horizon_seconds': 15, 'loop': True,
        'source_manifest': 'assets/demo_trajectories/manifest.json',
        'source_sha256': {record['path']: record['sha256'] for record in records
                          if record['condition'] == condition_name},
        'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'bytes': output.stat().st_size,
        'notes': 'Matched seed 123; illustrative yaw. Gust onset/reversal are preserved, '
                 'with labelled slower playback around the events. No simulation or training rerun.',
    }
    output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return report
