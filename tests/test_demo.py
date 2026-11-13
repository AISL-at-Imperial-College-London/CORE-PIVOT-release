import unittest
import hashlib
import json
from pathlib import Path
import numpy as np
from demo import playback


class DemoTests(unittest.TestCase):
    def test_readme_animation_is_bundled_and_verified(self):
        from PIL import Image
        from scripts.package_release import release_files
        root = Path(__file__).resolve().parents[1]
        path = root / 'assets/drone_demo.gif'
        record = json.loads(path.with_suffix('.json').read_text())
        self.assertIn(path, release_files())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record['sha256'])
        self.assertLess(path.stat().st_size, 10_000_000)
        with Image.open(path) as animation:
            self.assertEqual(animation.size, (record['width'], record['height']))
            self.assertEqual(animation.n_frames, record['frame_count'])
            self.assertEqual(animation.info['loop'], 0)
            self.assertEqual(animation.info['duration'], 1000 / record['fps'])

    def test_all_saved_paths_verified_and_matched(self):
        groups, lo, hi, maximum, records = playback.load_data()
        self.assertEqual(len(records), 16)
        self.assertEqual(len(groups), 4)
        self.assertTrue(np.all(hi > lo))
        self.assertGreater(maximum, 0)
        for group in groups:
            for data in group.values():
                np.testing.assert_allclose(data['position'](data['time']), data['state'][:, :3], atol=1e-12)

    def test_exact_gust_boundaries_and_zero_order_wind(self):
        groups, *_ = playback.load_data()
        times, _ = playback.frame_times(3)
        self.assertTrue(np.all(np.diff(times) >= 0))
        self.assertIn(4., times)
        self.assertIn(9., times)
        self.assertEqual(times[-1], 15.)
        wind = groups[3]['pd11']
        np.testing.assert_array_equal(playback.wind_at(wind, 4), wind['disturbance'][200])
        np.testing.assert_array_equal(playback.wind_at(wind, 9), wind['disturbance'][450])
        np.testing.assert_allclose(playback.wind_at(wind, 9), -playback.wind_at(wind, 8.98))

    def test_web_preview_preserves_events_and_horizon(self):
        times, events = playback.frame_times(3, fps=20)
        self.assertEqual(events, [4., 9.])
        self.assertIn(4., times)
        self.assertIn(9., times)
        self.assertEqual(times[0], 0.)
        self.assertEqual(times[-1], 15.)
        self.assertTrue(np.all(np.diff(times) >= 0))
        self.assertLessEqual(np.diff(times).max(), 1/20+1e-12)

    def test_web_scene_uses_its_actual_display_frame_rate(self):
        from demo.animation import build_scene
        import matplotlib.pyplot as plt
        fig, update, times, _ = build_scene(3, fps=20)
        update(int(np.argmin(abs(times-8.))))
        self.assertTrue(any('20 fps' in text.get_text() for text in fig.texts))
        self.assertFalse(any('120 fps' in text.get_text() for text in fig.texts))
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
