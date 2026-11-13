"""Small deterministic fixtures cross-checked against historical raw episodes."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from datasets.generation import (simulate_wind, augment_in_place, split_plan,
                                 controller_catalog, training_statistics, generate)
from environments.drone_env import DroneEnv, tracking_cost
from environments.wind import wind_trace


class PaperDataTests(unittest.TestCase):
    def test_locked_splits(self):
        splits = split_plan()
        groups = {name: set(sum(group.values(), [])) for name, group in splits.items()}
        self.assertEqual([len(groups[k]) for k in ('train', 'validation', 'test')], [160, 20, 20])
        self.assertEqual(set.union(*groups.values()), set(range(51000, 51200)))
        self.assertFalse(groups['train'] & groups['validation'])
        self.assertFalse(groups['train'] & groups['test'])
        self.assertFalse(groups['validation'] & groups['test'])
        self.assertEqual(len(splits['validation']['stronger']), 5)

    def test_pd_gains_are_nominal_perturbations(self):
        data = controller_catalog()
        rng = np.random.default_rng(42)
        expected = np.asarray([(2 * (1 + .5 * rng.uniform(-1, 1)),
                                1 * (1 + .5 * rng.uniform(-1, 1))) for _ in range(200)])
        # Historical CSV serialization/parsing changes gains by at most a few ULPs.
        np.testing.assert_allclose(data[['kp', 'kd']], expected, rtol=0, atol=1e-15)
        self.assertEqual(list(data.controller_id), list(range(200)))

    def test_historical_episode_fingerprints(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/data_fingerprints.json').read_text())
        for item in fixture['episodes']:
            arrays = simulate_wind(item['seed'], [0, 11, 199], stronger=item['stronger'])
            for field, digest in item['sha256'].items():
                self.assertEqual(hashlib.sha256(arrays[field].tobytes()).hexdigest(), digest)
            for start in (0, 750, 1500):
                np.testing.assert_array_equal(arrays['next_observations'][start:start + 749],
                                              arrays['observations'][start + 1:start + 750])
                np.testing.assert_array_equal(arrays['next_actions'][start:start + 749],
                                              arrays['actions'][start + 1:start + 750])
                self.assertTrue(arrays['terminals'][start + 749, 0])
                self.assertFalse(arrays['next_actions'][start + 749].any())

    def test_fixed_augmentation_keeps_targets_and_state_identity(self):
        arrays = simulate_wind(51000, [0, 11, 199])
        clean = {k: v.copy() for k, v in arrays.items()}
        record = augment_in_place(arrays)
        self.assertEqual(record['seed'], 104852)
        self.assertFalse(np.array_equal(clean['observations'][:, 12:15], arrays['observations'][:, 12:15]))
        for field in ('actions', 'next_actions', 'rewards', 'terminals', 'timeouts'):
            np.testing.assert_array_equal(arrays[field], clean[field])
        for start in (0, 750, 1500):
            np.testing.assert_array_equal(arrays['next_observations'][start:start + 749],
                                          arrays['observations'][start + 1:start + 750])
        second = {k: v.copy() for k, v in clean.items()}
        augment_in_place(second)
        for field in arrays:
            np.testing.assert_array_equal(second[field], arrays[field])

    def test_environment_scores_successor_state(self):
        env = DroneEnv()
        env.reset(seed=51000)
        _, reward, terminated, truncated, info = env.step(np.array([.4, -.1, .2]))
        self.assertEqual(reward, -tracking_cost(env.state, .02))
        np.testing.assert_array_equal(info['disturbance'], wind_trace(np.random.default_rng(51000))[0])
        self.assertFalse(terminated or truncated)

    def test_population_statistics(self):
        obs = simulate_wind(51000, [0, 11])['observations']
        mean, scale = training_statistics(obs)
        np.testing.assert_array_equal(mean, obs.astype(np.float64).mean(0))
        np.testing.assert_allclose(scale, np.maximum(obs.astype(np.float64).std(0), 1e-6), rtol=1e-14)

    def test_fresh_smoke_generation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            output = Path(tmp) / 'data'
            manifest = generate(output, smoke=True)
            self.assertEqual(manifest['rows'], 4500)
            self.assertEqual(manifest['previous_action_augmentation']['std_data_scaled'], .01)
            self.assertFalse(manifest['test_accessed'])
            with self.assertRaises(FileExistsError):
                generate(output, smoke=True)


if __name__ == '__main__':
    unittest.main()
