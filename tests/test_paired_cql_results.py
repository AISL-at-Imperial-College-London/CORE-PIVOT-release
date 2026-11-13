"""Regression checks for the completed paired validation-only backup ablation."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from scripts.plot_paper import ablation_step_mask, verify

ROOT = Path(__file__).resolve().parents[1]


class PairedCQLResultsTests(unittest.TestCase):
    def test_arrays_cover_all_seeds_steps_and_matched_winds(self):
        data = verify()
        curve = data['ablations']['cql']
        self.assertTrue(curve['fully_paired'])
        self.assertEqual(curve['deployment'], 'trained actor in both arms')
        with np.load(ROOT / 'results/paper_data/paired_cql_validation.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z['seeds'], np.arange(123, 128))
            np.testing.assert_array_equal(z['steps'], np.arange(0, 500001, 10000))
            np.testing.assert_array_equal(z['arms'], ['paired_cql', 'sarsa_cql'])
            self.assertEqual(z['episode_costs'].shape, (2, 5, 51, 20))
            self.assertEqual(z['episode_costs'].size, 10200)
            self.assertEqual(len(set(z['wind_seeds'])), 20)
            self.assertEqual(int(z['stronger'].sum()), 5)
            self.assertTrue(np.isfinite(z['episode_costs']).all())
            self.assertTrue((z['episode_costs'] >= 0).all())
            np.testing.assert_array_equal(z['episode_costs'][0, :, 0], z['episode_costs'][1, :, 0])
            means = np.stack([z['episode_costs'][..., ~z['stronger']].mean(-1),
                              z['episode_costs'][..., z['stronger']].mean(-1)], axis=-1)
            np.testing.assert_array_equal(means, z['seed_means'])
            for arm, name in enumerate(['CQL', 'SARSA-CQL']):
                np.testing.assert_array_equal(curve['curves'][name]['steps'], z['steps'])
                np.testing.assert_array_equal(curve['curves'][name]['costs'], means[arm])
            # Independently audited final means, not original selected-test costs.
            np.testing.assert_allclose(means[:, :, -1, :].mean(1),
                [[47.4406, 161.0743], [57.2401, 81.3406]], rtol=0, atol=5e-5)

    def test_other_reported_results_were_not_replaced(self):
        data = verify()
        audit = json.loads((ROOT / 'provenance/paired_cql_results.json').read_text())
        self.assertEqual(audit['evaluations'], 510)
        self.assertEqual(audit['validation_episodes'], 10200)
        self.assertTrue(all(audit['step_zero_episode_costs_bitwise_equal'].values()))
        for path, digest in audit['generated_files'].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)
        for dotted_key, digest in audit['unchanged_result_sections_sha256'].items():
            value = data
            for key in dotted_key.split('.'):
                value = value[key]
            actual = hashlib.sha256(json.dumps(value, sort_keys=True,
                                              separators=(',', ':')).encode()).hexdigest()
            self.assertEqual(actual, digest, dotted_key)

    def test_full_range_includes_untrained_policies(self):
        steps = np.arange(0, 500001, 10000)
        self.assertFalse(ablation_step_mask('cql', steps, False)[0])
        self.assertTrue(ablation_step_mask('cql', steps, True).all())
        self.assertTrue(ablation_step_mask('iql', steps, False).all())


if __name__ == '__main__':
    unittest.main()
