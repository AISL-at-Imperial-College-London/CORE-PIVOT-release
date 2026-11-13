import json
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch

from evaluation.search import refine_action
from evaluation.protocol import disturbance, robustness_conditions

ROOT = Path(__file__).resolve().parents[1]


class Quadratic(torch.nn.Module):
    def forward(self, observation, action):
        return -(action - .6).square().sum(-1)


def test_radius_zero_and_box_fallback_are_feasible():
    observation = torch.zeros(25)
    anchor = np.zeros(3, dtype=np.float32)
    np.testing.assert_array_equal(refine_action(Quadratic(), observation, anchor, 0, 40), anchor)
    for radius in [.1, None]:
        action = refine_action(Quadratic(), observation, anchor, radius, 40)
        assert np.abs(action).max() <= 1
        if radius is not None:
            assert np.linalg.norm(action.astype(float)) <= radius
        assert np.square(action - .6).sum() <= np.square(anchor - .6).sum()


def test_robustness_conditions_and_steps():
    conditions = robustness_conditions()
    assert len(conditions) == 17
    reverse = next(c for c in conditions if c['id'] == 'reversing_3')
    wind = disturbance(reverse, 51020)
    assert wind.shape == (750, 3)
    assert np.all(wind[:200] == 0)
    np.testing.assert_array_equal(wind[450], -wind[449])
    assert np.abs(wind).max() == 3


def test_bundled_results_keep_batches_and_variance_separate():
    from scripts.plot_paper import verify
    data = verify()
    assert data['original_test']['episodes'] == 720
    assert len(data['original_test']['rows']) == 720
    assert len(data['robustness']['conditions']) == 17
    assert data['ablations']['cql']['fully_paired']
    assert data['ablations']['iql']['fixed_std'] == 1
    for condition, methods in data['robustness']['conditions'].items():
        assert condition.startswith(('severity_', 'step_', 'reversing_'))
        assert 'sarsa-cql' in methods
        assert 'frozen-iql' not in methods
    for curve in data['ablations']['iql']['curves'].values():
        assert curve['steps'] == list(range(0, 50001, 1000))
        assert np.asarray(curve['costs']).shape == (5, 51, 2)


class EvaluationArtifactTests(unittest.TestCase):
    def test_search_fallback(self):
        test_radius_zero_and_box_fallback_are_feasible()

    def test_wind_families(self):
        test_robustness_conditions_and_steps()

    def test_paper_data(self):
        test_bundled_results_keep_batches_and_variance_separate()

    def test_native_backup_checkpoints_load_both_arms(self):
        from ablations.cql_backup import BackupCQL
        from algorithms.corl_baselines import CORLCQLConfig
        from evaluation.policies import load_model
        torch.set_num_threads(1)
        observations = np.random.default_rng(31).normal(size=(256, 25)).astype(np.float32)
        for backup, method in [('actor', 'cql'), ('sarsa', 'sarsa-cql')]:
            learner = BackupCQL(observations, CORLCQLConfig(target_gap=30), seed=123)
            learner.backup = backup
            expected = learner.predict(observations[0])
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'checkpoint.pt'
                torch.save(learner.state_dict(), path)
                loaded = load_model(method, path)
                np.testing.assert_array_equal(expected, loaded.predict(observations[0]))
                self.assertEqual(loaded.backup, backup)
                self.assertFalse(any(p.requires_grad for m in loaded.modules.values() for p in m.parameters()))
                with self.assertRaises(ValueError):
                    load_model('sarsa-cql' if backup == 'actor' else 'cql', path)

    def test_fixed_iql_extraction_checkpoints_load_both_objectives(self):
        from algorithms.corl_baselines import CORLIQL, CORLIQLConfig
        from extraction.frozen_iql import FrozenIQLExtractor, ExtractionConfig
        from evaluation.policies import load_model
        torch.set_num_threads(1)
        observations = np.random.default_rng(32).normal(size=(256, 25)).astype(np.float32)
        learner = CORLIQL(observations, CORLIQLConfig(weight_temperature=5), seed=123)
        learner.step = 500000  # Synthetic shape fixture, not a trained paper checkpoint.
        source = learner.state_dict()
        with tempfile.TemporaryDirectory() as folder:
            source_path = Path(folder) / 'source.pt'
            torch.save(source, source_path)
            for kind, method, coefficient in [('awr', 'awr', 5), ('ddpg_bc', 'ddpg-bc', 1)]:
                fit = FrozenIQLExtractor(source, ExtractionConfig(kind=kind, coefficient=coefficient))
                path = Path(folder) / (kind + '.pt')
                torch.save(fit.snapshot(), path)
                with torch.no_grad():
                    expected = fit.actor(fit.norm(torch.as_tensor(observations[:1]))).mean[0].numpy()
                loaded = load_model(method, path, source_critic=source_path)
                np.testing.assert_array_equal(expected, loaded.predict(observations[0]))


if __name__ == '__main__':
    unittest.main()
