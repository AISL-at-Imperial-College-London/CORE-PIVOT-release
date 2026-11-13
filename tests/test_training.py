"""Fast checks; the separately logged 1000-update archive test is stronger."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from algorithms.truncated_bc import TruncatedPolicy
from core_pivot_truncated_20261019 import DensityProposal
from training.factory import configure, fresh, LEARNERS, global_rng, restore_global_rng, cpu

ROOT = Path(__file__).resolve().parents[1]


def assert_same(test, first, second):
    if isinstance(first, torch.Tensor):
        test.assertTrue(torch.equal(first, second))
    elif isinstance(first, dict):
        test.assertEqual(first.keys(), second.keys())
        for key in first:
            assert_same(test, first[key], second[key])
    elif isinstance(first, (tuple, list)):
        test.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            assert_same(test, a, b)
    else:
        test.assertEqual(first, second)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure('cpu')

    def batch(self):
        rng = np.random.default_rng(18)
        batch = {n: torch.tensor(rng.normal(size=(8, 25)), dtype=torch.float32)
                 for n in ('observations', 'next_observations')}
        batch.update({n: torch.tensor(rng.uniform(-.9, .9, (8, 3)), dtype=torch.float32)
                      for n in ('actions', 'next_actions')})
        batch.update(rewards=-torch.ones(8, 1), terminals=torch.zeros(8, 1, dtype=torch.bool),
                     timeouts=torch.zeros(8, 1, dtype=torch.bool), bootstrap_mask=torch.ones(8, 1))
        return batch

    def test_numerical_source_hashes(self):
        manifest = json.loads((ROOT / 'provenance/training_sources.json').read_text())
        for record in manifest['kernels']:
            with self.subTest(file=record['file']):
                self.assertEqual(hashlib.sha256((ROOT / record['file']).read_bytes()).hexdigest(), record['sha256'])

    def test_six_families_exact_resume(self):
        for algorithm, (cls, _) in LEARNERS.items():
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as temporary:
                config = json.loads((ROOT / 'configs/train' / f'{algorithm}.json').read_text())['config']
                config['batch_size'] = 8
                if 'hidden_size' in config:
                    config['hidden_size'] = 32
                proposal = DensityProposal(np.zeros(25), np.ones(25), 32) if algorithm == 'core-pivot' else None
                learner, sampler = fresh(algorithm, config, np.zeros(25), np.ones(25), 123, proposal=proposal)
                expected = torch.Generator().manual_seed(20123 if algorithm == 'qbeta' else 10123)
                self.assertTrue(torch.equal(sampler.get_state(), expected.get_state()))
                frozen = copy.deepcopy(proposal.state_dict()) if proposal is not None else None
                batch = self.batch()
                learner.update(batch)
                path = Path(temporary) / 'checkpoint.pt'
                learner.save(path)
                rng = global_rng()
                restored = cls.load(path)
                restore_global_rng(rng)
                first = learner.update(batch)
                after = global_rng()
                restore_global_rng(rng)
                second = restored.update(batch)
                assert_same(self, first, second)
                assert_same(self, cpu(learner.state_dict()), cpu(restored.state_dict()))
                assert_same(self, after, global_rng())
                if frozen is not None:
                    assert_same(self, frozen, learner.proposal.state_dict())

    def test_cql_paper_flags(self):
        config = json.loads((ROOT / 'configs/train/cql.json').read_text())['config']
        learner, _ = fresh('cql', config, np.zeros(25), np.ones(25), 123)
        self.assertFalse(learner.trainer.backup_entropy)
        self.assertFalse(learner.trainer.cql_max_target_backup)
        self.assertTrue(learner.trainer.cql_lagrange)
        self.assertEqual(config['entropy_temperature'], .1)
        self.assertEqual(config['target_gap'], 30.)

    def test_single_critic_and_frozen_sarsa_proposal(self):
        config = json.loads((ROOT / 'configs/train/qbeta.json').read_text())['config']
        learner, _ = fresh('qbeta', config, np.zeros(25), np.ones(25), 123)
        self.assertEqual(set(learner.modules), {'critic', 'target'})
        self.assertNotIn('proposal', learner.modules)
        self.assertFalse(hasattr(learner.critic, 'q2'))


if __name__ == '__main__':
    unittest.main()
