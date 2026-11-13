import numpy as np
import torch
import unittest
from algorithms.corl_vendor.iql import GaussianPolicy, TwinQ, ValueFunction
from extraction.frozen_iql import ExtractionConfig, FrozenIQLExtractor, module_hash


def source_fixture():
    torch.manual_seed(12)
    return dict(step=500000, mean=np.zeros(25, np.float32), scale=np.ones(25, np.float32),
                modules=dict(q_target=TwinQ(25, 3).state_dict(), vf=ValueFunction(25).state_dict(),
                             actor=GaussianPolicy(25, 3, 1.).state_dict()))


class ExtractionTests(unittest.TestCase):
    def test_frozen_values_fixed_sigma_and_resume(self):
        for kind, coefficient in [('awr', 5.), ('ddpg_bc', 1.)]:
            with self.subTest(kind=kind):
                torch.set_num_threads(1)
                config = ExtractionConfig(kind=kind, coefficient=coefficient)
                source = source_fixture()
                first = FrozenIQLExtractor(source, config)
                second = FrozenIQLExtractor(source, config)
                rng = np.random.default_rng(13)
                obs = rng.normal(size=(512, 25)).astype(np.float32)
                actions = rng.uniform(-1, 1, size=(512, 3)).astype(np.float32)
                for _ in range(3):
                    first.update(*first.batch(obs, actions))
                second.restore(first.snapshot())
                assert first.update(*first.batch(obs, actions)) == second.update(*second.batch(obs, actions))
                assert module_hash({'actor': first.actor}) == module_hash({'actor': second.actor})
                first.assert_frozen()
                assert not first.actor.log_std.requires_grad
                assert torch.equal(first.actor.log_std.exp(), torch.ones(3))
