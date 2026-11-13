import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts.train_cql_backup import initialize, runtime_record
from scripts.train import save, sha, write_json
from scripts.cql_backup_commands import build_plan
from scripts.verify_cql_pairs import verify_pair
from training.factory import configure, cpu
from scripts.verify_backup_parity import assert_equal
from ablations.cql_backup import BackupCQL

ROOT = Path(__file__).resolve().parents[1]


class BackupTests(unittest.TestCase):
    def setUp(self):
        configure('cpu')
        self.config = json.loads((ROOT / 'configs/backup_actor.json').read_text())['config']
        self.norm = dict(mean=[0.] * 25, scale=[1.] * 25)
        rng = torch.Generator().manual_seed(17)
        self.batch = dict(observations=torch.randn(16, 25, generator=rng),
            next_observations=torch.randn(16, 25, generator=rng),
            actions=torch.rand(16, 3, generator=rng) * 2 - 1,
            next_actions=torch.rand(16, 3, generator=rng) * 2 - 1,
            rewards=-torch.rand(16, 1, generator=rng), terminals=torch.zeros(16, 1, dtype=torch.bool))

    def test_initial_networks_and_sampling_are_matched(self):
        actor, a_rng = initialize(self.norm, self.config, 123, 'actor')
        sarsa, s_rng = initialize(self.norm, self.config, 123, 'sarsa')
        assert_equal(actor.state_dict()['modules'], sarsa.state_dict()['modules'])
        assert_equal(a_rng.get_state(), s_rng.get_state())
        self.assertEqual(actor.trainer.alpha_multiplier, .1)
        self.assertFalse(actor.trainer.backup_entropy)
        self.assertEqual(actor.trainer.cql_target_action_gap, 30.)
        actor.update(self.batch)
        a_after = torch.get_rng_state().clone()
        sarsa.update(self.batch)
        assert_equal(a_after, torch.get_rng_state())
        self.assertIsNone(sarsa.trainer.recorded_next_actions)
        self.assertFalse(torch.equal(next(actor.modules['critic_1'].parameters()),
                                     next(sarsa.modules['critic_1'].parameters())))

    def test_five_seed_initial_pairing(self):
        for seed in range(123, 128):
            with self.subTest(seed=seed):
                actor, a_rng = initialize(self.norm, self.config, seed, 'actor')
                sarsa, s_rng = initialize(self.norm, self.config, seed, 'sarsa')
                first = {k: v for k, v in actor.state_dict().items() if k != 'backup'}
                second = {k: v for k, v in sarsa.state_dict().items() if k != 'backup'}
                assert_equal(first, second)
                assert_equal(a_rng.get_state(), s_rng.get_state())

    def test_paired_plan(self):
        plan = build_plan()
        self.assertFalse(plan['execution'])
        self.assertEqual(len([j for j in plan['jobs'] if j['stage'] == 'train']), 10)
        evaluations = [j for j in plan['jobs'] if j['stage'] == 'evaluate']
        self.assertEqual(len(evaluations), 510)
        self.assertEqual(plan['validation_episodes'], 10200)
        self.assertTrue(all(j['dependencies'] == ['verify_pairs'] for j in evaluations))
        self.assertFalse(any('--allow-test' in j['command'] for j in plan['jobs']))
        self.assertEqual(len({j['id'] for j in plan['jobs']}), len(plan['jobs']))

    def test_saved_pair_verifier_and_runtime_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            folders = [Path(tmp) / arm for arm in ('actor', 'sarsa')]
            for folder, arm in zip(folders, ('actor', 'sarsa')):
                folder.mkdir()
                model, sampler = initialize(self.norm, self.config, 123, arm)
                identity = dict(algorithm='cql-backup', backup=arm, seed=123, config=self.config,
                    data_manifest_sha256='test', normalization_sha256='test', initial_sha256=None,
                    sampler_seed=10123, per_update_torch_seed='seed*1000003+step',
                    actor_entropy=.1, target_entropy=False, smoke_only=True)
                write_json(folder / 'run.json', dict(identity=identity, requested_updates=1,
                    runtime=runtime_record('cpu'), kernel_sha256={'ablations/cql_backup.py':
                        sha(ROOT / 'ablations/cql_backup.py')}))
                save(model, sampler, folder, sha(folder / 'run.json'))
                model.update(self.batch)
                save(model, sampler, folder, sha(folder / 'run.json'))
                write_json(folder / 'completed_step000000001.json', dict(training_complete=True, step=1))
            result = verify_pair(*folders, 123, steps=1, checkpoint_every=1)
            self.assertTrue(result['passed'])
            path = folders[1] / 'run.json'
            run = json.loads(path.read_text())
            run['runtime']['torch'] = 'different runtime'
            path.write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, 'runtimes'):
                verify_pair(*folders, 123, steps=1, checkpoint_every=1)

    def test_serialized_resume_is_exact(self):
        learner, _ = initialize(self.norm, self.config, 123, 'sarsa')
        learner.update(self.batch)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path = Path(tmp) / 'model.pt'
            torch.save(cpu(learner.state_dict()), path)
            restored = BackupCQL.load(path)
            self.assertEqual(restored.backup, 'sarsa')
            self.assertEqual(restored.seed, 123)
            assert_equal(learner.update(self.batch), restored.update(self.batch))
            assert_equal(cpu(learner.state_dict()), cpu(restored.state_dict()))


if __name__ == '__main__':
    unittest.main()
