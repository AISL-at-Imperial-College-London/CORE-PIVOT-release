import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from evaluation.identity import checkpoint_identity, register_configuration


class IdentityTests(unittest.TestCase):
    def test_settings_cannot_be_mixed(self):
        known = {}
        record = dict(method='hoflon', policy_config={'discount': .99},
                      search={'radius': .1, 'candidates': 100})
        register_configuration(record, known)
        register_configuration(dict(record, seed=124), known)
        with self.assertRaisesRegex(ValueError, 'mixed policy/search'):
            register_configuration(dict(record, search={'radius': .25}), known)
        with self.assertRaisesRegex(ValueError, 'mixed policy/search'):
            register_configuration(dict(record, policy_config={'discount': .95}), known)

    def test_seed_sidecar_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'checkpoints').mkdir()
            checkpoint = root / 'checkpoints' / 'step.pt'
            config = {'discount': .99}
            torch.save({'config': config}, checkpoint)
            self.assertFalse(checkpoint_identity(checkpoint, 123)['seed_verified'])
            run = root / 'run.json'
            run.write_text(json.dumps({'identity': {'seed': 123, 'config': config}}))
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            checkpoint.with_suffix('.json').write_text(json.dumps({
                'model_sha256': digest(checkpoint), 'run_sha256': digest(run)}))
            self.assertTrue(checkpoint_identity(checkpoint, 123)['seed_verified'])
            with self.assertRaisesRegex(ValueError, 'does not match'):
                checkpoint_identity(checkpoint, 124)
            run.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                checkpoint_identity(checkpoint, 123)

    def test_extraction_seed_is_not_a_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'step.pt'
            torch.save({'config': {'seed': 123, 'fixed_std': 1}}, path)
            record = checkpoint_identity(path, 123)
            self.assertEqual(record['policy_config'], {'fixed_std': 1})
            self.assertTrue(record['seed_verified'])


if __name__ == '__main__':
    unittest.main()
