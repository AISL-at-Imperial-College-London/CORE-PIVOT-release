"""Evaluation-only identity checks: no learner or optimizer is constructed."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from evaluation.identity import checkpoint_identity, file_sha


class SelectedIdentityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / 'arbitrary_name.pt'
        self.record = {'config': {'discount': .99}, 'step': 360000}
        self.save()

    def save(self):
        torch.save(self.record, self.path)

    def manifest(self, **changes):
        entry = dict(path=self.path.name, seed=123, step=360000,
                     sha256=file_sha(self.path))
        entry.update(changes)
        self.write_manifest([entry])
        return entry

    def write_manifest(self, entries):
        (self.root / 'manifest.json').write_text(
            json.dumps({'checkpoints': entries}), encoding='utf-8')

    def test_exported_seed_verified_by_hash_linked_manifest(self):
        self.manifest()
        result = checkpoint_identity(self.path, 123)
        self.assertTrue(result['seed_verified'])
        self.assertEqual(result['checkpoint_seed'], 123)
        self.assertEqual(result['policy_config'], {'discount': .99})
        self.assertIn('hash-linked selected', result['seed_evidence'])

    def test_expected_seed_must_match(self):
        self.manifest()
        with self.assertRaisesRegex(ValueError, 'does not match'):
            checkpoint_identity(self.path, 124)

    def test_wrong_hash_is_rejected(self):
        self.manifest(sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            checkpoint_identity(self.path, 123)

    def test_wrong_step_is_rejected(self):
        self.manifest(step=500000)
        with self.assertRaisesRegex(ValueError, 'step disagree'):
            checkpoint_identity(self.path, 123)

    def test_manifest_and_internal_seed_must_agree(self):
        self.record['seed'] = 124
        self.save()
        self.manifest()
        with self.assertRaisesRegex(ValueError, 'manifest seed disagree'):
            checkpoint_identity(self.path, 123)

    def test_bc_bundle_reads_identity_and_trainer_configuration(self):
        self.record = dict(schema=1, identity={'seed': 123},
                           trainer={'config': {'hidden_size': 1024}, 'step': 500000})
        self.save()
        result = checkpoint_identity(self.path, 123)
        self.assertTrue(result['seed_verified'])
        self.assertEqual(result['policy_config'], {'hidden_size': 1024})
        self.manifest(step=500000)
        self.assertTrue(checkpoint_identity(self.path, 123)['seed_verified'])

    def test_conflicting_internal_seed_sources_are_rejected(self):
        self.record.update(seed=123, identity={'seed': 124})
        self.save()
        with self.assertRaisesRegex(ValueError, 'internal seeds disagree'):
            checkpoint_identity(self.path, 123)

    def test_bad_internal_identity_shape_is_rejected(self):
        self.record['identity'] = ['seed', 123]
        self.save()
        with self.assertRaisesRegex(ValueError, 'identity must be a mapping'):
            checkpoint_identity(self.path, 123)

    def test_seed_is_never_inferred_from_filename(self):
        renamed = self.path.with_name('core_seed123_360000.pt')
        self.path.rename(renamed)
        self.assertFalse(checkpoint_identity(renamed, 123)['seed_verified'])

    def test_manifest_paths_cannot_escape_directory(self):
        for name in ('../arbitrary_name.pt', '..\\arbitrary_name.pt',
                     str(self.path.resolve()), 'C:/outside/checkpoint.pt',
                     '\\outside\\checkpoint.pt'):
            with self.subTest(name=name):
                self.manifest(path=name)
                with self.assertRaisesRegex(ValueError, 'path'):
                    checkpoint_identity(self.path, 123)

    def test_duplicate_matching_manifest_entries_are_rejected(self):
        entry = self.manifest()
        self.write_manifest([entry, dict(entry)])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            checkpoint_identity(self.path, 123)

    def test_unavailable_other_entries_do_not_block_present_checkpoint(self):
        entry = self.manifest()
        missing = dict(path='missing.pt', seed=126, step=290000,
                       sha256=None, available=False)
        self.write_manifest([entry, missing])
        self.assertTrue(checkpoint_identity(self.path, 123)['seed_verified'])

    def test_manifest_seed_must_be_an_integer(self):
        self.manifest(seed='123')
        with self.assertRaisesRegex(ValueError, 'seed must be an integer'):
            checkpoint_identity(self.path, 123)

    def test_native_sidecar_still_checked_with_manifest(self):
        # Native sidecars live in checkpoints/ beside run.json in their parent.
        folder = self.root / 'checkpoints'
        folder.mkdir()
        self.path = folder / 'native.pt'
        self.save()
        run = self.root / 'run.json'
        run.write_text(json.dumps({'identity': {'seed': 123,
                                              'config': self.record['config']}}))
        self.path.with_suffix('.json').write_text(json.dumps({
            'model_sha256': file_sha(self.path), 'run_sha256': file_sha(run)}))
        descriptor = dict(path='native.pt', seed=124, step=360000,
                          sha256=file_sha(self.path))
        (folder / 'manifest.json').write_text(json.dumps({'checkpoints': [descriptor]}))
        with self.assertRaisesRegex(ValueError, 'manifest seed disagree'):
            checkpoint_identity(self.path, 123)
        run.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'sidecar hash mismatch'):
            checkpoint_identity(self.path, 123)


if __name__ == '__main__':
    unittest.main()
