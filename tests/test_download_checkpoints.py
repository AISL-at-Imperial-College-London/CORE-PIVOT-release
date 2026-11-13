"""Offline safety tests: only dummy bytes, stdlib, and mocked HTTP responses."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
import urllib.error
import urllib.request
import warnings
import zipfile

from scripts import download_checkpoints as downloader


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class Response(io.BytesIO):
    def __init__(self, data, headers=None):
        super().__init__(data)
        self.headers = headers or {}


class CheckpointDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / 'selected'
        self.manifest_path = self.root / 'manifest.json'
        self.archive = self.root / 'selected-checkpoints-v1.zip'
        self.contents = {'policy/seed123.pt': b'not a Torch model',
                         'bc/seed123.pt': b'a dummy behavior checkpoint'}
        self.data = {
            'schema_version': 1,
            'repository': 'AISL-at-Imperial-College-London/CORE-PIVOT-release',
            'release_tag': 'selected-checkpoints-v1',
            'checkpoints': [
                {'method': 'test-policy', 'seed': 123, 'step': 10000,
                 'path': 'policy/seed123.pt', 'sha256': sha256(self.contents['policy/seed123.pt']),
                 'size_bytes': len(self.contents['policy/seed123.pt']),
                 'bc_checkpoint': {'seed': 123, 'step': 300000, 'path': 'bc/seed123.pt',
                                   'sha256': sha256(self.contents['bc/seed123.pt']),
                                   'size_bytes': len(self.contents['bc/seed123.pt'])}},
            ],
        }
        self.make_archive()

    def make_archive(self, members=None):
        if members is None:
            members = list(self.contents.items())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(self.archive, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in members:
                    archive.writestr(name, data)
        payload = self.archive.read_bytes()
        self.data['asset'] = {'name': self.archive.name, 'size_bytes': len(payload), 'sha256': sha256(payload)}
        self.save_manifest()

    def save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.data), encoding='utf-8')
        return downloader.load_manifest(self.manifest_path)

    def run_main(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = downloader.main(['--manifest', str(self.manifest_path), '--output', str(self.output),
                                      *arguments])
        return result, stdout.getvalue(), stderr.getvalue()

    def test_offline_import_and_verify_never_resolve_credentials_or_network(self):
        with (patch.object(downloader, 'get_token', side_effect=AssertionError('credential lookup')),
              patch.object(downloader.urllib.request, 'build_opener', side_effect=AssertionError('network'))):
            status, stdout, stderr = self.run_main('--archive', str(self.archive))
            self.assertEqual(status, 0, stderr)
            self.assertIn('installed 2', stdout)
            self.assertEqual(self.run_main('--verify-only')[0], 0)
        for name, payload in self.contents.items():
            self.assertEqual((self.output / name).read_bytes(), payload)
        self.assertEqual((self.output / 'manifest.json').read_bytes(), self.manifest_path.read_bytes())

    def test_verify_only_never_copies_manifest_to_custom_output(self):
        downloader.install_archive(self.save_manifest(), self.archive, self.output)
        self.assertEqual(self.run_main('--verify-only')[0], 0)
        self.assertFalse((self.output / 'manifest.json').exists())

    def test_custom_output_refuses_conflicting_manifest_before_install(self):
        self.output.mkdir()
        target = self.output / 'manifest.json'
        target.write_bytes(b'user-owned manifest')
        result, _, stderr = self.run_main('--archive', str(self.archive))
        self.assertEqual(result, 1)
        self.assertIn('Existing output manifest differs', stderr)
        self.assertEqual(target.read_bytes(), b'user-owned manifest')
        self.assertFalse((self.output / 'policy/seed123.pt').exists())

    def test_matching_output_manifest_is_not_rewritten(self):
        self.output.mkdir()
        target = self.output / 'manifest.json'
        target.write_bytes(self.manifest_path.read_bytes())
        original_mtime = target.stat().st_mtime_ns
        self.assertEqual(self.run_main('--archive', str(self.archive))[0], 0)
        self.assertEqual(target.stat().st_mtime_ns, original_mtime)

    def test_import_does_not_download_run_credentials_or_load_models(self):
        original_import = __import__

        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'numpy', 'training', 'algorithms'}:
                raise AssertionError(f'Model/training dependency imported: {name}')
            return original_import(name, *args, **kwargs)

        with (patch('builtins.__import__', side_effect=guarded_import),
              patch.object(downloader.subprocess, 'run', side_effect=AssertionError('credential command')),
              patch.object(downloader.urllib.request, 'build_opener', side_effect=AssertionError('network'))):
            importlib.reload(downloader)

    def test_correct_existing_files_skip_download_and_keep_mtime(self):
        downloader.install_archive(self.save_manifest(), self.archive, self.output)
        original = (self.output / 'policy/seed123.pt').stat().st_mtime_ns
        with patch.object(downloader, 'download_archive', side_effect=AssertionError('download')):
            result, stdout, stderr = self.run_main()
        self.assertEqual(result, 0, stderr)
        self.assertIn('installed 0', stdout)
        self.assertEqual((self.output / 'policy/seed123.pt').stat().st_mtime_ns, original)

    def test_missing_verify_only_reports_failure_without_credentials(self):
        with patch.object(downloader, 'get_token', side_effect=AssertionError('credentials')):
            status, _, stderr = self.run_main('--verify-only')
        self.assertEqual(status, 1)
        self.assertIn('Missing 2', stderr)
        self.assertFalse(self.output.exists())

    def test_existing_divergent_file_is_never_overwritten(self):
        destination = self.output / 'policy/seed123.pt'
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b'user-owned existing weights')
        with patch.object(downloader, 'download_archive', side_effect=AssertionError('download')):
            result, _, stderr = self.run_main('--archive', str(self.archive))
        self.assertEqual(result, 1)
        self.assertIn('refusing to overwrite', stderr)
        self.assertEqual(destination.read_bytes(), b'user-owned existing weights')
        self.assertFalse((self.output / 'bc/seed123.pt').exists())

    def test_shared_paths_are_deduplicated(self):
        second = dict(self.data['checkpoints'][0], method='another-policy')
        self.data['checkpoints'].append(second)
        manifest = self.save_manifest()
        self.assertEqual(manifest.checkpoint_count, 2)
        self.assertEqual(len(manifest.files), 2)
        self.assertEqual(downloader.install_archive(manifest, self.archive, self.output), 2)

    def test_optional_per_file_sizes_are_supported(self):
        checkpoint = self.data['checkpoints'][0]
        del checkpoint['size_bytes']
        del checkpoint['bc_checkpoint']['size_bytes']
        self.assertEqual(downloader.install_archive(self.save_manifest(), self.archive, self.output), 2)

    def test_conflicting_shared_descriptor_is_rejected(self):
        second = dict(self.data['checkpoints'][0], method='another-policy', sha256='0' * 64)
        self.data['checkpoints'].append(second)
        with self.assertRaisesRegex(downloader.CheckpointError, 'Conflicting descriptors'):
            self.save_manifest()

    def test_invalid_manifest_paths_are_rejected(self):
        for name in ('../outside.pt', '/outside.pt', 'C:/outside.pt', 'folder\\outside.pt',
                     'folder/../outside.pt', 'folder//outside.pt', 'folder/./outside.pt',
                     'NUL.pt', 'trailing. /file.pt', 'folder/file.pt\0', ''):
            with self.subTest(name=name):
                self.data['checkpoints'][0]['path'] = name
                with self.assertRaises(downloader.CheckpointError):
                    self.save_manifest()

    def test_case_collision_is_rejected(self):
        second = dict(self.data['checkpoints'][0], path='POLICY/SEED123.PT')
        self.data['checkpoints'].append(second)
        with self.assertRaisesRegex(downloader.CheckpointError, 'Case-colliding'):
            self.save_manifest()

    def test_parent_file_collision_is_rejected(self):
        second = dict(self.data['checkpoints'][0], path='policy')
        self.data['checkpoints'].append(second)
        with self.assertRaisesRegex(downloader.CheckpointError, 'parent directory'):
            self.save_manifest()

    def test_invalid_schema_hash_and_size_are_rejected(self):
        for key, value in (('schema_version', 2), ('schema_version', True), ('checkpoints', []),
                           ('repository', 'https://github.com/user/repo')):
            with self.subTest(key=key, value=value):
                original = self.data[key]
                self.data[key] = value
                with self.assertRaises(downloader.CheckpointError):
                    self.save_manifest()
                self.data[key] = original
        for key, value in (('sha256', 'bad'), ('size_bytes', -1), ('size_bytes', True),
                           ('size_bytes', downloader.MAX_ARCHIVE_BYTES + 1)):
            with self.subTest(key=key, value=value):
                original = self.data['asset'][key]
                self.data['asset'][key] = value
                with self.assertRaises(downloader.CheckpointError):
                    self.save_manifest()
                self.data['asset'][key] = original

    def test_bad_archive_digest_fails_before_staging(self):
        self.data['asset']['sha256'] = '0' * 64
        manifest = self.save_manifest()
        with patch.object(downloader.tempfile, 'TemporaryDirectory', side_effect=AssertionError('staging')):
            with self.assertRaisesRegex(downloader.CheckpointError, 'Archive size/SHA-256'):
                downloader.install_archive(manifest, self.archive, self.output)
        self.assertFalse(self.output.exists())

    def test_archive_size_mismatch_is_rejected(self):
        self.data['asset']['size_bytes'] -= 1
        with self.assertRaisesRegex(downloader.CheckpointError, 'stream limit'):
            downloader.install_archive(self.save_manifest(), self.archive, self.output)
        self.assertFalse(self.output.exists())

    def test_all_file_digests_pass_before_any_file_is_installed(self):
        self.data['checkpoints'][0]['bc_checkpoint']['sha256'] = '0' * 64
        with self.assertRaisesRegex(downloader.CheckpointError, 'ZIP checkpoint size/SHA-256 mismatch'):
            downloader.install_archive(self.save_manifest(), self.archive, self.output)
        self.assertFalse(self.output.exists())

    def test_unexpected_duplicate_missing_traversal_and_directory_members_rejected(self):
        originals = list(self.contents.items())
        variants = [originals + [('unexpected.pt', b'extra')],
                    originals + [originals[0]], originals[:1],
                    originals + [('../escape.pt', b'escape')],
                    originals + [('policy/', b'')]]
        for members in variants:
            with self.subTest(members=members):
                self.make_archive(members)
                with self.assertRaises(downloader.CheckpointError):
                    downloader.install_archive(self.save_manifest(), self.archive, self.output)
                self.assertFalse(self.output.exists())
                self.assertFalse((self.root / 'escape.pt').exists())

    def test_symlink_member_is_rejected(self):
        member = zipfile.ZipInfo('policy/seed123.pt')
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.make_archive([(member, self.contents['policy/seed123.pt']),
                           ('bc/seed123.pt', self.contents['bc/seed123.pt'])])
        with self.assertRaisesRegex(downloader.CheckpointError, 'regular file'):
            downloader.install_archive(self.save_manifest(), self.archive, self.output)
        self.assertFalse(self.output.exists())

    def test_member_size_and_total_size_limits_are_checked_before_staging(self):
        checkpoint = self.data['checkpoints'][0]
        checkpoint['size_bytes'] += 1
        with self.assertRaisesRegex(downloader.CheckpointError, 'ZIP member size'):
            downloader.install_archive(self.save_manifest(), self.archive, self.output)
        checkpoint['size_bytes'] -= 1
        manifest = self.save_manifest()
        with patch.object(downloader, 'MAX_EXTRACTED_BYTES', 1):
            with self.assertRaisesRegex(downloader.CheckpointError, 'extracted size limit'):
                downloader.install_archive(manifest, self.archive, self.output)

    def test_output_symlink_is_rejected(self):
        outside = self.root / 'outside'
        outside.mkdir()
        self.output.mkdir()
        try:
            (self.output / 'policy').symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest('This platform does not permit unprivileged symlinks.')
        with self.assertRaisesRegex(downloader.CheckpointError, 'symlink/reparse-point'):
            downloader.install_archive(self.save_manifest(), self.archive, self.output)
        self.assertEqual(list(outside.iterdir()), [])

    def test_mocked_release_download_uses_asset_id_and_hashes_payload(self):
        manifest = self.save_manifest()
        payload = self.archive.read_bytes()
        metadata = {'assets': [{'id': 1234, 'name': manifest.asset_name, 'size': len(payload),
                                'browser_download_url': 'https://untrusted.invalid/not-used'}]}
        opener = Mock()
        opener.open.side_effect = [Response(json.dumps(metadata).encode()),
                                   Response(payload, {'Content-Length': str(len(payload))})]
        destination = self.root / 'download.zip'
        with (patch.object(downloader, 'get_token', return_value='secret-never-printed'),
              patch.object(downloader.urllib.request, 'build_opener', return_value=opener)):
            downloader.download_archive(manifest, destination)
        self.assertEqual(destination.read_bytes(), payload)
        requests = [call.args[0] for call in opener.open.call_args_list]
        self.assertTrue(requests[0].full_url.endswith('/releases/tags/selected-checkpoints-v1'))
        self.assertTrue(requests[1].full_url.endswith('/releases/assets/1234'))
        self.assertEqual(requests[1].get_header('Accept'), 'application/octet-stream')
        self.assertEqual(requests[1].get_header('Authorization'), 'Bearer secret-never-printed')
        self.assertNotIn('secret-never-printed', requests[1].full_url)

    def test_public_release_download_needs_no_authentication(self):
        manifest = self.save_manifest()
        payload = self.archive.read_bytes()
        metadata = {'assets': [{'id': 1234, 'name': manifest.asset_name, 'size': len(payload)}]}
        opener = Mock()
        opener.open.side_effect = [Response(json.dumps(metadata).encode()), Response(payload)]
        destination = self.root / 'public-download.zip'
        with (patch.object(downloader, 'get_token', return_value=None),
              patch.object(downloader.urllib.request, 'build_opener', return_value=opener)):
            downloader.download_archive(manifest, destination)
        self.assertEqual(destination.read_bytes(), payload)
        for call in opener.open.call_args_list:
            request = call.args[0]
            self.assertIn('/repos/AISL-at-Imperial-College-London/CORE-PIVOT-release/', request.full_url)
            self.assertIsNone(request.get_header('Authorization'))

    def test_download_rejects_wrong_metadata_length_or_digest(self):
        manifest = self.save_manifest()
        payload = self.archive.read_bytes()
        metadata = {'assets': [{'id': 1234, 'name': manifest.asset_name, 'size': len(payload)}]}
        variants = [(dict(metadata, assets=[]), payload, {}),
                    ({'assets': [dict(metadata['assets'][0], size=1)]}, payload, {}),
                    (metadata, payload, {'Content-Length': '1'}),
                    (metadata, b'x' * len(payload), {}),
                    (metadata, payload + b'x', {}),
                    (metadata, payload[:-1], {})]
        for index, (release, body, headers) in enumerate(variants):
            with self.subTest(index=index):
                opener = Mock()
                opener.open.side_effect = [Response(json.dumps(release).encode()), Response(body, headers)]
                with (patch.object(downloader, 'get_token', return_value=None),
                      patch.object(downloader.urllib.request, 'build_opener', return_value=opener)):
                    with self.assertRaises(downloader.CheckpointError):
                        downloader.download_archive(manifest, self.root / f'bad-{index}.zip')

    def test_cross_host_redirect_strips_authentication(self):
        request = urllib.request.Request('https://api.github.com/repos/user/repo/releases/assets/1',
                                         headers={'Authorization': 'Bearer secret', 'Cookie': 'secret-cookie'})
        handler = downloader.SafeRedirectHandler()
        redirect = handler.redirect_request(request, None, 302, 'Found', {},
                                            'https://release-assets.githubusercontent.com/asset?sig=example')
        self.assertIsNone(redirect.get_header('Authorization'))
        self.assertIsNone(redirect.get_header('Cookie'))
        same_host = handler.redirect_request(request, None, 302, 'Found', {},
                                             'https://api.github.com/another-location')
        self.assertEqual(same_host.get_header('Authorization'), 'Bearer secret')
        different_port = handler.redirect_request(request, None, 302, 'Found', {},
                                                  'https://api.github.com:444/another-location')
        self.assertIsNone(different_port.get_header('Authorization'))

    def test_insecure_or_userinfo_redirects_are_rejected(self):
        request = urllib.request.Request('https://api.github.com/original')
        for url in ('http://api.github.com/insecure', 'file:///etc/passwd', 'https://user:secret@example.org/file'):
            with self.subTest(url=url):
                with self.assertRaises(downloader.CheckpointError):
                    downloader.SafeRedirectHandler().redirect_request(request, None, 302, 'Found', {}, url)

    def test_environment_token_priority_and_no_subprocess(self):
        with (patch.dict(os.environ, {'GITHUB_TOKEN': 'first', 'GH_TOKEN': 'second'}, clear=True),
              patch.object(downloader.subprocess, 'run', side_effect=AssertionError('subprocess'))):
            self.assertEqual(downloader.get_token('user/repo'), 'first')
        with (patch.dict(os.environ, {'GH_TOKEN': 'second'}, clear=True),
              patch.object(downloader.subprocess, 'run', side_effect=AssertionError('subprocess'))):
            self.assertEqual(downloader.get_token('user/repo'), 'second')

    def test_gh_then_noninteractive_git_credential_fallback(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(downloader.subprocess, 'run') as run:
            run.return_value = subprocess.CompletedProcess([], 0, 'gh-secret\n', '')
            self.assertEqual(downloader.get_token('user/repo'), 'gh-secret')
            self.assertEqual(run.call_args.args[0][:3], ['gh', 'auth', 'token'])
            run.reset_mock()
            run.side_effect = [FileNotFoundError(),
                               subprocess.CompletedProcess([], 0, 'username=user\npassword=git-secret\n', '')]
            self.assertEqual(downloader.get_token('user/repo'), 'git-secret')
            call = run.call_args
            self.assertEqual(call.args[0][-2:], ['credential', 'fill'])
            self.assertIn('path=user/repo.git', call.kwargs['input'])
            self.assertEqual(call.kwargs['env']['GIT_TERMINAL_PROMPT'], '0')
            self.assertEqual(call.kwargs['env']['GCM_INTERACTIVE'], 'Never')
            self.assertEqual(call.kwargs['env']['GH_PROMPT_DISABLED'], '1')
            self.assertEqual(call.kwargs['env']['GIT_ASKPASS'], '')
            self.assertEqual(call.kwargs['timeout'], 15)

    def test_network_errors_do_not_echo_credentials_or_urls(self):
        secret = 'SENSITIVE-SECRET'
        opener = Mock()
        opener.open.side_effect = urllib.error.HTTPError('https://example.org/' + secret, 404, secret, {}, None)
        with (patch.object(downloader, 'get_token', return_value=secret),
              patch.object(downloader.urllib.request, 'build_opener', return_value=opener)):
            result, stdout, stderr = self.run_main()
        self.assertEqual(result, 1)
        self.assertIn('HTTP 404', stderr)
        self.assertNotIn(secret, stdout + stderr)


if __name__ == '__main__':
    unittest.main()
