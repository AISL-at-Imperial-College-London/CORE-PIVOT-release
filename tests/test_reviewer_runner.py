"""Launcher tests use fake checkpoint bytes and mocked subprocesses, never fits."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import run


class ReviewerRunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        shutil.copytree(run.ROOT / 'configs', self.root / 'configs')
        self.manifest = self.root / 'checkpoints' / 'selected' / 'manifest.json'
        self.manifest.parent.mkdir(parents=True)
        self.entries = []
        selected = run.read_json(self.root / 'configs' / 'experiments.json')['selected']
        for method in run.METHODS[:-1]:
            for seed in run.SEEDS:
                entry = dict(method=method, seed=seed, step=selected[method]['checkpoint'],
                             path=f'{method}_s{seed}.pt', sha256=hashlib.sha256(b'weights').hexdigest())
                if method in ('easybcq', 'hoflon'):
                    entry['bc_checkpoint'] = dict(path=f'bc_s{seed}.pt', seed=seed, step=500000,
                                                 sha256=hashlib.sha256(b'weights').hexdigest())
                self.entries.append(entry)
        self.save_manifest()

    def save_manifest(self):
        self.manifest.write_text(json.dumps(dict(schema_version=1, checkpoints=self.entries)))

    def args(self, *argv):
        return run.make_parser().parse_args(argv)

    def call(self, *argv):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run.main(argv, root=self.root)

    def create_data(self):
        folder = self.root / 'data' / 'paper'
        folder.mkdir(parents=True, exist_ok=True)
        for name in ('manifest.json', 'normalization.json'):
            (folder / name).write_text('{}')
        for name in run.REPLAY_FIELDS:
            (folder / f'{name}.npy').write_bytes(b'fake data; subprocess is mocked')
        return folder

    def create_native(self, algorithm, seed, step, *, complete=False):
        data = self.create_data()
        checkpoint = run.native_path(self.root / 'runs', algorithm, seed, step)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        identity = dict(algorithm=algorithm, seed=seed, smoke_only=False,
                        config=run.train_spec(self.root, algorithm)['config'],
                        data_manifest_sha256=run.sha256(data / 'manifest.json'),
                        normalization_sha256=run.sha256(data / 'normalization.json'))
        record = checkpoint.parent.parent / 'run.json'
        record.write_text(json.dumps(dict(identity=identity)))
        checkpoint.write_bytes(b'fake native checkpoint; no deserialization')
        checkpoint.with_suffix('.json').write_text(json.dumps(dict(
            step=step, model_sha256=run.sha256(checkpoint), run_sha256=run.sha256(record))))
        if complete:
            (record.parent / 'completed_step000500000.json').write_text(json.dumps(
                dict(step=500000, training_complete=True)))
        return checkpoint

    def test_single_training_plan_uses_paper_defaults(self):
        plan = run.build_training(self.args('train', 'iql', '--seed', '124', '--device', 'cuda'), self.root)
        self.assertEqual(len(plan.jobs), 1)
        command = plan.jobs[0].command
        self.assertEqual(command[command.index('--steps') + 1], '500000')
        self.assertEqual(command[command.index('--device') + 1], 'cuda')
        self.assertEqual(Path(command[command.index('--data-dir') + 1]), self.root / 'data' / 'paper')
        self.assertEqual(plan.jobs[0].output, self.root / 'runs' / 'iql_s124')

    def test_all_training_is_30_unique_fits_in_dependency_order(self):
        plan = run.build_training(self.args('train', 'all', '--all-seeds'), self.root)
        self.assertEqual(len(plan.jobs), 30)
        self.assertEqual(len({job.output for job in plan.jobs}), 30)
        outputs = [job.output.name for job in plan.jobs]
        for seed in run.SEEDS:
            self.assertLess(outputs.index(f'bc_s{seed}'), outputs.index(f'qbeta_s{seed}'))
            self.assertLess(outputs.index('bc_s123'), outputs.index(f'core-pivot_s{seed}'))

    def test_hoflon_and_easybcq_share_fit_and_matched_bc(self):
        plans = [run.build_training(self.args('train', method, '--seed', '127'), self.root)
                 for method in ('easybcq', 'hoflon')]
        self.assertEqual([job.command for job in plans[0].jobs], [job.command for job in plans[1].jobs])
        self.assertEqual([job.output.name for job in plans[0].jobs], ['bc_s127', 'qbeta_s127'])

    def test_core_always_depends_on_bc123(self):
        plan = run.build_training(self.args('train', 'core-pivot', '--seed', '127'), self.root)
        self.assertEqual([job.output.name for job in plan.jobs], ['bc_s123', 'core-pivot_s127'])
        self.assertIn(str(run.native_path(self.root / 'runs', 'bc', 123, 500000)), plan.jobs[-1].command)

    @patch('run.subprocess.run')
    def test_dry_runs_never_execute_or_create_outputs(self, launch):
        for command in [('train', 'all', '--all-seeds'), ('evaluate', 'all', '--all-seeds'),
                        ('evaluate', 'all', '--source', 'trained'), ('evaluate', 'hoflon', '--condition', 'all')]:
            self.assertEqual(self.call(*command, '--dry-run'), 0)
        launch.assert_not_called()
        self.assertFalse((self.root / 'runs').exists())
        self.assertFalse((self.root / 'data').exists())

    @patch('run.subprocess.run')
    def test_best_pd_training_is_noop_and_evaluation_needs_no_weights(self, launch):
        self.assertEqual(self.call('train', 'best-pd', '--all-seeds'), 0)
        launch.assert_not_called()
        self.assertEqual(self.call('evaluate', 'best-pd', '--all-seeds'), 0)
        launch.assert_called_once()
        self.assertNotIn('--checkpoint', launch.call_args.args[0])
        self.assertIn('scripts/evaluate.py', launch.call_args.args[0])

    def test_selected_steps_search_and_proposal_are_explicit(self):
        plan = run.build_evaluation(self.args('evaluate', 'hoflon', '--seed', '124'), self.root)
        job = plan.jobs[0]
        self.assertEqual(len(job.hashes), 2)
        for flag, value in [('--expected-step', '310000'), ('--candidates', '100'),
                            ('--radius', '0.1'), ('--iterations', '40'), ('--split', 'validation')]:
            self.assertEqual(job.command[job.command.index(flag) + 1], value)
        self.assertEqual(job.hashes[1][0].name, 'bc_s124.pt')

    def test_all_evaluation_runs_fixed_pd_once(self):
        plan = run.build_evaluation(self.args('evaluate', 'all', '--all-seeds', '--condition', 'all'), self.root)
        self.assertEqual(len(plan.jobs), 36)
        self.assertEqual(sum('--checkpoint' not in job.command for job in plan.jobs), 1)
        self.assertTrue(all('scripts/evaluate.py' in job.command for job in plan.jobs))

    @patch('run.subprocess.run')
    def test_test_split_requires_explicit_gate_even_for_dry_run(self, launch):
        with self.assertRaises(SystemExit):
            self.call('evaluate', 'best-pd', '--split', 'test', '--dry-run')
        self.assertEqual(self.call('evaluate', 'best-pd', '--split', 'test', '--allow-test'), 0)
        self.assertIn('--allow-test', launch.call_args.args[0])

    @patch('run.subprocess.run')
    def test_missing_weights_never_trigger_training_or_download(self, launch):
        with self.assertRaises(SystemExit):
            self.call('evaluate', 'iql')
        launch.assert_not_called()

    @patch('run.subprocess.run')
    def test_primary_and_proposal_hashes_are_verified_before_execution(self, launch):
        (self.manifest.parent / 'hoflon_s123.pt').write_bytes(b'weights')
        (self.manifest.parent / 'bc_s123.pt').write_bytes(b'corrupt')
        with self.assertRaises(SystemExit):
            self.call('evaluate', 'hoflon')
        launch.assert_not_called()
        (self.manifest.parent / 'bc_s123.pt').write_bytes(b'weights')
        self.assertEqual(self.call('evaluate', 'hoflon'), 0)
        launch.assert_called_once()
        launch.reset_mock()
        (self.manifest.parent / 'hoflon_s123.pt').write_bytes(b'corrupt')
        with self.assertRaises(SystemExit):
            self.call('evaluate', 'hoflon')
        launch.assert_not_called()

    def test_manifest_rejects_wrong_proposal_seed_or_step_and_path_escape(self):
        entry = next(e for e in self.entries if e['method'] == 'hoflon' and e['seed'] == 123)
        for key, value in [('seed', 124), ('step', 1), ('path', '../outside.pt')]:
            original = entry['bc_checkpoint'][key]
            entry['bc_checkpoint'][key] = value
            self.save_manifest()
            with self.assertRaises(ValueError):
                run.build_evaluation(self.args('evaluate', 'hoflon'), self.root)
            entry['bc_checkpoint'][key] = original

    @patch('run.subprocess.run')
    def test_existing_incomplete_training_directory_is_not_overwritten(self, launch):
        (self.root / 'runs' / 'iql_s123').mkdir(parents=True)
        with self.assertRaises(SystemExit):
            self.call('train', 'iql')
        launch.assert_not_called()

    @patch('run.subprocess.run')
    def test_completed_fit_is_reused_without_launch(self, launch):
        self.create_native('iql', 123, 500000, complete=True)
        self.assertEqual(self.call('train', 'iql'), 0)
        launch.assert_not_called()

    @patch('run.subprocess.run')
    def test_reuse_rejects_changed_data(self, launch):
        self.create_native('iql', 123, 500000, complete=True)
        (self.root / 'data' / 'paper' / 'normalization.json').write_text('{"changed": true}')
        with self.assertRaises(SystemExit):
            self.call('train', 'iql')
        launch.assert_not_called()

    @patch('run.subprocess.run')
    def test_trained_source_only_evaluates_verified_native_selected_step(self, launch):
        self.create_native('qbeta', 125, 310000)
        self.create_native('bc', 125, 500000)
        self.assertEqual(self.call('evaluate', 'hoflon', '--source', 'trained', '--seed', '125'), 0)
        launch.assert_called_once()
        command = launch.call_args.args[0]
        self.assertIn('scripts/evaluate.py', command)
        self.assertIn('--expected-step', command)
        self.assertNotIn('scripts/train.py', command)

    @patch('run.subprocess.run')
    def test_existing_evaluation_or_trajectory_outputs_are_never_overwritten(self, launch):
        output = self.root / 'previous.json'
        output.write_text('{}')
        with self.assertRaises(SystemExit):
            self.call('evaluate', 'best-pd', '--output', str(output))
        trajectories = self.root / 'new_trajectories'
        trajectories.mkdir()
        with self.assertRaises(SystemExit):
            self.call('evaluate', 'best-pd', '--output', str(self.root / 'new.json'), '--trajectories')
        launch.assert_not_called()

    @patch('run.subprocess.run')
    def test_training_only_launches_after_complete_preflight(self, launch):
        with self.assertRaises(SystemExit):
            self.call('train', 'iql')
        launch.assert_not_called()
        self.create_data()
        self.assertEqual(self.call('train', 'iql'), 0)
        launch.assert_called_once()
        self.assertIn('scripts/train.py', launch.call_args.args[0])

    def test_single_output_is_rejected_for_multiple_evaluations(self):
        with self.assertRaises(ValueError):
            run.build_evaluation(self.args('evaluate', 'all', '--output', 'one.json'), self.root)


if __name__ == '__main__':
    unittest.main()
