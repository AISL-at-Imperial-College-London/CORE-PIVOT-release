"""Reviewer-facing figure manifest and source-package checks; no model training."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CurrentFigureTests(unittest.TestCase):
    def test_manifest_covers_current_manuscript(self):
        manifest = json.loads((ROOT / 'assets/paper_figures/manifest.json').read_text())
        figures = manifest['figures']
        self.assertEqual([item['id'] for item in figures], list(range(1, 20)))
        stems = [item['stem'] for item in figures]
        self.assertEqual(len(stems), len(set(stems)))
        for item in figures:
            self.assertEqual(Path(item['stem']).name, item['stem'])
            self.assertTrue(item['caption'].strip())
            self.assertTrue(item['kind'])

    def test_list_needs_only_standard_library(self):
        code = r'''
import runpy, sys
def no_scientific_import(event, args):
    if event == 'import' and args[0].split('.')[0] in {'torch', 'numpy', 'scipy', 'matplotlib'}:
        raise AssertionError('Listing figures must not import scientific dependencies: ' + args[0])
sys.addaudithook(no_scientific_import)
sys.argv = ['scripts/reproduce_paper.py', '--list']
runpy.run_path('scripts/reproduce_paper.py', run_name='__main__')
'''
        result = subprocess.run([sys.executable, '-S', '-c', code], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('19', result.stdout)
        self.assertIn('10', result.stdout)

    def test_archived_plot_data_hashes(self):
        directory = ROOT / 'results/paper_data'
        for source in ('provenance.json', 'supplement_provenance.json'):
            manifest = json.loads((directory / source).read_text())
            for name, expected in manifest['generated_files'].items():
                with self.subTest(name=name):
                    self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), expected)

    def test_release_has_no_timing_only_artifacts(self):
        from scripts.package_release import release_files
        files = release_files()
        for path in files:
            relative = path.relative_to(ROOT)
            self.assertNotIn('runs', relative.parts)
            self.assertNotIn('outputs', relative.parts)
            self.assertNotIn('.venv', relative.parts)
            name = path.stem.lower()
            self.assertFalse(name.startswith(('timing_', 'computational_cost', 'deployment_timing')),
                             str(relative))
        self.assertTrue(any(p.suffix == '.tex' and 'paper_sources' in p.parts for p in files))


if __name__ == '__main__':
    unittest.main()
