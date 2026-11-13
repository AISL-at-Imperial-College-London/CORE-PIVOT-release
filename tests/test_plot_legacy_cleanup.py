"""Generated-file migration checks; no model training or evaluation."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    'paper_plots', Path(__file__).resolve().parents[1]/'scripts/plot_paper.py')
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


class LegacyCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def files(self, names):
        for name in names:
            (self.root/name).write_text('fixture')

    def test_only_exact_names_for_current_plot_are_removed(self):
        old = ['original_test_violins.'+ext for ext in ('svg','pdf','png')]
        new = ['held_out_test_performance.'+ext for ext in ('svg','pdf','png')]
        keep = ['original_test_violins_custom.svg', 'robustness.svg',
                'original_test_summary.json', 'notes.txt']
        self.files(old+new+keep)
        plots.remove_legacy_outputs(self.root, 'held_out_test_performance', ('svg','pdf','png'))
        self.assertTrue(all(not (self.root/name).exists() for name in old))
        self.assertTrue(all((self.root/name).exists() for name in new+keep))

    def test_missing_replacement_preserves_legacy(self):
        self.files(['robustness.svg','robustness_analysis.svg'])
        plots.remove_legacy_outputs(self.root, 'robustness_analysis', ('svg','pdf','png'))
        self.assertTrue((self.root/'robustness.svg').exists())

    def test_failed_save_preserves_legacy(self):
        self.files(['robustness.svg'])
        class BrokenFigure:
            def savefig(self, *args, **kwargs):
                raise OSError('simulated disk error')
        with self.assertRaises(OSError):
            plots.save(BrokenFigure(), self.root, 'robustness_analysis')
        self.assertTrue((self.root/'robustness.svg').exists())

    def test_successful_save_removes_legacy(self):
        self.files(['robustness.svg','robustness.pdf','robustness.png'])
        class Figure:
            def savefig(self, path, **kwargs):
                path.write_text('new plot')
        with patch.object(plots.plt, 'close'):
            plots.save(Figure(), self.root, 'robustness_analysis')
        self.assertTrue(all(not (self.root/('robustness.'+ext)).exists()
                            for ext in ('svg','pdf','png')))

    def test_summary_and_repeated_cleanup(self):
        self.files(['original_test_summary.json','held_out_test_summary.json'])
        for _ in range(2):
            plots.remove_legacy_outputs(self.root, 'held_out_test_summary', ('json',))
        self.assertFalse((self.root/'original_test_summary.json').exists())
        self.assertTrue((self.root/'held_out_test_summary.json').exists())

    def test_directory_with_legacy_name_is_preserved(self):
        (self.root/'robustness.svg').mkdir()
        self.files(['robustness_analysis.svg'])
        plots.remove_legacy_outputs(self.root, 'robustness_analysis', ('svg',))
        self.assertTrue((self.root/'robustness.svg').is_dir())


if __name__ == '__main__':
    unittest.main()
