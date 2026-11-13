"""Archive/table checks only; no training or policy evaluation."""
import copy
import csv
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_tables import export_tables, held_out_rows, robustness_rows, verify


class TableExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = verify()

    def test_held_out_known_means(self):
        rows = {(row['method'], row['condition']): row for row in held_out_rows(self.data)}
        self.assertEqual(len(rows), 16)
        self.assertAlmostEqual(rows['hoflon', 'test_ordinary']['mean'], 29.1, delta=.05)
        self.assertAlmostEqual(rows['td3-bc', 'test_stronger']['mean'], 64.1, delta=.05)
        pivot = rows['core-pivot', 'test_stronger']
        self.assertAlmostEqual(pivot['mean'], 77.5, delta=.05)
        self.assertAlmostEqual(pivot['sample_sd'], 22.7, delta=.05)
        self.assertEqual(pivot['n_training_seeds'], 5)
        self.assertEqual(pivot['n_episodes'], 25)

    def test_pd_variability_is_across_winds(self):
        row = next(r for r in held_out_rows(self.data)
                   if r['method'] == 'pd11' and r['condition'] == 'test_ordinary')
        raw = [r['cost'] for r in self.data['original_test']['rows']
               if r['method'] == 'pd11' and not r['stronger']]
        self.assertEqual(row['n_training_seeds'], 0)
        self.assertEqual(row['n_episodes'], 15)
        self.assertAlmostEqual(row['sample_sd'], statistics.stdev(raw))
        self.assertEqual(row['variability'], 'wind_realisations_fixed_PD')

    def test_robustness_counts_and_seed_statistics(self):
        rows = robustness_rows(self.data)
        self.assertEqual(len(rows), 17*9)
        row = next(r for r in rows if r['condition'] == 'severity_1.5' and r['method'] == 'core-pivot')
        values = [row[f'seed_{s}'] for s in range(123, 128)]
        self.assertAlmostEqual(row['mean'], 81.19187436427146)
        self.assertAlmostEqual(row['sample_sd'], statistics.stdev(values))
        self.assertEqual(row['minimum'], min(values))
        self.assertEqual(row['maximum'], max(values))

    def test_unavailable_pd_robustness_range_is_not_invented(self):
        for row in robustness_rows(self.data):
            if row['method'] == 'pd11':
                self.assertIsNone(row['sample_sd'])
                self.assertIsNone(row['minimum'])
                self.assertIsNone(row['maximum'])

    def test_csv_outputs_and_blank_values(self):
        with tempfile.TemporaryDirectory() as directory:
            export_tables(self.data, directory)
            with (Path(directory)/'held_out_test.csv').open(newline='', encoding='utf-8') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 16)
            with (Path(directory)/'robustness_analysis.csv').open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 153)
            pd = next(r for r in rows if r['method'] == 'pd11')
            self.assertEqual(pd['sample_sd'], '')
            self.assertEqual(pd['minimum'], '')

    def test_incomplete_seed_group_rejected(self):
        data = copy.deepcopy(self.data)
        del data['robustness']['conditions']['severity_1.5']['core-pivot']['seed_means']['123']
        with self.assertRaises(ValueError):
            robustness_rows(data)


if __name__ == '__main__':
    unittest.main()
