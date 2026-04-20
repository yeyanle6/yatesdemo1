import csv
import tempfile
import unittest
from pathlib import Path

from evaluate_dataset import load_ecg_series


class ECGAlignmentTests(unittest.TestCase):
    def _write_csv(self, rows):
        fd, path = tempfile.mkstemp(suffix='.csv')
        Path(path).unlink(missing_ok=True)
        p = Path(path)
        with p.open('w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['time', 'RRI', 'HF', 'LF/HF', 'LF ratio', 'HR'])
            for r in rows:
                w.writerow(r)
        return p

    def test_timestamp_alignment_interpolates_missing_seconds(self):
        p = self._write_csv([
            ['09:00:00', '1000', '120', '1.0', '50', '60'],
            ['09:00:01', '990', '110', '1.1', '52', '61'],
            ['09:00:03', '970', '90', '1.3', '57', '63'],
        ])
        try:
            series = load_ecg_series(p, align_mode='timestamp')
            self.assertIn(2, series)
            self.assertEqual(series[2].source, 'interpolated')
            self.assertTrue(series[2].hr_interpolated)
            self.assertAlmostEqual(series[2].hr or 0.0, 62.0, places=5)
            self.assertAlmostEqual(series[2].hf or 0.0, 100.0, places=5)
            self.assertAlmostEqual(series[2].lfhf or 0.0, 1.2, places=5)
        finally:
            p.unlink(missing_ok=True)

    def test_index_alignment_keeps_dense_index(self):
        p = self._write_csv([
            ['09:00:00', '1000', '120', '1.0', '50', '60'],
            ['09:00:02', '990', '110', '1.1', '52', '61'],
            ['09:00:04', '980', '100', '1.2', '54', '62'],
        ])
        try:
            series = load_ecg_series(p, align_mode='index')
            self.assertEqual(sorted(series.keys()), [0, 1, 2])
            self.assertEqual(series[1].source, 'observed')
            self.assertAlmostEqual(series[1].hr or 0.0, 61.0, places=5)
        finally:
            p.unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
