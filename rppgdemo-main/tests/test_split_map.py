import unittest
from pathlib import Path

from evaluate_dataset import Sample, build_split_map


class SplitMapTests(unittest.TestCase):
    def test_holdout_list_marks_only_requested_samples(self):
        samples = [
            Sample(group='001', stem='3-1', video_path=Path('/tmp/001_3-1.mp4'), ecg_csv_path=Path('/tmp/001_3-1.csv')),
            Sample(group='001', stem='3-3', video_path=Path('/tmp/001_3-3.mp4'), ecg_csv_path=Path('/tmp/001_3-3.csv')),
            Sample(group='002', stem='3-4', video_path=Path('/tmp/002_3-4.mp4'), ecg_csv_path=Path('/tmp/002_3-4.csv')),
        ]
        split = build_split_map(samples, holdout_list='001/3-3,002/3-4')
        self.assertEqual(split['001/3-1'], 'train')
        self.assertEqual(split['001/3-3'], 'test')
        self.assertEqual(split['002/3-4'], 'test')


if __name__ == '__main__':
    unittest.main()
