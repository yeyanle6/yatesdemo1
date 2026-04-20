import math
import unittest

import numpy as np

from evaluate_dataset import compute_lf_metrics_from_bvp
from main import Config


def synth_bvp(fs: float, duration_sec: float, lf_amp: float, hf_amp: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = 0.0
    peak_times = []
    while t < duration_sec:
        rr = 1.0 + lf_amp * math.sin(2.0 * math.pi * 0.10 * t) + hf_amp * math.sin(2.0 * math.pi * 0.25 * t)
        rr = min(1.5, max(0.55, rr))
        t += rr
        peak_times.append(t)

    n = int(duration_sec * fs)
    signal = np.zeros(n, dtype=np.float64)
    for pt in peak_times:
        idx = int(pt * fs)
        if 0 <= idx < n:
            signal[idx] = 1.0

    kernel = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    kernel /= np.sum(kernel)
    signal = np.convolve(signal, kernel, mode='same')
    signal += 0.0008 * rng.normal(size=n)
    return signal


class LFFrequencyConsistencyTests(unittest.TestCase):
    def test_lf_dominant_signal_has_higher_lfhf_ratio(self):
        fs = 30.0
        dur = 130.0
        cfg = Config()

        bvp_lf = synth_bvp(fs, dur, lf_amp=0.11, hf_amp=0.02, seed=7)
        bvp_hf = synth_bvp(fs, dur, lf_amp=0.02, hf_amp=0.11, seed=11)

        hf1, lfhf1, lfr1 = compute_lf_metrics_from_bvp(bvp_lf, fs, lf_window_sec=60.0, lf_resample_fs=4.0, cfg=cfg)
        hf2, lfhf2, lfr2 = compute_lf_metrics_from_bvp(bvp_hf, fs, lf_window_sec=60.0, lf_resample_fs=4.0, cfg=cfg)

        self.assertIsNotNone(hf1)
        self.assertIsNotNone(hf2)
        self.assertIsNotNone(lfhf1)
        self.assertIsNotNone(lfhf2)
        self.assertIsNotNone(lfr1)
        self.assertIsNotNone(lfr2)
        self.assertGreater(lfhf1, lfhf2)
        self.assertGreater(lfr1, lfr2)


if __name__ == '__main__':
    unittest.main()
