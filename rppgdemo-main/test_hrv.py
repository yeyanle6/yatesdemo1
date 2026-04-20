#!/usr/bin/env python3
"""Extract BVP signal from video, detect peaks, and compute HRV metrics."""

import sys
import math
from collections import deque
from typing import Dict, Deque, List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import welch
from scipy.interpolate import interp1d

from main import (
    Config,
    RPPGAlgorithms,
    FusionEngine,
    SignalQualityController,
    assess_signal_quality,
    detect_peaks_and_pnn50,
    extract_rgb_mean,
    create_roi_extractor,
)


def compute_hrv_time_domain(ibi_ms: np.ndarray) -> dict:
    """Time-domain HRV metrics from IBI series (in ms)."""
    if len(ibi_ms) < 5:
        return {}

    nn = ibi_ms
    nn_diff = np.diff(nn)

    mean_rr = float(np.mean(nn))
    sdnn = float(np.std(nn, ddof=1))
    rmssd = float(np.sqrt(np.mean(nn_diff ** 2)))
    nn50 = int(np.sum(np.abs(nn_diff) > 50))
    pnn50 = float(nn50) / len(nn_diff) * 100.0 if len(nn_diff) > 0 else 0.0
    nn20 = int(np.sum(np.abs(nn_diff) > 20))
    pnn20 = float(nn20) / len(nn_diff) * 100.0 if len(nn_diff) > 0 else 0.0
    mean_hr = 60000.0 / mean_rr if mean_rr > 0 else 0.0

    return {
        "Mean RR (ms)": mean_rr,
        "SDNN (ms)": sdnn,
        "RMSSD (ms)": rmssd,
        "pNN50 (%)": pnn50,
        "pNN20 (%)": pnn20,
        "Mean HR (bpm)": mean_hr,
        "Min HR (bpm)": 60000.0 / np.max(nn) if np.max(nn) > 0 else 0,
        "Max HR (bpm)": 60000.0 / np.min(nn) if np.min(nn) > 0 else 0,
        "HR range (bpm)": 60000.0 / np.min(nn) - 60000.0 / np.max(nn) if np.min(nn) > 0 else 0,
    }


def compute_hrv_freq_domain(ibi_ms: np.ndarray, peak_times_sec: np.ndarray) -> dict:
    """Frequency-domain HRV from IBI series.

    Resample IBI to uniform 4 Hz, then Welch PSD.
    LF: 0.04-0.15 Hz, HF: 0.15-0.40 Hz.
    """
    if len(ibi_ms) < 20 or len(peak_times_sec) < 20:
        return {}

    # Build IBI time series: each IBI[i] is at peak_times_sec[i+1]
    ibi_times = peak_times_sec[1:]  # time of each IBI
    if len(ibi_times) != len(ibi_ms):
        ibi_times = ibi_times[:len(ibi_ms)]

    # Resample to uniform 4 Hz using cubic interpolation
    resample_fs = 4.0
    t_start = ibi_times[0]
    t_end = ibi_times[-1]
    duration = t_end - t_start
    if duration < 30:
        return {"error": f"Duration too short ({duration:.0f}s), need >=30s"}

    t_uniform = np.arange(t_start, t_end, 1.0 / resample_fs)
    interp_func = interp1d(ibi_times, ibi_ms, kind='cubic', fill_value='extrapolate')
    ibi_resampled = interp_func(t_uniform)

    # Remove mean (detrend)
    ibi_resampled = ibi_resampled - np.mean(ibi_resampled)

    # Welch PSD
    nperseg = min(256, len(ibi_resampled))
    freqs, psd = welch(ibi_resampled, fs=resample_fs, nperseg=nperseg,
                       noverlap=nperseg // 2, nfft=1024)

    # Frequency bands
    vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)
    total_mask = (freqs >= 0.003) & (freqs < 0.40)

    # Band powers (ms^2)
    df = freqs[1] - freqs[0]
    vlf_power = float(np.trapz(psd[vlf_mask], freqs[vlf_mask])) if np.any(vlf_mask) else 0.0
    lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if np.any(lf_mask) else 0.0
    hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else 0.0
    total_power = vlf_power + lf_power + hf_power

    # Peak frequencies
    lf_peak = float(freqs[lf_mask][np.argmax(psd[lf_mask])]) if np.any(lf_mask) and np.max(psd[lf_mask]) > 0 else 0.0
    hf_peak = float(freqs[hf_mask][np.argmax(psd[hf_mask])]) if np.any(hf_mask) and np.max(psd[hf_mask]) > 0 else 0.0

    # Normalized units (excluding VLF)
    lf_hf_total = lf_power + hf_power
    lf_nu = (lf_power / lf_hf_total * 100.0) if lf_hf_total > 0 else 0.0
    hf_nu = (hf_power / lf_hf_total * 100.0) if lf_hf_total > 0 else 0.0
    lf_hf_ratio = lf_power / hf_power if hf_power > 1e-10 else float('inf')

    return {
        "VLF power (ms²)": vlf_power,
        "LF power (ms²)": lf_power,
        "HF power (ms²)": hf_power,
        "Total power (ms²)": total_power,
        "LF norm (n.u.)": lf_nu,
        "HF norm (n.u.)": hf_nu,
        "LF/HF ratio": lf_hf_ratio,
        "LF peak (Hz)": lf_peak,
        "HF peak (Hz)": hf_peak,
        "Analysis duration (s)": duration,
    }


def process_video(video_path: str, mp_model: str = "") -> Tuple[np.ndarray, float, List[int]]:
    """Process video and return full merged BVP signal, sampling rate, and peaks."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    print(f"Video: {video_path}")
    print(f"  FPS={video_fps:.2f}  frames={total_frames}  duration={duration:.1f}s")

    cfg = Config()
    cfg.roi_preset = "hybrid7"

    extractor, roi_mode_used = create_roi_extractor(
        "auto", mp_face_detector_model=mp_model, strict=False,
        roi_preset=cfg.roi_preset,
        roi_scale_x=cfg.roi_scale_x,
        roi_scale_y=cfg.roi_scale_y,
        roi_shift_y=cfg.roi_shift_y,
    )
    print(f"  ROI mode={roi_mode_used}")

    alg = RPPGAlgorithms(cfg)

    roi_buf: Dict[str, Dict[str, Deque[float]]] = {}
    roi_weights: Dict[str, float] = {}

    fs = video_fps
    frame_idx = 0

    print("  Processing frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        rois = extractor.extract(frame)
        if not rois:
            continue

        for roi in rois:
            rgb = extract_rgb_mean(frame, roi)
            if rgb is None:
                continue
            if roi.name not in roi_buf:
                max_len = int(duration * video_fps) + 100
                roi_buf[roi.name] = {
                    "r": deque(maxlen=max_len),
                    "g": deque(maxlen=max_len),
                    "b": deque(maxlen=max_len),
                }
            roi_buf[roi.name]["r"].append(rgb[0])
            roi_buf[roi.name]["g"].append(rgb[1])
            roi_buf[roi.name]["b"].append(rgb[2])
            roi_weights[roi.name] = roi.weight

        if frame_idx % 600 == 0:
            print(f"    {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.0f}%)")

    cap.release()
    print(f"  Done. {frame_idx} frames processed, {len(roi_buf)} ROIs.")

    # Merge ROIs with weights -> single BVP
    if not roi_buf:
        print("ERROR: no ROI data extracted")
        sys.exit(1)

    # Filter out ROIs with too few samples (need at least 50% of max)
    lengths = {name: len(buf["r"]) for name, buf in roi_buf.items()}
    max_len = max(lengths.values())
    min_required = int(max_len * 0.5)
    valid_rois = {name: buf for name, buf in roi_buf.items() if lengths[name] >= min_required}
    print(f"  ROI lengths: {', '.join(f'{n}={l}' for n, l in sorted(lengths.items()))}")
    print(f"  Using {len(valid_rois)}/{len(roi_buf)} ROIs (>={min_required} samples)")

    min_common = min(len(buf["r"]) for buf in valid_rois.values())
    print(f"  Common signal length: {min_common} samples ({min_common/fs:.1f}s)")

    weights = []
    r_all, g_all, b_all = [], [], []
    for name, buf in valid_rois.items():
        r_all.append(np.array(list(buf["r"]), dtype=np.float64)[-min_common:])
        g_all.append(np.array(list(buf["g"]), dtype=np.float64)[-min_common:])
        b_all.append(np.array(list(buf["b"]), dtype=np.float64)[-min_common:])
        weights.append(roi_weights.get(name, 1.0))

    w = np.array(weights, dtype=np.float64)
    w = w / np.sum(w)
    r_mix = np.sum(np.vstack(r_all) * w[:, None], axis=0)
    g_mix = np.sum(np.vstack(g_all) * w[:, None], axis=0)
    b_mix = np.sum(np.vstack(b_all) * w[:, None], axis=0)

    # Extract BVP using POS (best for peak detection)
    _, bvp_pos = alg.pos(r_mix, g_mix, b_mix, fs)
    # Also CBCR
    _, bvp_cbcr = alg.cbcr_pos(r_mix, g_mix, b_mix, fs)

    return bvp_pos, bvp_cbcr, fs, r_mix, g_mix, b_mix


def analyze_bvp(bvp: np.ndarray, fs: float, algo_name: str):
    """Detect peaks in BVP and compute HRV."""

    print(f"\n{'='*60}")
    print(f"  Algorithm: {algo_name}")
    print(f"  Signal length: {len(bvp)} samples ({len(bvp)/fs:.1f}s)")
    print(f"{'='*60}")

    # Detect peaks
    _, pnn50_raw, peaks = detect_peaks_and_pnn50(bvp, fs, hr_min_sec=3.0, pnn50_min_sec=10.0)

    print(f"\n  Peaks detected: {len(peaks)}")
    if len(peaks) < 10:
        print("  ERROR: too few peaks for HRV analysis")
        return

    # IBI in ms
    ibi_ms = np.array([(peaks[i] - peaks[i-1]) * (1000.0 / fs) for i in range(1, len(peaks))])
    peak_times = np.array([p / fs for p in peaks])

    # Filter physiologically valid IBIs (400-1500 ms = 40-150 bpm)
    valid_mask = (ibi_ms >= 400) & (ibi_ms <= 1500)
    valid_ratio = np.sum(valid_mask) / len(ibi_ms) * 100
    print(f"  Valid IBIs: {np.sum(valid_mask)}/{len(ibi_ms)} ({valid_ratio:.1f}%)")
    print(f"  IBI range: {np.min(ibi_ms):.0f} - {np.max(ibi_ms):.0f} ms")

    if valid_ratio < 50:
        print("  WARNING: <50% valid IBIs, results unreliable")

    # Use all IBIs for analysis (already corrected by detect_peaks_and_pnn50)
    print(f"\n--- Time-Domain HRV ---")
    td = compute_hrv_time_domain(ibi_ms)
    for k, v in td.items():
        print(f"  {k:>20s}: {v:8.2f}")

    print(f"\n--- Frequency-Domain HRV ---")
    fd = compute_hrv_freq_domain(ibi_ms, peak_times)
    for k, v in fd.items():
        if isinstance(v, str):
            print(f"  {k:>25s}: {v}")
        elif v == float('inf'):
            print(f"  {k:>25s}:      inf")
        else:
            print(f"  {k:>25s}: {v:10.3f}")

    # Print IBI histogram
    print(f"\n--- IBI Distribution ---")
    hist_bins = [400, 500, 600, 650, 700, 750, 800, 850, 900, 1000, 1200, 1500]
    counts, _ = np.histogram(ibi_ms, bins=hist_bins)
    for i in range(len(counts)):
        bar = "#" * min(counts[i], 60)
        print(f"  {hist_bins[i]:4d}-{hist_bins[i+1]:4d}ms: {counts[i]:4d} {bar}")

    return td, fd


def main():
    import argparse
    parser = argparse.ArgumentParser(description="rPPG HRV analysis from video")
    parser.add_argument("video", help="path to video file")
    parser.add_argument("--mp-face-detector-model", default="models/blaze_face_short_range.tflite",
                        help="path to MediaPipe face detector .tflite model")
    args = parser.parse_args()

    bvp_pos, bvp_cbcr, fs, r, g, b = process_video(args.video, mp_model=args.mp_face_detector_model)

    analyze_bvp(bvp_pos, fs, "POS")
    analyze_bvp(bvp_cbcr, fs, "CBCR")


if __name__ == "__main__":
    main()
