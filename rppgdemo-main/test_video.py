#!/usr/bin/env python3
"""Offline rPPG test: run algorithms on a video file and compare with ground-truth CSV."""

import csv
import sys
import time
from collections import deque
from typing import Dict, Deque, List, Optional, Tuple

import cv2
import numpy as np

# Reuse everything from main.py
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


def load_ground_truth(csv_path: str) -> List[Tuple[int, float]]:
    """Load HR ground truth from CSV.  Returns [(second_index, hr), ...]."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            hr = row.get("HR", "").strip()
            if not hr:
                break
            rows.append((i, float(hr)))
    return rows


def run_video(video_path: str, roi_mode: str = "auto",
              roi_preset: str = "hybrid7",
              mp_model: str = "") -> List[Tuple[float, float]]:
    """Process video and return [(elapsed_sec, hr_bpm), ...] at 1-second intervals."""

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
    cfg.roi_preset = roi_preset

    extractor, roi_mode_used = create_roi_extractor(
        roi_mode,
        mp_face_detector_model=mp_model,
        strict=False,
        roi_preset=cfg.roi_preset,
        roi_scale_x=cfg.roi_scale_x,
        roi_scale_y=cfg.roi_scale_y,
        roi_shift_y=cfg.roi_shift_y,
    )
    print(f"  ROI mode={roi_mode_used}")

    alg = RPPGAlgorithms(cfg)
    fusion = FusionEngine(cfg)
    quality_ctl = SignalQualityController(publish_interval=1.0)

    roi_buf: Dict[str, Dict[str, Deque[float]]] = {}
    roi_weights: Dict[str, float] = {}

    fs = video_fps
    hr_best = 0.0
    sqi = 0.0
    snr_db = 0.0
    freq_conf = 0.0
    ppi_hr = None
    pnn50 = None

    compute_interval = 0.5
    last_compute_t = -1.0
    last_print_t = -1.0

    results: List[Tuple[float, float]] = []
    # Per-algorithm results for comparison
    algo_results: Dict[str, List[Tuple[float, float]]] = {
        "GREEN": [], "CHROM": [], "POS": [], "CBCR": [], "FUSED": []
    }

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = frame_idx / video_fps
        frame_idx += 1

        rois = extractor.extract(frame)
        if not rois:
            if elapsed - last_print_t >= 1.0:
                last_print_t = elapsed
            continue

        for roi in rois:
            rgb = extract_rgb_mean(frame, roi)
            if rgb is None:
                continue
            if roi.name not in roi_buf:
                max_len = int(cfg.buffer_sec * video_fps)
                roi_buf[roi.name] = {
                    "r": deque(maxlen=max_len),
                    "g": deque(maxlen=max_len),
                    "b": deque(maxlen=max_len),
                }
            roi_buf[roi.name]["r"].append(rgb[0])
            roi_buf[roi.name]["g"].append(rgb[1])
            roi_buf[roi.name]["b"].append(rgb[2])
            roi_weights[roi.name] = roi.weight

        if elapsed - last_compute_t >= compute_interval:
            last_compute_t = elapsed

            tagged: List[Tuple[float, str, str]] = []
            merged_rgb: Dict[str, List[np.ndarray]] = {"r": [], "g": [], "b": []}
            merged_w: List[float] = []

            min_len = int(max(cfg.welch_seg_sec * fs, cfg.pos_window_sec * fs) + 5)
            max_compute_len = int(cfg.welch_seg_sec * fs * 2.5) + 10

            # Track per-algo medians for this compute cycle
            cycle_algo: Dict[str, List[float]] = {"GREEN": [], "CHROM": [], "POS": [], "CBCR": []}

            for name, buf in roi_buf.items():
                if len(buf["g"]) < min_len:
                    continue

                r = np.array(buf["r"], dtype=np.float64)[-max_compute_len:]
                g = np.array(buf["g"], dtype=np.float64)[-max_compute_len:]
                b = np.array(buf["b"], dtype=np.float64)[-max_compute_len:]

                hr_g, _ = alg.green(r, g, b, fs)
                hr_c, _ = alg.chrom(r, g, b, fs)
                hr_p, _ = alg.pos(r, g, b, fs)
                hr_cb, _ = alg.cbcr_pos(r, g, b, fs)

                for hr_val, algo_name in [(hr_g, "GREEN"), (hr_c, "CHROM"),
                                           (hr_p, "POS"), (hr_cb, "CBCR")]:
                    if cfg.min_bpm_valid <= hr_val <= cfg.max_bpm_valid:
                        tagged.append((hr_val, algo_name, name))
                        cycle_algo[algo_name].append(hr_val)

                w = roi_weights.get(name, 1.0)
                merged_rgb["r"].append(r)
                merged_rgb["g"].append(g)
                merged_rgb["b"].append(b)
                merged_w.append(w)

            hr_final: Optional[float] = None
            if merged_w:
                min_common = min(len(x) for x in merged_rgb["r"])
                if min_common >= min_len:
                    wt = np.array(merged_w, dtype=np.float64)
                    wt = wt / (np.sum(wt) + 1e-9)
                    rr = np.vstack([x[-min_common:] for x in merged_rgb["r"]])
                    gg = np.vstack([x[-min_common:] for x in merged_rgb["g"]])
                    bb = np.vstack([x[-min_common:] for x in merged_rgb["b"]])
                    r_mix = np.sum(rr * wt[:, None], axis=0)
                    g_mix = np.sum(gg * wt[:, None], axis=0)
                    b_mix = np.sum(bb * wt[:, None], axis=0)
                    quality_signal = alg.bandpass(alg.robust_norm(alg.detrend_poly2(g_mix)), fs)
                    sqi, snr_db = assess_signal_quality(quality_signal, fs)
                    _, bvp = alg.pos(r_mix, g_mix, b_mix, fs)
                    ppi_hr, pnn50, _ = detect_peaks_and_pnn50(bvp, fs)

            if tagged:
                hr_raw, freq_conf = fusion.harmonic_temporal_fusion(tagged)
                hr_final = fusion.apply_physiological_constraints(hr_raw)

            if hr_final is not None and cfg.enable_ppi_assist:
                hr_final = fusion.apply_ppi_assist(hr_final, ppi_hr, sqi, freq_conf=freq_conf)
            if hr_final is not None:
                hr_best = fusion.update_and_get_best(hr_final)

            # Record per-algo median for this cycle
            for algo_name in ["GREEN", "CHROM", "POS", "CBCR"]:
                vals = cycle_algo[algo_name]
                if vals:
                    med = float(np.median(vals))
                    algo_results[algo_name].append((elapsed, med))

        if elapsed - last_print_t >= 1.0:
            last_print_t = elapsed
            results.append((elapsed, hr_best))
            algo_results["FUSED"].append((elapsed, hr_best))
            print(f"  t={elapsed:5.1f}s  HR={hr_best:6.1f}  SQI={sqi:.3f}  SNR={snr_db:.1f}  FCONF={freq_conf:.3f}")

    cap.release()
    return results, algo_results


def compare(results: List[Tuple[float, float]],
            algo_results: Dict[str, List[Tuple[float, float]]],
            gt: List[Tuple[int, float]]) -> None:
    """Print comparison table and error metrics."""

    # Build GT dict: second -> HR
    gt_dict = {sec: hr for sec, hr in gt}

    print("\n" + "=" * 90)
    print(f"{'Time':>6s}  {'GT':>6s}  {'FUSED':>6s}  {'Err':>6s}  "
          f"{'GREEN':>6s}  {'CHROM':>6s}  {'POS':>6s}  {'CBCR':>6s}")
    print("-" * 90)

    # Build algo lookup: algo -> {time_int -> hr}
    algo_at = {}
    for algo_name in ["GREEN", "CHROM", "POS", "CBCR"]:
        lookup = {}
        for t, hr in algo_results.get(algo_name, []):
            lookup[int(round(t))] = hr
        algo_at[algo_name] = lookup

    errors = {"FUSED": [], "GREEN": [], "CHROM": [], "POS": [], "CBCR": []}

    for t_sec, fused_hr in results:
        t_int = int(round(t_sec))
        gt_hr = gt_dict.get(t_int)
        if gt_hr is None:
            continue

        err = fused_hr - gt_hr

        algo_vals = {}
        for algo_name in ["GREEN", "CHROM", "POS", "CBCR"]:
            v = algo_at[algo_name].get(t_int)
            algo_vals[algo_name] = v

        green_s = f"{algo_vals['GREEN']:6.1f}" if algo_vals['GREEN'] else "    --"
        chrom_s = f"{algo_vals['CHROM']:6.1f}" if algo_vals['CHROM'] else "    --"
        pos_s = f"{algo_vals['POS']:6.1f}" if algo_vals['POS'] else "    --"
        cbcr_s = f"{algo_vals['CBCR']:6.1f}" if algo_vals['CBCR'] else "    --"

        print(f"{t_int:5d}s  {gt_hr:6.1f}  {fused_hr:6.1f}  {err:+6.1f}  "
              f"{green_s}  {chrom_s}  {pos_s}  {cbcr_s}")

        errors["FUSED"].append(abs(err))
        for algo_name in ["GREEN", "CHROM", "POS", "CBCR"]:
            v = algo_vals[algo_name]
            if v is not None:
                errors[algo_name].append(abs(v - gt_hr))

    print("=" * 90)
    print("\n--- Error Summary (vs ground truth) ---")
    print(f"{'Algorithm':>10s}  {'MAE':>7s}  {'RMSE':>7s}  {'Max':>7s}  {'Samples':>7s}")
    print("-" * 48)
    for algo_name in ["FUSED", "GREEN", "CHROM", "POS", "CBCR"]:
        errs = errors[algo_name]
        if not errs:
            continue
        e = np.array(errs)
        mae = float(np.mean(e))
        rmse = float(np.sqrt(np.mean(e ** 2)))
        mx = float(np.max(e))
        print(f"{algo_name:>10s}  {mae:7.2f}  {rmse:7.2f}  {mx:7.2f}  {len(errs):>7d}")


def print_analysis(results: List[Tuple[float, float]],
                   algo_results: Dict[str, List[Tuple[float, float]]]) -> None:
    """Print per-algorithm analysis without ground truth."""

    # Build algo lookup: algo -> {time_int -> hr}
    algo_at = {}
    for algo_name in ["GREEN", "CHROM", "POS", "CBCR"]:
        lookup = {}
        for t, hr in algo_results.get(algo_name, []):
            lookup[int(round(t))] = hr
        algo_at[algo_name] = lookup

    print("\n" + "=" * 80)
    print(f"{'Time':>6s}  {'FUSED':>6s}  {'GREEN':>6s}  {'CHROM':>6s}  {'POS':>6s}  {'CBCR':>6s}")
    print("-" * 80)

    for t_sec, fused_hr in results:
        t_int = int(round(t_sec))
        green_s = f"{algo_at['GREEN'].get(t_int, 0):6.1f}" if algo_at['GREEN'].get(t_int) else "    --"
        chrom_s = f"{algo_at['CHROM'].get(t_int, 0):6.1f}" if algo_at['CHROM'].get(t_int) else "    --"
        pos_s = f"{algo_at['POS'].get(t_int, 0):6.1f}" if algo_at['POS'].get(t_int) else "    --"
        cbcr_s = f"{algo_at['CBCR'].get(t_int, 0):6.1f}" if algo_at['CBCR'].get(t_int) else "    --"
        print(f"{t_int:5d}s  {fused_hr:6.1f}  {green_s}  {chrom_s}  {pos_s}  {cbcr_s}")

    print("=" * 80)

    # Statistics per algo
    print("\n--- Algorithm Statistics (valid data only) ---")
    print(f"{'Algorithm':>10s}  {'Mean':>7s}  {'Std':>7s}  {'Min':>7s}  {'Max':>7s}  {'Samples':>7s}")
    print("-" * 55)
    for algo_name in ["FUSED", "GREEN", "CHROM", "POS", "CBCR"]:
        if algo_name == "FUSED":
            vals = [hr for _, hr in results if hr > 0]
        else:
            vals = [hr for _, hr in algo_results.get(algo_name, []) if hr > 0]
        if not vals:
            continue
        a = np.array(vals)
        print(f"{algo_name:>10s}  {np.mean(a):7.1f}  {np.std(a):7.1f}  {np.min(a):7.1f}  {np.max(a):7.1f}  {len(vals):>7d}")

    # Inter-algorithm agreement: how close are the 4 algos at each second
    print("\n--- Inter-Algorithm Agreement (per-second std of 4 algos) ---")
    agree_stds = []
    for t_sec, _ in results:
        t_int = int(round(t_sec))
        vals = []
        for algo_name in ["GREEN", "CHROM", "POS", "CBCR"]:
            v = algo_at[algo_name].get(t_int)
            if v and v > 0:
                vals.append(v)
        if len(vals) >= 3:
            agree_stds.append(np.std(vals))
    if agree_stds:
        a = np.array(agree_stds)
        print(f"  Mean std: {np.mean(a):.1f} BPM   Median: {np.median(a):.1f}   Min: {np.min(a):.1f}   Max: {np.max(a):.1f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Offline rPPG video test")
    parser.add_argument("video", help="path to video file (.mp4)")
    parser.add_argument("csv", nargs="?", default=None, help="path to ground-truth CSV (optional)")
    parser.add_argument("--roi-mode", default="auto", choices=["auto", "mediapipe", "opencv"])
    parser.add_argument("--roi-preset", default="hybrid7",
                        choices=["classic4", "hybrid7", "cheek_forehead6", "whole_face"])
    parser.add_argument("--mp-face-detector-model", default="")
    args = parser.parse_args()

    results, algo_results = run_video(
        args.video,
        roi_mode=args.roi_mode,
        roi_preset=args.roi_preset,
        mp_model=args.mp_face_detector_model,
    )

    if args.csv:
        gt = load_ground_truth(args.csv)
        print(f"Ground truth: {len(gt)} samples, HR range "
              f"{min(h for _,h in gt):.0f}-{max(h for _,h in gt):.0f} BPM")
        compare(results, algo_results, gt)
    else:
        print_analysis(results, algo_results)


if __name__ == "__main__":
    main()
