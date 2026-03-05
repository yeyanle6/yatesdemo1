#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from main import (
    Config,
    FusionEngine,
    RPPGAlgorithms,
    SignalQualityController,
    assess_signal_quality,
    create_roi_extractor,
    detect_peaks_and_pnn50,
    extract_rgb_mean,
)


def load_ecg_hr(csv_path: Path) -> List[float]:
    hrs: List[float] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "HR" not in reader.fieldnames:
            return hrs
        for row in reader:
            v = row.get("HR", "").strip()
            if not v:
                continue
            try:
                hrs.append(float(v))
            except ValueError:
                continue
    return hrs


def _draw_polyline(
    canvas: np.ndarray,
    values: List[float],
    x0: int,
    y0: int,
    w: int,
    h: int,
    color: Tuple[int, int, int],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    if len(values) < 2 or w <= 2 or h <= 2:
        return
    arr = np.array(values, dtype=np.float32)
    if vmin is None:
        vmin = float(np.min(arr))
    if vmax is None:
        vmax = float(np.max(arr))
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6

    n = len(arr)
    pts: List[Tuple[int, int]] = []
    for i, v in enumerate(arr):
        xn = i / max(1, n - 1)
        yn = (float(v) - vmin) / (vmax - vmin)
        x = x0 + int(xn * (w - 1))
        y = y0 + h - 1 - int(yn * (h - 1))
        pts.append((x, y))
    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)


def _draw_panel(
    frame: np.ndarray,
    g_hist: List[float],
    hr_hist: List[float],
    ecg_hr: Optional[float],
    metrics: Dict[str, float | str | None],
) -> np.ndarray:
    h, w = frame.shape[:2]
    panel_w = min(460, int(w * 0.45))
    panel = np.full((h, panel_w, 3), 22, dtype=np.uint8)

    # Header
    cv2.putText(panel, "rPPG Pipeline Visualization", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 2, cv2.LINE_AA)

    # Metrics text
    hr = metrics.get("hr", 0.0) or 0.0
    ppi_hr = metrics.get("ppi_hr", None)
    sqi = metrics.get("sqi", 0.0) or 0.0
    fconf = metrics.get("fconf", 0.0) or 0.0
    snr = metrics.get("snr_db", 0.0) or 0.0
    state = str(metrics.get("state", "BAD"))
    pnn50 = metrics.get("pnn50", None)
    lines = [
        f"HR (rPPG): {hr:6.1f} bpm",
        f"PPI_HR   : {ppi_hr:6.1f} bpm" if ppi_hr is not None else "PPI_HR   : NA",
        f"ECG_HR   : {ecg_hr:6.1f} bpm" if ecg_hr is not None else "ECG_HR   : NA",
        f"SQI/FCONF: {sqi:0.3f} / {fconf:0.3f}",
        f"SNR(dB)  : {snr:0.1f}",
        f"State    : {state}",
        f"pNN50(exp): {pnn50:0.3f}" if pnn50 is not None else "pNN50(exp): NA",
    ]
    for i, t in enumerate(lines):
        cv2.putText(panel, t, (12, 54 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (210, 210, 210), 1, cv2.LINE_AA)

    # Signal chart
    y_base = 220
    chart_h = 130
    cv2.rectangle(panel, (10, y_base), (panel_w - 10, y_base + chart_h), (70, 70, 70), 1)
    cv2.putText(panel, "Green Signal (normalized)", (14, y_base - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    if g_hist:
        g = np.array(g_hist[-180:], dtype=np.float32)
        g = (g - np.mean(g)) / (np.std(g) + 1e-6)
        _draw_polyline(panel, g.tolist(), 12, y_base + 2, panel_w - 24, chart_h - 4, (60, 200, 255), -3.0, 3.0)

    # HR trend chart
    y2 = y_base + chart_h + 34
    chart2_h = 130
    cv2.rectangle(panel, (10, y2), (panel_w - 10, y2 + chart2_h), (70, 70, 70), 1)
    cv2.putText(panel, "HR Trend (rPPG=orange, ECG=green point)", (14, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (180, 180, 180), 1, cv2.LINE_AA)
    if hr_hist:
        hh = hr_hist[-180:]
        _draw_polyline(panel, hh, 12, y2 + 2, panel_w - 24, chart2_h - 4, (0, 165, 255), 40.0, 170.0)
        if ecg_hr is not None:
            x = panel_w - 20
            y = y2 + chart2_h - 1 - int((ecg_hr - 40.0) / 130.0 * (chart2_h - 1))
            y = max(y2 + 2, min(y2 + chart2_h - 2, y))
            cv2.circle(panel, (x, y), 4, (80, 220, 120), -1, cv2.LINE_AA)

    return np.concatenate([frame, panel], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export annotated rPPG processing visualization video")
    parser.add_argument("--input-video", required=True, help="path to input video")
    parser.add_argument("--output-video", required=True, help="path to output annotated mp4")
    parser.add_argument("--ecg-csv", default="", help="optional ECG csv (must contain HR column)")
    parser.add_argument("--roi-mode", choices=["auto", "mediapipe", "opencv"], default="opencv")
    parser.add_argument("--roi-preset", choices=["classic4", "hybrid7", "cheek_forehead6", "whole_face"], default="hybrid7")
    parser.add_argument("--roi-scale-x", type=float, default=1.1)
    parser.add_argument("--roi-scale-y", type=float, default=1.1)
    parser.add_argument("--roi-shift-y", type=float, default=0.0)
    parser.add_argument("--mp-face-detector-model", default="")
    parser.add_argument("--strict-roi", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="0 means full video")
    parser.add_argument("--write-fps", type=float, default=0.0, help="0 means input fps")
    args = parser.parse_args()

    in_path = Path(args.input_video)
    out_path = Path(args.output_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open input video: {in_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = frame_count
    if args.max_seconds > 0:
        max_frames = min(frame_count, int(args.max_seconds * video_fps))

    cfg = Config()
    cfg.roi_preset = args.roi_preset
    cfg.roi_scale_x = args.roi_scale_x
    cfg.roi_scale_y = args.roi_scale_y
    cfg.roi_shift_y = args.roi_shift_y

    extractor, roi_mode_used = create_roi_extractor(
        args.roi_mode,
        mp_face_detector_model=args.mp_face_detector_model,
        strict=args.strict_roi,
        roi_preset=cfg.roi_preset,
        roi_scale_x=cfg.roi_scale_x,
        roi_scale_y=cfg.roi_scale_y,
        roi_shift_y=cfg.roi_shift_y,
    )
    print(f"[VIS] roi_mode_used={roi_mode_used}, roi_preset={cfg.roi_preset}")

    alg = RPPGAlgorithms(cfg)
    fusion = FusionEngine(cfg)
    quality_ctl = SignalQualityController(publish_interval=1.0)

    ecg = load_ecg_hr(Path(args.ecg_csv)) if args.ecg_csv else []
    has_ecg = len(ecg) > 0

    roi_buf: Dict[str, Dict[str, Deque[float]]] = {}
    roi_weights: Dict[str, float] = {}
    t_buf: Deque[float] = deque(maxlen=5000)
    max_len = int(cfg.buffer_sec * max(10.0, video_fps))

    g_mixed_hist: Deque[float] = deque(maxlen=600)
    hr_hist: Deque[float] = deque(maxlen=600)

    hr_best = 0.0
    ppi_hr: Optional[float] = None
    pnn50: Optional[float] = None
    sqi = 0.0
    snr_db = 0.0
    freq_conf = 0.0
    last_sec = -1

    writer: Optional[cv2.VideoWriter] = None
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if max_frames > 0 and frame_idx >= max_frames:
                break

            t = frame_idx / max(video_fps, 1e-6)
            t_buf.append(t)

            rois = extractor.extract(frame)
            for roi in rois:
                rgb = extract_rgb_mean(frame, roi)
                if rgb is None:
                    continue
                if roi.name not in roi_buf:
                    roi_buf[roi.name] = {
                        "r": deque(maxlen=max_len),
                        "g": deque(maxlen=max_len),
                        "b": deque(maxlen=max_len),
                    }
                roi_buf[roi.name]["r"].append(rgb[0])
                roi_buf[roi.name]["g"].append(rgb[1])
                roi_buf[roi.name]["b"].append(rgb[2])
                roi_weights[roi.name] = roi.weight

            if len(t_buf) > 40:
                dt = np.diff(np.array(t_buf, dtype=np.float64)[-120:])
                fs = float(1.0 / max(1e-6, np.median(dt)))
            else:
                fs = video_fps

            sec = int(t)
            if sec != last_sec:
                last_sec = sec
                tagged: List[Tuple[float, str, str]] = []
                merged_rgb: Dict[str, List[np.ndarray]] = {"r": [], "g": [], "b": []}
                merged_w: List[float] = []

                min_len = int(max(cfg.welch_seg_sec * fs, cfg.pos_window_sec * fs) + 5)
                for name, buf in roi_buf.items():
                    if len(buf["g"]) < min_len:
                        continue
                    r = np.array(buf["r"], dtype=np.float64)
                    g = np.array(buf["g"], dtype=np.float64)
                    b = np.array(buf["b"], dtype=np.float64)

                    hr_g, _ = alg.green(r, g, b, fs)
                    hr_c, _ = alg.chrom(r, g, b, fs)
                    hr_p, _ = alg.pos(r, g, b, fs)

                    if cfg.min_bpm_valid <= hr_g <= cfg.max_bpm_valid:
                        tagged.append((hr_g, "GREEN", name))
                    if cfg.min_bpm_valid <= hr_c <= cfg.max_bpm_valid:
                        tagged.append((hr_c, "CHROM", name))
                    if cfg.min_bpm_valid <= hr_p <= cfg.max_bpm_valid:
                        tagged.append((hr_p, "POS", name))

                    merged_rgb["r"].append(r)
                    merged_rgb["g"].append(g)
                    merged_rgb["b"].append(b)
                    merged_w.append(roi_weights.get(name, 1.0))

                hr_final: Optional[float] = None
                if merged_w:
                    min_common = min(len(x) for x in merged_rgb["r"])
                    if min_common >= min_len:
                        w = np.array(merged_w, dtype=np.float64)
                        w = w / (np.sum(w) + 1e-9)
                        rr = np.vstack([x[-min_common:] for x in merged_rgb["r"]])
                        gg = np.vstack([x[-min_common:] for x in merged_rgb["g"]])
                        bb = np.vstack([x[-min_common:] for x in merged_rgb["b"]])
                        r_mix = np.sum(rr * w[:, None], axis=0)
                        g_mix = np.sum(gg * w[:, None], axis=0)
                        b_mix = np.sum(bb * w[:, None], axis=0)
                        g_mixed_hist.append(float(g_mix[-1]))

                        qsig = alg.bandpass(alg.robust_norm(alg.detrend_poly2(g_mix)), fs)
                        sqi, snr_db = assess_signal_quality(qsig, fs)

                        _, bvp = alg.pos(r_mix, g_mix, b_mix, fs)
                        ppi_hr, pnn50, _ = detect_peaks_and_pnn50(bvp, fs)

                if tagged:
                    hr_raw, freq_conf = fusion.harmonic_temporal_fusion(tagged)
                    hr_final = fusion.apply_physiological_constraints(hr_raw)
                    if cfg.enable_ppi_assist:
                        hr_final = fusion.apply_ppi_assist(hr_final, ppi_hr, sqi, freq_conf=freq_conf)
                    hr_best = fusion.update_and_get_best(hr_final)
                    hr_hist.append(hr_best)

                if hr_best > 0:
                    combined_quality = max(0.0, min(1.0, 0.7 * sqi + 0.3 * freq_conf))
                    if freq_conf < cfg.freq_conf_gate:
                        combined_quality *= 0.75
                    conf = max(0.0, min(1.0, 0.45 * sqi + 0.35 * freq_conf + 0.20))
                    quality_ctl.apply(hr_best, conf, combined_quality, float(sec))

            # Draw ROI rectangles on original frame.
            for roi in rois:
                h, w = frame.shape[:2]
                x, y, rw, rh = roi.rect
                x0, y0 = int(x * w), int(y * h)
                x1, y1 = int((x + rw) * w), int((y + rh) * h)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 220, 80), 1)
                cv2.putText(
                    frame,
                    f"{roi.name}:{roi.weight:.2f}",
                    (x0, max(10, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (0, 220, 80),
                    1,
                    cv2.LINE_AA,
                )

            ecg_hr = ecg[last_sec] if has_ecg and 0 <= last_sec < len(ecg) else None
            metrics = {
                "hr": hr_best,
                "ppi_hr": ppi_hr,
                "pnn50": pnn50,
                "sqi": sqi,
                "fconf": freq_conf,
                "snr_db": snr_db,
                "state": quality_ctl.state,
            }
            out_frame = _draw_panel(frame, list(g_mixed_hist), list(hr_hist), ecg_hr, metrics)

            if writer is None:
                out_h, out_w = out_frame.shape[:2]
                write_fps = args.write_fps if args.write_fps > 0 else video_fps
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, write_fps, (out_w, out_h))
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open video writer: {out_path}")

            writer.write(out_frame)
            frame_idx += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    print(f"[VIS] done: {out_path}")


if __name__ == "__main__":
    main()
