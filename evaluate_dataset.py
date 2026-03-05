#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from main import (
    Config,
    RPPGAlgorithms,
    FusionEngine,
    SignalQualityController,
    assess_signal_quality,
    create_roi_extractor,
    detect_peaks_and_pnn50,
    extract_rgb_mean,
)


@dataclass
class Sample:
    group: str
    stem: str
    video_path: Path
    ecg_csv_path: Path


def discover_samples(data_dir: Path) -> List[Sample]:
    samples: List[Sample] = []
    for group_dir in sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.isdigit()):
        video_dir = group_dir / "video"
        ecg_dir = group_dir / "csvdata"
        if not video_dir.exists() or not ecg_dir.exists():
            continue
        for video_path in sorted(video_dir.iterdir()):
            if video_path.suffix.lower() not in {".mp4", ".mov"}:
                continue
            stem = video_path.stem
            ecg_path = ecg_dir / f"{stem}.csv"
            if ecg_path.exists():
                samples.append(Sample(group=group_dir.name, stem=stem, video_path=video_path, ecg_csv_path=ecg_path))
    return samples


def load_ecg_hr(csv_path: Path) -> List[float]:
    hrs: List[float] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "HR" not in reader.fieldnames:
            raise RuntimeError(f"ECG CSV missing HR column: {csv_path}")
        for row in reader:
            v = row.get("HR", "").strip()
            if v == "":
                continue
            try:
                hrs.append(float(v))
            except ValueError:
                continue
    return hrs


def process_video(
    sample: Sample,
    cfg: Config,
    roi_mode: str,
    mp_face_detector_model: str = "",
    strict_roi: bool = False,
) -> List[Dict[str, object]]:
    cap = cv2.VideoCapture(str(sample.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {sample.video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    extractor, roi_mode_used = create_roi_extractor(
        roi_mode,
        mp_face_detector_model=mp_face_detector_model,
        strict=strict_roi,
        roi_preset=cfg.roi_preset,
        roi_scale_x=cfg.roi_scale_x,
        roi_scale_y=cfg.roi_scale_y,
        roi_shift_y=cfg.roi_shift_y,
    )
    alg = RPPGAlgorithms(cfg)
    fusion = FusionEngine(cfg)
    quality_ctl = SignalQualityController(publish_interval=1.0)

    roi_buf: Dict[str, Dict[str, Deque[float]]] = {}
    roi_weights: Dict[str, float] = {}
    t_buf: Deque[float] = deque(maxlen=5000)
    max_len = int(cfg.buffer_sec * max(10.0, video_fps))

    predictions: Dict[int, Dict[str, object]] = {}
    last_sec = -1
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
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

            hr_best = 0.0
            hr_pub: Optional[float] = None
            ppi_hr: Optional[float] = None
            pnn50: Optional[float] = None
            sqi = 0.0
            snr_db = 0.0
            freq_conf = 0.0

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

                    qsig = alg.bandpass(alg.robust_norm(alg.detrend_poly2(g_mix)), fs)
                    sqi, snr_db = assess_signal_quality(qsig, fs)

                    _, bvp = alg.pos(r_mix, g_mix, b_mix, fs)
                    ppi_hr, pnn50, _ = detect_peaks_and_pnn50(bvp, fs)

            if tagged:
                hr_raw, freq_conf = fusion.harmonic_temporal_fusion(tagged)
                hr_final = fusion.apply_physiological_constraints(hr_raw)

            if hr_final is not None and cfg.enable_ppi_assist:
                hr_final = fusion.apply_ppi_assist(hr_final, ppi_hr, sqi, freq_conf=freq_conf)
            if hr_final is not None:
                hr_best = fusion.update_and_get_best(hr_final)

            if hr_best > 0:
                combined_quality = max(0.0, min(1.0, 0.7 * sqi + 0.3 * freq_conf))
                if freq_conf < cfg.freq_conf_gate:
                    combined_quality *= 0.75
                conf = max(0.0, min(1.0, 0.45 * sqi + 0.35 * freq_conf + 0.20))
                published, _, _ = quality_ctl.apply(hr_best, conf, combined_quality, float(sec))
                if published is not None:
                    hr_pub = published

            predictions[sec] = {
                "group": sample.group,
                "stem": sample.stem,
                "roi_mode_requested": roi_mode,
                "roi_mode": roi_mode_used,
                "video_path": str(sample.video_path),
                "ecg_csv": str(sample.ecg_csv_path),
                "video_fps": video_fps,
                "fs_est": fs,
                "sec": sec,
                "hr_best": hr_best if hr_best > 0 else None,
                "hr_published": hr_pub,
                "ppi_hr": ppi_hr,
                "pnn50": pnn50,
                "pnn50_reliable": pnn50 is not None,
                "sqi": sqi,
                "frequency_confidence": freq_conf,
                "snr_db": snr_db,
                "state": quality_ctl.state,
            }

        frame_idx += 1

    cap.release()
    return [predictions[k] for k in sorted(predictions.keys())]


def compare_to_ecg(pred_rows: List[Dict[str, object]], ecg_hr: List[float], use_published: bool) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for r in pred_rows:
        sec = int(r["sec"])
        if sec < 0 or sec >= len(ecg_hr):
            continue
        pred = r.get("hr_published") if use_published else r.get("hr_best")
        if pred is None:
            continue

        ref = float(ecg_hr[sec])
        est = float(pred)
        err = est - ref
        abs_err = abs(err)
        ape = abs_err / ref * 100.0 if ref > 1e-6 else None

        merged = dict(r)
        merged.update(
            {
                "ecg_hr": ref,
                "rppg_hr": est,
                "error": err,
                "abs_error": abs_err,
                "ape_percent": ape,
            }
        )
        out.append(merged)
    return out


def summarize(rows: List[Dict[str, object]]) -> Dict[str, float]:
    if not rows:
        return {"n": 0, "mae": math.nan, "rmse": math.nan, "mape": math.nan, "corr": math.nan}
    e = np.array([float(r["error"]) for r in rows], dtype=np.float64)
    ae = np.abs(e)
    ref = np.array([float(r["ecg_hr"]) for r in rows], dtype=np.float64)
    est = np.array([float(r["rppg_hr"]) for r in rows], dtype=np.float64)
    ape_vals = [float(r["ape_percent"]) for r in rows if r["ape_percent"] is not None]

    mae = float(np.mean(ae))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mape = float(np.mean(ape_vals)) if ape_vals else math.nan
    corr = float(np.corrcoef(ref, est)[0, 1]) if len(rows) >= 2 else math.nan
    return {"n": len(rows), "mae": mae, "rmse": rmse, "mape": mape, "corr": corr}


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in columns})


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline rPPG-vs-ECG evaluation for data/001,002,003")
    parser.add_argument("--data-dir", default="/Users/liangwenwang/Downloads/Code/demo1/data")
    parser.add_argument("--out-dir", default="/Users/liangwenwang/Downloads/Code/demo1/rpi_rppg/results")
    parser.add_argument("--roi-mode", choices=["auto", "mediapipe", "opencv", "both"], default="both",
                        help="ROI mode for evaluation. both = run mediapipe and opencv separately")
    parser.add_argument("--roi-preset", choices=["classic4", "hybrid7", "cheek_forehead6", "whole_face"],
                        default="hybrid7", help="ROI geometry preset")
    parser.add_argument("--roi-scale-x", type=float, default=1.1, help="ROI width scale in face-relative coords")
    parser.add_argument("--roi-scale-y", type=float, default=1.1, help="ROI height scale in face-relative coords")
    parser.add_argument("--roi-shift-y", type=float, default=0.0, help="ROI vertical shift in face-relative coords")
    parser.add_argument("--mp-face-detector-model", default="",
                        help="path to MediaPipe tasks face detector model (.tflite/.task) for py3.12 tasks-only builds")
    parser.add_argument("--strict-roi", action="store_true",
                        help="fail fast if ROI backend initialization fails")
    parser.add_argument("--use-published", action="store_true", help="compare ECG against hr_published (state-machine output) instead of hr_best")
    parser.add_argument("--enable-ppi-assist", action="store_true",
                        help="force-enable PPI-assisted HR correction")
    parser.add_argument("--disable-ppi-assist", action="store_true",
                        help="force-disable PPI-assisted HR correction")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    samples = discover_samples(data_dir)
    if not samples:
        raise RuntimeError(f"no paired samples found under {data_dir}")

    roi_modes = ["mediapipe", "opencv"] if args.roi_mode == "both" else [args.roi_mode]
    cfg = Config()
    if args.enable_ppi_assist and args.disable_ppi_assist:
        raise RuntimeError("cannot set both --enable-ppi-assist and --disable-ppi-assist")
    if args.enable_ppi_assist:
        cfg.enable_ppi_assist = True
    if args.disable_ppi_assist:
        cfg.enable_ppi_assist = False
    cfg.roi_preset = args.roi_preset
    cfg.roi_scale_x = args.roi_scale_x
    cfg.roi_scale_y = args.roi_scale_y
    cfg.roi_shift_y = args.roi_shift_y

    for roi_mode in roi_modes:
        all_detail_rows: List[Dict[str, object]] = []
        summary_rows: List[Dict[str, object]] = []
        print(f"\n[MODE] roi_mode={roi_mode}")

        mode_failed = False
        for s in samples:
            print(f"[RUN] group={s.group} stem={s.stem} video={s.video_path.name}")
            try:
                pred_rows = process_video(
                    s,
                    cfg,
                    roi_mode=roi_mode,
                    mp_face_detector_model=args.mp_face_detector_model,
                    strict_roi=args.strict_roi,
                )
            except Exception as e:
                print(f"[SKIP] roi_mode={roi_mode} failed: {type(e).__name__}: {e}")
                mode_failed = True
                break
            ecg = load_ecg_hr(s.ecg_csv_path)
            compared = compare_to_ecg(pred_rows, ecg, use_published=args.use_published)
            stats = summarize(compared)
            used_mode = compared[0]["roi_mode"] if compared else "none"

            for r in compared:
                all_detail_rows.append(r)

            summary_rows.append(
                {
                    "group": s.group,
                    "stem": s.stem,
                    "roi_mode_requested": roi_mode,
                    "roi_mode_used": used_mode,
                    "video_path": str(s.video_path),
                    "ecg_csv": str(s.ecg_csv_path),
                    "n": stats["n"],
                    "mae": stats["mae"],
                    "rmse": stats["rmse"],
                    "mape": stats["mape"],
                    "corr": stats["corr"],
                }
            )
            print(
                f"      n={stats['n']} MAE={stats['mae']:.3f} RMSE={stats['rmse']:.3f} "
                f"MAPE={stats['mape']:.2f}% corr={stats['corr']:.3f}"
            )

        if mode_failed:
            continue

        overall = summarize(all_detail_rows)
        summary_rows.append(
            {
                "group": "ALL",
                "stem": "ALL",
                "roi_mode_requested": roi_mode,
                "roi_mode_used": "mixed",
                "video_path": "-",
                "ecg_csv": "-",
                "n": overall["n"],
                "mae": overall["mae"],
                "rmse": overall["rmse"],
                "mape": overall["mape"],
                "corr": overall["corr"],
            }
        )

        detail_cols = [
            "group",
            "stem",
            "roi_mode_requested",
            "roi_mode",
            "video_path",
            "ecg_csv",
            "video_fps",
            "fs_est",
            "sec",
            "ecg_hr",
            "rppg_hr",
            "hr_best",
            "hr_published",
            "ppi_hr",
            "pnn50",
            "pnn50_reliable",
            "sqi",
            "frequency_confidence",
            "snr_db",
            "state",
            "error",
            "abs_error",
            "ape_percent",
        ]
        summary_cols = [
            "group",
            "stem",
            "roi_mode_requested",
            "roi_mode_used",
            "video_path",
            "ecg_csv",
            "n",
            "mae",
            "rmse",
            "mape",
            "corr",
        ]

        metric_tag = "published" if args.use_published else "best"
        detail_path = out_dir / f"rppg_ecg_comparison_{metric_tag}_{roi_mode}.csv"
        summary_path = out_dir / f"rppg_ecg_summary_{metric_tag}_{roi_mode}.csv"

        write_csv(detail_path, all_detail_rows, detail_cols)
        write_csv(summary_path, summary_rows, summary_cols)

        print(f"\n[OK] detail:  {detail_path}")
        print(f"[OK] summary: {summary_path}")
        print(
            f"[ALL] n={overall['n']} MAE={overall['mae']:.3f} RMSE={overall['rmse']:.3f} "
            f"MAPE={overall['mape']:.2f}% corr={overall['corr']:.3f}"
        )


if __name__ == "__main__":
    main()
