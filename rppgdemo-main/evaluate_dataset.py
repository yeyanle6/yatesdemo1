#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

import cv2
import numpy as np
from scipy.signal import welch

from main import (
    AdaptiveROIController,
    Config,
    FusionEngine,
    RPPGAlgorithms,
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


@dataclass
class ECGRow:
    sec: int
    hr: Optional[float]
    rri_ms: Optional[float]
    hf: Optional[float]
    lfhf: Optional[float]
    lfratio: Optional[float]
    source: str
    hr_interpolated: bool = False
    hf_interpolated: bool = False
    lfhf_interpolated: bool = False
    lfratio_interpolated: bool = False


def sample_token(group: str, stem: str) -> str:
    return f"{group}/{stem}"


def discover_samples(data_dir: Path) -> List[Sample]:
    samples: List[Sample] = []
    seen: Set[Tuple[str, str, str]] = set()

    candidate_groups: List[Path] = []
    candidate_groups.extend(sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.isdigit()))

    for maybe_videos in sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.lower() == "videos"):
        candidate_groups.extend(sorted(p for p in maybe_videos.iterdir() if p.is_dir() and p.name.isdigit()))

    for group_dir in candidate_groups:
        ecg_dir = group_dir / "csvdata"
        if not ecg_dir.exists():
            continue

        # 兼容两种数据结构:
        # A) <group>/video/*.mp4
        # B) <group>/*.mp4  (如 Videos/1/1-1.mp4)
        video_roots: List[Path] = []
        nested_video_dir = group_dir / "video"
        if nested_video_dir.exists():
            video_roots.append(nested_video_dir)
        video_roots.append(group_dir)

        for root in video_roots:
            for video_path in sorted(root.iterdir()):
                if not video_path.is_file():
                    continue
                if video_path.suffix.lower() not in {".mp4", ".mov"}:
                    continue
                stem = video_path.stem
                ecg_path = ecg_dir / f"{stem}.csv"
                if not ecg_path.exists():
                    continue
                key = (group_dir.name, stem, str(video_path))
                if key in seen:
                    continue
                seen.add(key)
                samples.append(
                    Sample(
                        group=group_dir.name,
                        stem=stem,
                        video_path=video_path,
                        ecg_csv_path=ecg_path,
                    )
                )

    return sorted(samples, key=lambda s: (s.group, s.stem, str(s.video_path)))


def _to_float(v: str) -> Optional[float]:
    txt = v.strip()
    if txt == "":
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def _clock_to_seconds(value: str) -> Optional[int]:
    txt = value.strip()
    if txt == "":
        return None
    parts = txt.split(":")
    if len(parts) != 3:
        return None
    try:
        hh = int(parts[0])
        mm = int(parts[1])
        ss = int(parts[2])
    except ValueError:
        return None
    if hh < 0 or mm < 0 or mm >= 60 or ss < 0 or ss >= 60:
        return None
    return hh * 3600 + mm * 60 + ss


def _metric_interp(a: Optional[float], b: Optional[float], ratio: float) -> Tuple[Optional[float], bool]:
    if a is None or b is None:
        return None, False
    return a + (b - a) * ratio, True


def load_ecg_series(csv_path: Path, align_mode: str = "timestamp") -> Dict[int, ECGRow]:
    raw_rows: List[Tuple[int, Optional[int], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"invalid ECG CSV (no header): {csv_path}")
        if "HR" not in reader.fieldnames:
            raise RuntimeError(f"ECG CSV missing HR column: {csv_path}")

        for idx, row in enumerate(reader):
            hr = _to_float(row.get("HR", ""))
            rri = _to_float(row.get("RRI", ""))
            hf = _to_float(row.get("HF", ""))
            lfhf = _to_float(row.get("LF/HF", ""))
            lfratio = _to_float(row.get("LF ratio", ""))
            sec_abs = _clock_to_seconds(row.get("time", ""))
            if hr is None and rri is None and hf is None and lfhf is None and lfratio is None:
                continue
            raw_rows.append((idx, sec_abs, hr, rri, hf, lfhf, lfratio))

    if not raw_rows:
        return {}

    points: List[ECGRow] = []
    if align_mode == "timestamp":
        base_abs = None
        rollover = 0
        prev_abs = None
        for _, sec_abs, hr, rri, hf, lfhf, lfratio in raw_rows:
            if sec_abs is None:
                continue
            if base_abs is None:
                base_abs = sec_abs
                prev_abs = sec_abs
            else:
                assert prev_abs is not None
                if sec_abs < prev_abs:
                    rollover += 24 * 3600
                prev_abs = sec_abs
            rel = (sec_abs + rollover) - base_abs
            points.append(
                ECGRow(
                    sec=int(rel),
                    hr=hr,
                    rri_ms=rri,
                    hf=hf,
                    lfhf=lfhf,
                    lfratio=lfratio,
                    source="observed",
                )
            )

        # Fallback if timestamp parsing fails for most rows.
        if len(points) < max(3, len(raw_rows) // 2):
            points = [
                ECGRow(
                    sec=i,
                    hr=hr,
                    rri_ms=rri,
                    hf=hf,
                    lfhf=lfhf,
                    lfratio=lfratio,
                    source="observed",
                )
                for i, (_, _, hr, rri, hf, lfhf, lfratio) in enumerate(raw_rows)
            ]
    else:
        points = [
            ECGRow(
                sec=i,
                hr=hr,
                rri_ms=rri,
                hf=hf,
                lfhf=lfhf,
                lfratio=lfratio,
                source="observed",
            )
            for i, (_, _, hr, rri, hf, lfhf, lfratio) in enumerate(raw_rows)
        ]

    if not points:
        return {}

    by_sec: Dict[int, ECGRow] = {}
    for row in points:
        if row.sec not in by_sec:
            by_sec[row.sec] = row

    secs = sorted(by_sec.keys())
    if not secs:
        return {}

    filled: Dict[int, ECGRow] = {}
    for sec in range(secs[0], secs[-1] + 1):
        if sec in by_sec:
            filled[sec] = by_sec[sec]
            continue

        left_candidates = [s for s in secs if s < sec]
        right_candidates = [s for s in secs if s > sec]
        if not left_candidates or not right_candidates:
            continue

        lsec = left_candidates[-1]
        rsec = right_candidates[0]
        lrow = by_sec[lsec]
        rrow = by_sec[rsec]
        span = max(1, rsec - lsec)
        ratio = float(sec - lsec) / float(span)

        hr, hr_i = _metric_interp(lrow.hr, rrow.hr, ratio)
        hf, hf_i = _metric_interp(lrow.hf, rrow.hf, ratio)
        lfhf, lfhf_i = _metric_interp(lrow.lfhf, rrow.lfhf, ratio)
        lfratio, lfratio_i = _metric_interp(lrow.lfratio, rrow.lfratio, ratio)
        rri, _ = _metric_interp(lrow.rri_ms, rrow.rri_ms, ratio)

        filled[sec] = ECGRow(
            sec=sec,
            hr=hr,
            rri_ms=rri,
            hf=hf,
            lfhf=lfhf,
            lfratio=lfratio,
            source="interpolated",
            hr_interpolated=hr_i,
            hf_interpolated=hf_i,
            lfhf_interpolated=lfhf_i,
            lfratio_interpolated=lfratio_i,
        )

    for sec in secs:
        filled[sec] = by_sec[sec]

    return dict(sorted(filled.items()))


def load_ecg_hr(csv_path: Path) -> List[float]:
    # Backward-compatible helper used by existing scripts.
    series = load_ecg_series(csv_path, align_mode="index")
    out: List[float] = []
    for sec in sorted(series.keys()):
        v = series[sec].hr
        if v is not None:
            out.append(float(v))
    return out


def parse_sample_token(token: str) -> Optional[Tuple[str, str]]:
    text = token.strip()
    if text == "":
        return None
    text = text.replace("\\", "/")
    if "/" in text:
        parts = [p for p in text.split("/") if p]
        if len(parts) >= 2:
            return parts[-2], parts[-1]
    m = re.match(r"^(\d{1,3})[-_](.+)$", text)
    if m:
        return m.group(1).zfill(3), m.group(2)
    return None


def build_split_map(samples: List[Sample], split_file: str = "", holdout_list: str = "") -> Dict[str, str]:
    split_map: Dict[str, str] = {sample_token(s.group, s.stem): "train" for s in samples}

    def _set_split(g: str, stem: str, split_name: str) -> None:
        key = sample_token(g, stem)
        if key in split_map:
            split_map[key] = split_name

    if split_file:
        path = Path(split_file)
        if not path.exists():
            raise RuntimeError(f"split file not found: {path}")

        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for split_name in ("train", "test"):
                    for token in payload.get(split_name, []):
                        parsed = parse_sample_token(str(token))
                        if parsed is None:
                            continue
                        _set_split(parsed[0], parsed[1], split_name)
        else:
            text = path.read_text(encoding="utf-8").splitlines()
            for raw in text:
                line = raw.strip()
                if line == "" or line.startswith("#"):
                    continue
                cols = [c.strip() for c in re.split(r"[\s,]+", line) if c.strip()]
                if len(cols) < 2:
                    continue

                split_name = ""
                sample_part = ""
                if cols[0].lower() in {"train", "test"}:
                    split_name = cols[0].lower()
                    if len(cols) >= 3:
                        sample_part = f"{cols[1]}/{cols[2]}"
                    else:
                        sample_part = cols[1]
                elif cols[-1].lower() in {"train", "test"}:
                    split_name = cols[-1].lower()
                    if len(cols) >= 3:
                        sample_part = f"{cols[0]}/{cols[1]}"
                    else:
                        sample_part = cols[0]
                else:
                    continue

                parsed = parse_sample_token(sample_part)
                if parsed is None:
                    continue
                _set_split(parsed[0], parsed[1], split_name)

    if holdout_list:
        for token in [x.strip() for x in holdout_list.split(",") if x.strip()]:
            parsed = parse_sample_token(token)
            if parsed is None:
                continue
            _set_split(parsed[0], parsed[1], "test")

    return split_map


def parse_metrics_arg(text: str) -> Set[str]:
    tokens = {x.strip().lower() for x in text.split(",") if x.strip()}
    out: Set[str] = set()
    if "hr" in tokens:
        out.add("hr")
    if "lf" in tokens:
        out.update({"hf", "lfhf", "lfratio"})
    if "hf" in tokens:
        out.add("hf")
    if "lfhf" in tokens or "lf/hf" in tokens:
        out.add("lfhf")
    if "lfratio" in tokens or "lf_ratio" in tokens:
        out.add("lfratio")
    if not out:
        out = {"hr", "hf", "lfhf", "lfratio"}
    return out


def apply_config_profile(cfg: Config, profile: str) -> None:
    if profile == "python_latest":
        return
    if profile == "ios_like_v413":
        # Approximate iOS v4.13 runtime behavior for offline comparison.
        cfg.low_hz = 0.65
        cfg.high_hz = 4.0
        cfg.filter_order = 6
        cfg.welch_seg_sec = 2.0
        cfg.welch_overlap = 0.80
        cfg.nfft = 2048
        cfg.buffer_sec = 10.0
        cfg.min_bpm_valid = 45.0
        cfg.max_bpm_valid = 170.0
        cfg.clamp_min_bpm = 40.0
        cfg.clamp_max_bpm = 120.0
        cfg.high_hr_cluster_boost = 0.0
        cfg.high_hr_bias_gain = 1.0
        cfg.high_hr_bias_offset = 0.0
        cfg.enable_ppi_assist = False
        cfg.use_cbcr_candidate = False
        cfg.output_drop_guard_enabled = False
        return
    raise RuntimeError(f"unsupported config profile: {profile}")


def compute_lf_metrics_from_bvp(
    bvp: np.ndarray,
    fs: float,
    lf_window_sec: float,
    lf_resample_fs: float,
    cfg: Config,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(bvp) < int(max(10.0, lf_window_sec * 0.6) * fs):
        return None, None, None

    window_n = min(len(bvp), int(max(10.0, lf_window_sec) * fs))
    segment = bvp[-window_n:]
    _, _, peaks = detect_peaks_and_pnn50(
        segment,
        fs,
        hr_min_sec=3.0,
        pnn50_min_sec=10.0,
        peak_threshold_k=cfg.peak_threshold_k,
        peak_window_divisor=cfg.peak_window_divisor,
        peak_missing_threshold_scale=cfg.peak_missing_threshold_scale,
        peak_min_distance_hz=cfg.peak_min_distance_hz,
        peak_max_distance_hz=cfg.peak_max_distance_hz,
    )
    if len(peaks) < 12:
        return None, None, None

    peak_times = np.array(peaks, dtype=np.float64) / max(1e-6, fs)
    ibi_ms = np.diff(peak_times) * 1000.0
    if len(ibi_ms) < 10:
        return None, None, None

    ibi_times = peak_times[1:]
    duration = float(ibi_times[-1] - ibi_times[0]) if len(ibi_times) >= 2 else 0.0
    if duration < min(30.0, max(18.0, lf_window_sec * 0.65)):
        return None, None, None

    step = 1.0 / max(1e-6, lf_resample_fs)
    t_uniform = np.arange(ibi_times[0], ibi_times[-1], step)
    if len(t_uniform) < 32:
        return None, None, None

    ibi_uniform = np.interp(t_uniform, ibi_times, ibi_ms)
    ibi_uniform = ibi_uniform - np.mean(ibi_uniform)

    nperseg = min(256, len(ibi_uniform))
    if nperseg < 8:
        return None, None, None
    freqs, psd = welch(
        ibi_uniform,
        fs=max(1e-6, lf_resample_fs),
        nperseg=nperseg,
        noverlap=nperseg // 2,
        nfft=1024,
        window="hann",
    )

    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)

    # Use trapz for compatibility with NumPy 1.x in locked desktop env.
    lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if np.any(lf_mask) else 0.0
    hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else 0.0
    lfhf = (lf_power / hf_power) if hf_power > 1e-9 else None
    lfratio = (100.0 * lf_power / (lf_power + hf_power)) if (lf_power + hf_power) > 1e-9 else None

    return hf_power, lfhf, lfratio


def process_video(
    sample: Sample,
    cfg: Config,
    roi_mode: str,
    mp_face_detector_model: str = "",
    strict_roi: bool = False,
    include_cbcr: bool = True,
    lf_window_sec: float = 30.0,
    lf_resample_fs: float = 4.0,
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
    adaptive_roi_ctl = AdaptiveROIController(cfg, extractor)
    quality_ctl = SignalQualityController(publish_interval=1.0)

    roi_buf: Dict[str, Dict[str, Deque[float]]] = {}
    roi_weights: Dict[str, float] = {}
    t_buf: Deque[float] = deque(maxlen=5000)
    # Offline LF metrics need longer history than realtime HR tracking.
    required_buffer_sec = max(cfg.buffer_sec, lf_window_sec + 5.0)
    max_len = int(required_buffer_sec * max(10.0, video_fps))

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
                hr_cb = 0.0
                if include_cbcr and cfg.use_cbcr_candidate:
                    hr_cb, _ = alg.cbcr_pos(r, g, b, fs)

                if cfg.min_bpm_valid <= hr_g <= cfg.max_bpm_valid:
                    tagged.append((hr_g, "GREEN", name))
                if cfg.min_bpm_valid <= hr_c <= cfg.max_bpm_valid:
                    tagged.append((hr_c, "CHROM", name))
                if cfg.min_bpm_valid <= hr_p <= cfg.max_bpm_valid:
                    tagged.append((hr_p, "POS", name))
                if include_cbcr and cfg.use_cbcr_candidate and cfg.min_bpm_valid <= hr_cb <= cfg.max_bpm_valid:
                    tagged.append((hr_cb, "CBCR", name))

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
            hf_est: Optional[float] = None
            lfhf_est: Optional[float] = None
            lfratio_est: Optional[float] = None
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
                    sqi, snr_db = assess_signal_quality(
                        qsig,
                        fs,
                        snr_weight=cfg.quality_snr_weight,
                        stability_weight=cfg.quality_stability_weight,
                        corr_weight=cfg.quality_corr_weight,
                        periodicity_weight=cfg.quality_periodicity_weight,
                        snr_norm_div=cfg.quality_snr_norm_div,
                        signal_strength_min=cfg.quality_signal_strength_min,
                    )

                    _, bvp = alg.pos(r_mix, g_mix, b_mix, fs)
                    ppi_hr, pnn50, _ = detect_peaks_and_pnn50(
                        bvp,
                        fs,
                        peak_threshold_k=cfg.peak_threshold_k,
                        peak_window_divisor=cfg.peak_window_divisor,
                        peak_missing_threshold_scale=cfg.peak_missing_threshold_scale,
                        peak_min_distance_hz=cfg.peak_min_distance_hz,
                        peak_max_distance_hz=cfg.peak_max_distance_hz,
                    )

                    hf_est, lfhf_est, lfratio_est = compute_lf_metrics_from_bvp(
                        bvp,
                        fs,
                        lf_window_sec=lf_window_sec,
                        lf_resample_fs=lf_resample_fs,
                        cfg=cfg,
                    )

            if tagged:
                hr_raw, freq_conf = fusion.harmonic_temporal_fusion(tagged)
                adapt_msg = adaptive_roi_ctl.update(
                    now_sec=float(sec),
                    tagged=tagged,
                    hr_raw=hr_raw,
                    ppi_hr=ppi_hr,
                    freq_conf=freq_conf,
                )
                if adapt_msg and cfg.adaptive_roi_debug:
                    print(
                        f"[ROI-ADAPT] sample={sample.group}/{sample.stem} "
                        f"sec={sec} {adapt_msg}"
                    )
                hr_final = fusion.apply_physiological_constraints(hr_raw)

            if hr_final is not None and cfg.enable_ppi_assist:
                hr_final = fusion.apply_ppi_assist(hr_final, ppi_hr, sqi, freq_conf=freq_conf)
            if hr_final is not None:
                hr_best = fusion.update_and_get_best(hr_final)

            if hr_best > 0:
                combined_quality = max(
                    0.0,
                    min(
                        1.0,
                        cfg.publish_quality_sqi_weight * sqi +
                        cfg.publish_quality_freq_weight * freq_conf,
                    ),
                )
                if freq_conf < cfg.freq_conf_gate:
                    combined_quality *= cfg.publish_quality_low_freq_penalty
                conf = max(
                    0.0,
                    min(
                        1.0,
                        cfg.publish_conf_sqi_weight * sqi +
                        cfg.publish_conf_freq_weight * freq_conf +
                        cfg.publish_conf_bias,
                    ),
                )
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
                "hf_est": hf_est,
                "lfhf_est": lfhf_est,
                "lfratio_est": lfratio_est,
            }

        frame_idx += 1

    cap.release()
    return [predictions[k] for k in sorted(predictions.keys())]


def _compare_metric(
    merged: Dict[str, object],
    metric_name: str,
    ref_value: Optional[float],
    est_value: Optional[float],
) -> bool:
    merged[f"ecg_{metric_name}"] = ref_value
    merged[f"est_{metric_name}"] = est_value
    merged[f"error_{metric_name}"] = None
    merged[f"abs_error_{metric_name}"] = None
    merged[f"ape_percent_{metric_name}"] = None
    if ref_value is None or est_value is None:
        return False
    err = float(est_value - ref_value)
    ae = abs(err)
    ape = (ae / abs(ref_value) * 100.0) if abs(ref_value) > 1e-9 else None
    merged[f"error_{metric_name}"] = err
    merged[f"abs_error_{metric_name}"] = ae
    merged[f"ape_percent_{metric_name}"] = ape
    return True


def compare_to_ecg(
    pred_rows: List[Dict[str, object]],
    ecg: Dict[int, ECGRow] | List[float],
    use_published: bool,
    metrics: Optional[Set[str]] = None,
) -> List[Dict[str, object]]:
    enabled = set(metrics or {"hr", "hf", "lfhf", "lfratio"})

    ecg_map: Dict[int, ECGRow] = {}
    if isinstance(ecg, list):
        for sec, val in enumerate(ecg):
            ecg_map[sec] = ECGRow(sec=sec, hr=float(val), rri_ms=None, hf=None, lfhf=None, lfratio=None, source="observed")
    else:
        ecg_map = ecg

    out: List[Dict[str, object]] = []
    for r in pred_rows:
        sec = int(r["sec"])
        ecg_row = ecg_map.get(sec)
        if ecg_row is None:
            continue

        merged = dict(r)
        merged["ecg_sec"] = ecg_row.sec
        merged["ecg_source"] = ecg_row.source
        merged["ecg_hr_interpolated"] = ecg_row.hr_interpolated
        merged["ecg_hf_interpolated"] = ecg_row.hf_interpolated
        merged["ecg_lfhf_interpolated"] = ecg_row.lfhf_interpolated
        merged["ecg_lfratio_interpolated"] = ecg_row.lfratio_interpolated

        any_compared = False

        if "hr" in enabled:
            pred = r.get("hr_published") if use_published else r.get("hr_best")
            hr_compared = _compare_metric(merged, "hr", ecg_row.hr, None if pred is None else float(pred))
            if hr_compared:
                merged["ecg_hr"] = merged["ecg_hr"]
                merged["rppg_hr"] = merged["est_hr"]
                merged["error"] = merged["error_hr"]
                merged["abs_error"] = merged["abs_error_hr"]
                merged["ape_percent"] = merged["ape_percent_hr"]
                any_compared = True
            else:
                merged["ecg_hr"] = merged.get("ecg_hr")
                merged["rppg_hr"] = merged.get("est_hr")
                merged["error"] = merged.get("error_hr")
                merged["abs_error"] = merged.get("abs_error_hr")
                merged["ape_percent"] = merged.get("ape_percent_hr")

        if "hf" in enabled:
            any_compared = _compare_metric(merged, "hf", ecg_row.hf, None if r.get("hf_est") is None else float(r["hf_est"])) or any_compared
        if "lfhf" in enabled:
            any_compared = _compare_metric(merged, "lfhf", ecg_row.lfhf, None if r.get("lfhf_est") is None else float(r["lfhf_est"])) or any_compared
        if "lfratio" in enabled:
            any_compared = _compare_metric(merged, "lfratio", ecg_row.lfratio, None if r.get("lfratio_est") is None else float(r["lfratio_est"])) or any_compared

        if any_compared:
            out.append(merged)

    return out


def summarize_metric(rows: List[Dict[str, object]], ref_key: str, est_key: str) -> Dict[str, float]:
    vals = [r for r in rows if r.get(ref_key) is not None and r.get(est_key) is not None]
    if not vals:
        return {
            "n": 0.0,
            "mae": math.nan,
            "rmse": math.nan,
            "mape": math.nan,
            "corr": math.nan,
            "bias": math.nan,
        }

    ref = np.array([float(r[ref_key]) for r in vals], dtype=np.float64)
    est = np.array([float(r[est_key]) for r in vals], dtype=np.float64)
    err = est - ref
    abs_err = np.abs(err)
    ape: List[float] = []
    for rv, ae in zip(ref, abs_err):
        if abs(rv) > 1e-9:
            ape.append(float(ae / abs(rv) * 100.0))

    corr = float(np.corrcoef(ref, est)[0, 1]) if len(vals) >= 2 else math.nan
    return {
        "n": float(len(vals)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(ape)) if ape else math.nan,
        "corr": corr,
        "bias": float(np.mean(err)),
    }


def summarize(rows: List[Dict[str, object]]) -> Dict[str, float]:
    # Backward-compatible HR summary.
    s = summarize_metric(rows, "ecg_hr", "rppg_hr")
    return {
        "n": int(s["n"]),
        "mae": s["mae"],
        "rmse": s["rmse"],
        "mape": s["mape"],
        "corr": s["corr"],
    }


def _safe_mean(xs: Iterable[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    if not vals:
        return math.nan
    return float(np.mean(np.array(vals, dtype=np.float64)))


def summarize_all_metrics(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    hr = summarize_metric(rows, "ecg_hr", "est_hr")
    hf = summarize_metric(rows, "ecg_hf", "est_hf")
    lfhf = summarize_metric(rows, "ecg_lfhf", "est_lfhf")
    lfratio = summarize_metric(rows, "ecg_lfratio", "est_lfratio")

    lf_mape = _safe_mean([hf["mape"], lfhf["mape"], lfratio["mape"]])
    lf_corr = _safe_mean([hf["corr"], lfhf["corr"], lfratio["corr"]])
    lf_mae = _safe_mean([hf["mae"], lfhf["mae"], lfratio["mae"]])

    return {
        "hr": hr,
        "hf": hf,
        "lfhf": lfhf,
        "lfratio": lfratio,
        "lf_composite": {
            "n": float(min(hf["n"], lfhf["n"], lfratio["n"])),
            "mae": lf_mae,
            "mape": lf_mape,
            "corr": lf_corr,
        },
    }


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in columns})


def _append_sample_summary(
    summary_rows: List[Dict[str, object]],
    sample: Sample,
    split_name: str,
    roi_mode: str,
    used_mode: str,
    metric_summary: Dict[str, Dict[str, float]],
) -> None:
    hr = metric_summary["hr"]
    hf = metric_summary["hf"]
    lfhf = metric_summary["lfhf"]
    lfratio = metric_summary["lfratio"]
    lf_comp = metric_summary["lf_composite"]
    summary_rows.append(
        {
            "split": split_name,
            "group": sample.group,
            "stem": sample.stem,
            "roi_mode_requested": roi_mode,
            "roi_mode_used": used_mode,
            "video_path": str(sample.video_path),
            "ecg_csv": str(sample.ecg_csv_path),
            "hr_n": hr["n"],
            "hr_mae": hr["mae"],
            "hr_rmse": hr["rmse"],
            "hr_mape": hr["mape"],
            "hr_corr": hr["corr"],
            "hf_n": hf["n"],
            "hf_mae": hf["mae"],
            "hf_mape": hf["mape"],
            "hf_corr": hf["corr"],
            "lfhf_n": lfhf["n"],
            "lfhf_mae": lfhf["mae"],
            "lfhf_mape": lfhf["mape"],
            "lfhf_corr": lfhf["corr"],
            "lfratio_n": lfratio["n"],
            "lfratio_mae": lfratio["mae"],
            "lfratio_mape": lfratio["mape"],
            "lfratio_corr": lfratio["corr"],
            "lf_composite_mae": lf_comp["mae"],
            "lf_composite_mape": lf_comp["mape"],
            "lf_composite_corr": lf_comp["corr"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline rPPG-vs-ECG evaluation for Demo2 data")
    parser.add_argument("--data-dir", default="/Users/liangwenwang/Downloads/Code/Demo2")
    parser.add_argument("--out-dir", default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results")
    parser.add_argument(
        "--roi-mode",
        choices=["auto", "mediapipe", "opencv", "both"],
        default="both",
        help="ROI mode for evaluation. both = run mediapipe and opencv separately",
    )
    parser.add_argument("--roi-preset", choices=["classic4", "hybrid7", "cheek_forehead6", "whole_face"], default="hybrid7")
    parser.add_argument("--roi-scale-x", type=float, default=1.1)
    parser.add_argument("--roi-scale-y", type=float, default=1.1)
    parser.add_argument("--roi-shift-y", type=float, default=0.0)
    parser.add_argument("--mp-face-detector-model", default="")
    parser.add_argument("--strict-roi", action="store_true")
    parser.add_argument("--use-published", action="store_true")
    parser.add_argument("--enable-ppi-assist", action="store_true")
    parser.add_argument("--disable-ppi-assist", action="store_true")
    parser.add_argument("--enable-adaptive-roi", action="store_true")
    parser.add_argument("--disable-adaptive-roi", action="store_true")
    parser.add_argument("--debug-adaptive-roi", action="store_true")
    parser.add_argument("--split-file", default="", help="optional split definition file (train/test)")
    parser.add_argument("--holdout-list", default="", help="comma-separated test items like 001/3-3,002/3-4")
    parser.add_argument("--split-set", choices=["all", "train", "test"], default="all")
    parser.add_argument("--align-mode", choices=["timestamp", "index"], default="timestamp")
    parser.add_argument("--metrics", default="hr,lf", help="comma-separated metric families: hr,lf,hf,lfhf,lfratio")
    parser.add_argument("--lf-window-sec", type=float, default=30.0)
    parser.add_argument("--lf-resample-fs", type=float, default=4.0)
    parser.add_argument(
        "--config-profile",
        choices=["python_latest", "ios_like_v413"],
        default="python_latest",
        help="parameter profile used to evaluate rPPG pipeline",
    )
    parser.add_argument("--disable-cbcr", action="store_true", help="exclude CBCR candidates from offline fusion")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    samples = discover_samples(data_dir)
    if not samples:
        raise RuntimeError(f"no paired samples found under {data_dir}")

    split_map = build_split_map(samples, split_file=args.split_file, holdout_list=args.holdout_list)
    metrics = parse_metrics_arg(args.metrics)

    roi_modes = ["mediapipe", "opencv"] if args.roi_mode == "both" else [args.roi_mode]
    cfg = Config()
    apply_config_profile(cfg, args.config_profile)
    if args.enable_ppi_assist and args.disable_ppi_assist:
        raise RuntimeError("cannot set both --enable-ppi-assist and --disable-ppi-assist")
    if args.enable_ppi_assist:
        cfg.enable_ppi_assist = True
    if args.disable_ppi_assist:
        cfg.enable_ppi_assist = False
    if args.enable_adaptive_roi and args.disable_adaptive_roi:
        raise RuntimeError("cannot set both --enable-adaptive-roi and --disable-adaptive-roi")
    if args.enable_adaptive_roi:
        cfg.enable_adaptive_roi = True
    if args.disable_adaptive_roi:
        cfg.enable_adaptive_roi = False
    if args.debug_adaptive_roi:
        cfg.adaptive_roi_debug = True
    cfg.roi_preset = args.roi_preset
    cfg.roi_scale_x = args.roi_scale_x
    cfg.roi_scale_y = args.roi_scale_y
    cfg.roi_shift_y = args.roi_shift_y

    for roi_mode in roi_modes:
        all_detail_rows: List[Dict[str, object]] = []
        summary_rows: List[Dict[str, object]] = []
        per_split_rows: Dict[str, List[Dict[str, object]]] = {"train": [], "test": []}
        print(
            f"\n[MODE] profile={args.config_profile} roi_mode={roi_mode} "
            f"metrics={sorted(metrics)} align={args.align_mode}"
        )

        mode_failed = False
        for s in samples:
            token = sample_token(s.group, s.stem)
            split_name = split_map.get(token, "train")
            if args.split_set != "all" and split_name != args.split_set:
                continue

            print(f"[RUN] split={split_name} group={s.group} stem={s.stem} video={s.video_path.name}")
            try:
                pred_rows = process_video(
                    s,
                    cfg,
                    roi_mode=roi_mode,
                    mp_face_detector_model=args.mp_face_detector_model,
                    strict_roi=args.strict_roi,
                    include_cbcr=not args.disable_cbcr,
                    lf_window_sec=args.lf_window_sec,
                    lf_resample_fs=args.lf_resample_fs,
                )
            except Exception as e:
                print(f"[SKIP] roi_mode={roi_mode} failed: {type(e).__name__}: {e}")
                mode_failed = True
                break

            ecg = load_ecg_series(s.ecg_csv_path, align_mode=args.align_mode)
            compared = compare_to_ecg(
                pred_rows,
                ecg,
                use_published=args.use_published,
                metrics=metrics,
            )

            used_mode = compared[0]["roi_mode"] if compared else "none"
            for r in compared:
                r["split"] = split_name
                r["config_profile"] = args.config_profile
            all_detail_rows.extend(compared)
            per_split_rows.setdefault(split_name, []).extend(compared)

            metric_summary = summarize_all_metrics(compared)
            _append_sample_summary(summary_rows, s, split_name, roi_mode, used_mode, metric_summary)
            hr = metric_summary["hr"]
            lf = metric_summary["lf_composite"]
            print(
                f"      HR: n={int(hr['n'])} MAE={hr['mae']:.3f} RMSE={hr['rmse']:.3f} corr={hr['corr']:.3f} | "
                f"LF: MAE={lf['mae']:.3f} MAPE={lf['mape']:.2f}% corr={lf['corr']:.3f}"
            )

        if mode_failed:
            continue

        def _append_overall(split_name: str, rows: List[Dict[str, object]]) -> None:
            m = summarize_all_metrics(rows)
            summary_rows.append(
                {
                    "split": split_name,
                    "group": "ALL",
                    "stem": "ALL",
                    "roi_mode_requested": roi_mode,
                    "roi_mode_used": "mixed",
                    "video_path": "-",
                    "ecg_csv": "-",
                    "hr_n": m["hr"]["n"],
                    "hr_mae": m["hr"]["mae"],
                    "hr_rmse": m["hr"]["rmse"],
                    "hr_mape": m["hr"]["mape"],
                    "hr_corr": m["hr"]["corr"],
                    "hf_n": m["hf"]["n"],
                    "hf_mae": m["hf"]["mae"],
                    "hf_mape": m["hf"]["mape"],
                    "hf_corr": m["hf"]["corr"],
                    "lfhf_n": m["lfhf"]["n"],
                    "lfhf_mae": m["lfhf"]["mae"],
                    "lfhf_mape": m["lfhf"]["mape"],
                    "lfhf_corr": m["lfhf"]["corr"],
                    "lfratio_n": m["lfratio"]["n"],
                    "lfratio_mae": m["lfratio"]["mae"],
                    "lfratio_mape": m["lfratio"]["mape"],
                    "lfratio_corr": m["lfratio"]["corr"],
                    "lf_composite_mae": m["lf_composite"]["mae"],
                    "lf_composite_mape": m["lf_composite"]["mape"],
                    "lf_composite_corr": m["lf_composite"]["corr"],
                }
            )

        if per_split_rows.get("train"):
            _append_overall("TRAIN_ALL", per_split_rows["train"])
        if per_split_rows.get("test"):
            _append_overall("TEST_ALL", per_split_rows["test"])
        if all_detail_rows:
            _append_overall("ALL", all_detail_rows)
        for row in summary_rows:
            row["config_profile"] = args.config_profile

        detail_cols = [
            "split",
            "config_profile",
            "group",
            "stem",
            "roi_mode_requested",
            "roi_mode",
            "video_path",
            "ecg_csv",
            "video_fps",
            "fs_est",
            "sec",
            "ecg_sec",
            "ecg_source",
            "ecg_hr_interpolated",
            "ecg_hf_interpolated",
            "ecg_lfhf_interpolated",
            "ecg_lfratio_interpolated",
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
            "hf_est",
            "lfhf_est",
            "lfratio_est",
            "ecg_hf",
            "est_hf",
            "error_hf",
            "abs_error_hf",
            "ape_percent_hf",
            "ecg_lfhf",
            "est_lfhf",
            "error_lfhf",
            "abs_error_lfhf",
            "ape_percent_lfhf",
            "ecg_lfratio",
            "est_lfratio",
            "error_lfratio",
            "abs_error_lfratio",
            "ape_percent_lfratio",
            "error",
            "abs_error",
            "ape_percent",
            "est_hr",
            "error_hr",
            "abs_error_hr",
            "ape_percent_hr",
        ]

        summary_cols = [
            "split",
            "config_profile",
            "group",
            "stem",
            "roi_mode_requested",
            "roi_mode_used",
            "video_path",
            "ecg_csv",
            "hr_n",
            "hr_mae",
            "hr_rmse",
            "hr_mape",
            "hr_corr",
            "hf_n",
            "hf_mae",
            "hf_mape",
            "hf_corr",
            "lfhf_n",
            "lfhf_mae",
            "lfhf_mape",
            "lfhf_corr",
            "lfratio_n",
            "lfratio_mae",
            "lfratio_mape",
            "lfratio_corr",
            "lf_composite_mae",
            "lf_composite_mape",
            "lf_composite_corr",
        ]

        metric_tag = "published" if args.use_published else "best"
        profile_suffix = "" if args.config_profile == "python_latest" else f"_{args.config_profile}"
        detail_path = out_dir / f"rppg_ecg_comparison_{metric_tag}_{roi_mode}_{args.align_mode}{profile_suffix}.csv"
        summary_path = out_dir / f"rppg_ecg_summary_{metric_tag}_{roi_mode}_{args.align_mode}{profile_suffix}.csv"

        write_csv(detail_path, all_detail_rows, detail_cols)
        write_csv(summary_path, summary_rows, summary_cols)

        overall = summarize_all_metrics(all_detail_rows)
        print(f"\n[OK] detail:  {detail_path}")
        print(f"[OK] summary: {summary_path}")
        print(
            f"[ALL] HR_MAE={overall['hr']['mae']:.3f} HR_RMSE={overall['hr']['rmse']:.3f} "
            f"HR_corr={overall['hr']['corr']:.3f} | "
            f"LF_MAE={overall['lf_composite']['mae']:.3f} LF_MAPE={overall['lf_composite']['mape']:.2f}% "
            f"LF_corr={overall['lf_composite']['corr']:.3f}"
        )


if __name__ == "__main__":
    main()
