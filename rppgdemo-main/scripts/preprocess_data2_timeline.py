#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


def _to_float(v: str) -> Optional[float]:
    txt = (v or "").strip()
    if not txt:
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def _clock_to_seconds(value: str) -> Optional[int]:
    txt = (value or "").strip()
    if not txt:
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


def _sec_to_clock(sec: int) -> str:
    x = int(sec) % (24 * 3600)
    hh = x // 3600
    mm = (x % 3600) // 60
    ss = x % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _clock_delta_sec(start_sec: int, end_sec: int) -> int:
    delta = int(end_sec) - int(start_sec)
    day = 24 * 3600
    if delta > day // 2:
        delta -= day
    elif delta < -day // 2:
        delta += day
    return delta


def _read_csv_rows_with_fallback(csv_path: Path) -> List[List[str]]:
    encodings = ("utf-8-sig", "utf-8", "cp932", "shift_jis", "latin1")
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            with csv_path.open("r", encoding=enc, newline="") as f:
                return list(csv.reader(f))
        except UnicodeDecodeError as e:
            last_err = e
    raise RuntimeError(f"failed to decode CSV: {csv_path}: {last_err}")


def _read_ecg_table(csv_path: Path) -> Tuple[List[str], List[List[str]], str]:
    rows = _read_csv_rows_with_fallback(csv_path)
    if not rows:
        raise RuntimeError(f"invalid ECG CSV (empty): {csv_path}")

    start_date = ""
    header_idx: Optional[int] = None
    for i, row in enumerate(rows):
        cols = [c.strip() for c in row]
        if cols and cols[0].lower() == "start" and len(cols) > 1:
            start_date = cols[1]
        if not cols:
            continue
        upper = {c.upper() for c in cols if c}
        lower = {c.lower() for c in cols if c}
        if "HR" in upper and "time" in lower:
            header_idx = i
            break

    if header_idx is None:
        raise RuntimeError(f"cannot find ECG header row with time/HR: {csv_path}")

    header = [c.strip() for c in rows[header_idx]]
    data_rows = rows[header_idx + 1 :]
    return header, data_rows, start_date


def _table_col_idx(header: List[str], col: str) -> Optional[int]:
    key = col.strip().lower()
    for i, name in enumerate(header):
        if name.strip().lower() == key:
            return i
    return None


def _safe_cell(row: List[str], idx: Optional[int]) -> str:
    if idx is None or idx < 0 or idx >= len(row):
        return ""
    return row[idx]


def _parse_ecg_bounds(csv_path: Path) -> Tuple[str, int, int]:
    header, data_rows, start_date = _read_ecg_table(csv_path)
    time_idx = _table_col_idx(header, "time")
    if time_idx is None:
        raise RuntimeError(f"time column not found in ECG CSV: {csv_path}")
    secs: List[int] = []
    for row in data_rows:
        sec = _clock_to_seconds(_safe_cell(row, time_idx))
        if sec is not None:
            secs.append(sec)
    if not secs:
        raise RuntimeError(f"no valid time rows in ECG CSV: {csv_path}")
    return start_date, secs[0], secs[-1]


def _parse_condition_token(stem: str) -> str:
    m = re.match(r"^(C\d+)(?:[-_].*)?$", stem.strip(), flags=re.IGNORECASE)
    return m.group(1).upper() if m else ""


def _probe_video_meta(video_path: Path) -> Tuple[Optional[int], Optional[float]]:
    start_sec: Optional[int] = None
    duration_sec: Optional[float] = None

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:format_tags=creation_time:stream_tags=creation_time",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            payload = json.loads(proc.stdout)
            fmt = payload.get("format", {}) if isinstance(payload, dict) else {}
            streams = payload.get("streams", []) if isinstance(payload, dict) else []
            duration_txt = fmt.get("duration")
            if duration_txt is not None:
                duration_sec = _to_float(str(duration_txt))

            creation_raw = ""
            if isinstance(fmt.get("tags"), dict):
                creation_raw = str(fmt["tags"].get("creation_time", "")).strip()
            if not creation_raw and isinstance(streams, list):
                for s in streams:
                    tags = s.get("tags", {}) if isinstance(s, dict) else {}
                    if isinstance(tags, dict):
                        creation_raw = str(tags.get("creation_time", "")).strip()
                        if creation_raw:
                            break
            if creation_raw:
                iso = creation_raw.replace("Z", "+00:00")
                dt = datetime.fromisoformat(iso)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                local_dt = dt.astimezone()
                start_sec = local_dt.hour * 3600 + local_dt.minute * 60 + local_dt.second
    except Exception:
        pass

    if duration_sec is None:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            if fps > 1e-6 and frames > 0:
                duration_sec = frames / fps
        cap.release()

    if start_sec is None:
        # Lenovo filenames: WIN_YYYYMMDD_HH_MM_SS_Pro.mp4
        m = re.search(r"_(\d{2})_(\d{2})_(\d{2})_", video_path.name)
        if m:
            hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
            start_sec = hh * 3600 + mm * 60 + ss

    return start_sec, duration_sec


@dataclass
class RefWindow:
    condition: str
    start_sec: int
    end_sec: int
    duration_sec: float
    source: str


def _build_reference_windows(data2_dir: Path, ref_device: str, ecg_base_sec: int) -> Dict[str, RefWindow]:
    video_dir = data2_dir / ref_device / "video"
    out: Dict[str, RefWindow] = {}
    if not video_dir.exists():
        return out

    for video_path in sorted(video_dir.iterdir()):
        if not video_path.is_file() or video_path.suffix.lower() not in {".mp4", ".mov"}:
            continue
        cond = _parse_condition_token(video_path.stem)
        if not cond:
            continue
        start_sec, duration_sec = _probe_video_meta(video_path)
        if start_sec is None:
            continue
        dur = float(duration_sec or 0.0)
        end_sec = int(round(start_sec + dur))
        out[cond] = RefWindow(
            condition=cond,
            start_sec=start_sec,
            end_sec=end_sec,
            duration_sec=dur,
            source=ref_device,
        )

    # Fallback if ref device metadata is not readable.
    if not out and ref_device != "lenovo-720p-30fps":
        return _build_reference_windows(data2_dir, "lenovo-720p-30fps", ecg_base_sec)
    return out


def _cond_sort_key(cond: str) -> Tuple[int, str]:
    m = re.match(r"^C(\d+)$", cond.upper())
    if m:
        return int(m.group(1)), cond
    return 10**9, cond


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in columns})


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Data2 video-ECG timeline mapping table")
    parser.add_argument("--data2-dir", default="/Users/liangwenwang/Downloads/Code/Demo2/Data2")
    parser.add_argument("--ecg-csv", default="")
    parser.add_argument("--ref-device", default="iphone13pro-1080p-30fps")
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--condition-csv", default="")
    args = parser.parse_args()

    data2_dir = Path(args.data2_dir)
    if not data2_dir.exists():
        raise RuntimeError(f"data2 dir not found: {data2_dir}")

    ecg_csv = Path(args.ecg_csv) if args.ecg_csv else data2_dir / "csvdata" / "C1-1_1.csv"
    if not ecg_csv.exists():
        raise RuntimeError(f"ecg csv not found: {ecg_csv}")

    out_csv = Path(args.out_csv) if args.out_csv else data2_dir / "csvdata" / "data2_video_timeline.csv"
    cond_csv = Path(args.condition_csv) if args.condition_csv else data2_dir / "csvdata" / "data2_condition_windows.csv"

    ecg_date, ecg_first_sec, ecg_last_sec = _parse_ecg_bounds(ecg_csv)
    ref_windows = _build_reference_windows(data2_dir, args.ref_device, ecg_first_sec)
    if not ref_windows:
        raise RuntimeError("failed to build reference windows (check video metadata/filenames)")

    device_dirs = sorted(p for p in data2_dir.iterdir() if p.is_dir() and (p / "video").exists())

    rows: List[Dict[str, object]] = []
    for dev_dir in device_dirs:
        for video_path in sorted((dev_dir / "video").iterdir()):
            if not video_path.is_file() or video_path.suffix.lower() not in {".mp4", ".mov"}:
                continue
            cond = _parse_condition_token(video_path.stem)
            if not cond:
                continue
            ref = ref_windows.get(cond)
            if ref is None:
                continue

            own_start_sec, own_duration_sec = _probe_video_meta(video_path)
            own_end_sec: Optional[int] = None
            if own_start_sec is not None and own_duration_sec is not None:
                own_end_sec = int(round(own_start_sec + own_duration_sec))

            ecg_offset_sec = _clock_delta_sec(ecg_first_sec, ref.start_sec)
            ecg_win_start_sec = ecg_offset_sec
            ecg_win_end_sec = ecg_offset_sec + int(round(ref.duration_sec))
            rows.append(
                {
                    "condition": cond,
                    "device": dev_dir.name,
                    "video_stem": video_path.stem,
                    "video_file": video_path.name,
                    "video_path": str(video_path),
                    "ecg_csv": str(ecg_csv),
                    "ecg_date": ecg_date,
                    "ecg_global_start_clock": _sec_to_clock(ecg_first_sec),
                    "ecg_global_end_clock": _sec_to_clock(ecg_last_sec),
                    "segment_source": ref.source,
                    "ref_start_clock": _sec_to_clock(ref.start_sec),
                    "ref_end_clock": _sec_to_clock(ref.end_sec),
                    "ref_duration_sec": round(ref.duration_sec, 3),
                    "video_start_clock": _sec_to_clock(own_start_sec) if own_start_sec is not None else "",
                    "video_end_clock": _sec_to_clock(own_end_sec) if own_end_sec is not None else "",
                    "video_duration_sec": round(float(own_duration_sec), 3) if own_duration_sec is not None else "",
                    "ecg_offset_sec": ecg_offset_sec,
                    "ecg_window_start_sec": ecg_win_start_sec,
                    "ecg_window_end_sec": ecg_win_end_sec,
                    "ecg_window_start_clock": _sec_to_clock(ref.start_sec),
                    "ecg_window_end_clock": _sec_to_clock(ref.end_sec),
                }
            )

    rows.sort(key=lambda r: (_cond_sort_key(str(r["condition"])), str(r["device"]), str(r["video_stem"])))

    columns = [
        "condition",
        "device",
        "video_stem",
        "video_file",
        "video_path",
        "ecg_csv",
        "ecg_date",
        "ecg_global_start_clock",
        "ecg_global_end_clock",
        "segment_source",
        "ref_start_clock",
        "ref_end_clock",
        "ref_duration_sec",
        "video_start_clock",
        "video_end_clock",
        "video_duration_sec",
        "ecg_offset_sec",
        "ecg_window_start_sec",
        "ecg_window_end_sec",
        "ecg_window_start_clock",
        "ecg_window_end_clock",
    ]
    write_csv(out_csv, rows, columns)

    cond_rows = []
    for cond, ref in sorted(ref_windows.items(), key=lambda kv: _cond_sort_key(kv[0])):
        ecg_offset_sec = _clock_delta_sec(ecg_first_sec, ref.start_sec)
        cond_rows.append(
            {
                "condition": cond,
                "segment_source": ref.source,
                "ref_start_clock": _sec_to_clock(ref.start_sec),
                "ref_end_clock": _sec_to_clock(ref.end_sec),
                "ref_duration_sec": round(ref.duration_sec, 3),
                "ecg_offset_sec": ecg_offset_sec,
                "ecg_window_start_sec": ecg_offset_sec,
                "ecg_window_end_sec": ecg_offset_sec + int(round(ref.duration_sec)),
            }
        )
    write_csv(
        cond_csv,
        cond_rows,
        [
            "condition",
            "segment_source",
            "ref_start_clock",
            "ref_end_clock",
            "ref_duration_sec",
            "ecg_offset_sec",
            "ecg_window_start_sec",
            "ecg_window_end_sec",
        ],
    )

    print(f"[OK] timeline csv: {out_csv}")
    print(f"[OK] condition csv: {cond_csv}")
    print(f"[INFO] rows={len(rows)} conditions={len(cond_rows)} ecg={ecg_csv.name}")


if __name__ == "__main__":
    main()
