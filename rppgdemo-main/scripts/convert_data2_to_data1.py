#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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


def _read_ecg_table(ecg_csv: Path) -> Tuple[List[str], List[List[str]]]:
    rows = _read_csv_rows_with_fallback(ecg_csv)
    if not rows:
        raise RuntimeError(f"invalid ECG CSV (empty): {ecg_csv}")

    header_idx: Optional[int] = None
    for i, row in enumerate(rows):
        cols = [c.strip() for c in row]
        upper = {c.upper() for c in cols if c}
        lower = {c.lower() for c in cols if c}
        if "HR" in upper and "time" in lower:
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError(f"cannot find ECG table header (time/HR): {ecg_csv}")

    header = [c.strip() for c in rows[header_idx]]
    return header, rows[header_idx + 1 :]


@dataclass
class ECGRecord:
    abs_sec: int
    row: List[str]


def _build_ecg_records(header: List[str], data_rows: List[List[str]]) -> Tuple[int, List[ECGRecord]]:
    try:
        time_idx = [c.lower() for c in header].index("time")
    except ValueError as e:
        raise RuntimeError("ECG header missing 'time'") from e

    records: List[ECGRecord] = []
    base_clock: Optional[int] = None
    prev_clock: Optional[int] = None
    rollover_days = 0

    for row in data_rows:
        if time_idx >= len(row):
            continue
        sec_clock = _clock_to_seconds(row[time_idx])
        if sec_clock is None:
            continue

        if base_clock is None:
            base_clock = sec_clock
            prev_clock = sec_clock
        else:
            assert prev_clock is not None
            if sec_clock < prev_clock:
                rollover_days += 1
            prev_clock = sec_clock

        abs_sec = sec_clock + rollover_days * 24 * 3600
        records.append(ECGRecord(abs_sec=abs_sec, row=row))

    if not records or base_clock is None:
        raise RuntimeError("no valid ECG time rows")
    return base_clock, records


def _slice_records(records: List[ECGRecord], start_abs: int, end_abs: int) -> List[List[str]]:
    return [r.row for r in records if start_abs <= r.abs_sec <= end_abs]


def _materialize_video(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        dst.symlink_to(src)
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    raise RuntimeError(f"unsupported link mode: {mode}")


def _write_csv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def _ensure_timeline_csv(
    repo_root: Path,
    data2_dir: Path,
    timeline_csv: Path,
) -> None:
    if timeline_csv.exists():
        return
    preprocess = repo_root / "scripts" / "preprocess_data2_timeline.py"
    if not preprocess.exists():
        raise RuntimeError(f"timeline csv not found and preprocess script missing: {preprocess}")
    subprocess.run(
        [
            sys.executable,
            str(preprocess),
            "--data2-dir",
            str(data2_dir),
            "--out-csv",
            str(timeline_csv),
        ],
        check=True,
    )


def _device_to_group_map(devices: List[str]) -> Dict[str, str]:
    # Use 3-digit numeric groups to match Data1 discovery rules.
    out: Dict[str, str] = {}
    for idx, dev in enumerate(sorted(set(devices)), start=1):
        out[dev] = f"{100 + idx:03d}"
    return out


def _read_timeline_rows(timeline_csv: Path) -> List[Dict[str, str]]:
    with timeline_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"timeline csv has no rows: {timeline_csv}")
    required = {"condition", "device", "video_stem", "video_path", "ecg_window_start_sec", "ecg_window_end_sec"}
    missing = [c for c in required if c not in (rows[0].keys() if rows else [])]
    if missing:
        raise RuntimeError(f"timeline csv missing columns: {missing}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Data2 into Data1-like paired dataset format")
    parser.add_argument("--data2-dir", default="/Users/liangwenwang/Downloads/Code/Demo2/Data2")
    parser.add_argument("--out-dir", default="/Users/liangwenwang/Downloads/Code/Demo2/Data2_as_Data1")
    parser.add_argument("--ecg-master-csv", default="")
    parser.add_argument("--timeline-csv", default="")
    parser.add_argument("--link-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    parser.add_argument("--clean-out-dir", action="store_true", help="delete output dir before writing")
    args = parser.parse_args()

    data2_dir = Path(args.data2_dir)
    out_dir = Path(args.out_dir)
    repo_root = Path(__file__).resolve().parents[1]

    if not data2_dir.exists():
        raise RuntimeError(f"data2 dir not found: {data2_dir}")
    if args.clean_out_dir and out_dir.exists():
        shutil.rmtree(out_dir)

    ecg_master = Path(args.ecg_master_csv) if args.ecg_master_csv else data2_dir / "csvdata" / "C1-1_1.csv"
    if not ecg_master.exists():
        raise RuntimeError(f"ecg master csv not found: {ecg_master}")

    timeline_csv = Path(args.timeline_csv) if args.timeline_csv else data2_dir / "csvdata" / "data2_video_timeline.csv"
    _ensure_timeline_csv(repo_root=repo_root, data2_dir=data2_dir, timeline_csv=timeline_csv)
    timeline_rows = _read_timeline_rows(timeline_csv)

    header, data_rows = _read_ecg_table(ecg_master)
    ecg_base_clock, ecg_records = _build_ecg_records(header=header, data_rows=data_rows)

    device_map = _device_to_group_map([r["device"] for r in timeline_rows])
    manifest: List[Dict[str, object]] = []

    for r in timeline_rows:
        device = r["device"]
        group = device_map[device]
        stem = r["video_stem"]
        cond = r["condition"]
        video_src = Path(r["video_path"])
        if not video_src.exists():
            raise RuntimeError(f"video not found: {video_src}")

        try:
            win_start_rel = int(float(r["ecg_window_start_sec"]))
            win_end_rel = int(float(r["ecg_window_end_sec"]))
        except ValueError as e:
            raise RuntimeError(f"invalid window in timeline row: {r}") from e

        start_abs = ecg_base_clock + win_start_rel
        end_abs = ecg_base_clock + win_end_rel
        sliced_rows = _slice_records(ecg_records, start_abs=start_abs, end_abs=end_abs)
        if not sliced_rows:
            raise RuntimeError(
                f"no ECG rows for {device}/{stem} window [{win_start_rel}, {win_end_rel}] "
                f"(abs [{start_abs}, {end_abs}])"
            )

        out_group = out_dir / group
        out_video = out_group / "video" / f"{stem}{video_src.suffix}"
        out_csv = out_group / "csvdata" / f"{stem}.csv"
        _materialize_video(video_src, out_video, mode=args.link_mode)
        _write_csv(out_csv, header=header, rows=sliced_rows)

        manifest.append(
            {
                "group": group,
                "device": device,
                "condition": cond,
                "stem": stem,
                "video_path": str(out_video),
                "ecg_csv_path": str(out_csv),
                "ecg_rows": len(sliced_rows),
                "ecg_window_start_sec": win_start_rel,
                "ecg_window_end_sec": win_end_rel,
            }
        )

    manifest_path = out_dir / "manifest_data2_as_data1.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "device",
                "condition",
                "stem",
                "video_path",
                "ecg_csv_path",
                "ecg_rows",
                "ecg_window_start_sec",
                "ecg_window_end_sec",
            ],
        )
        w.writeheader()
        for row in manifest:
            w.writerow(row)

    device_map_path = out_dir / "device_group_map.csv"
    with device_map_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "device"])
        for dev, grp in sorted(device_map.items(), key=lambda x: x[1]):
            w.writerow([grp, dev])

    print(f"[OK] out dir: {out_dir}")
    print(f"[OK] manifest: {manifest_path}")
    print(f"[OK] device-group map: {device_map_path}")
    print(f"[INFO] samples={len(manifest)} groups={len(device_map)} ecg={ecg_master.name}")


if __name__ == "__main__":
    main()
