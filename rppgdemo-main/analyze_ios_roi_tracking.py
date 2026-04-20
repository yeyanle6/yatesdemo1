#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Row:
    frame_index: int
    sec: int
    region_name: str
    roi_x: float
    roi_y: float
    roi_w: float
    roi_h: float
    face_x: float
    face_y: float
    face_w: float
    face_h: float


def _to_float(v: str) -> float:
    try:
        return float((v or "").strip())
    except Exception:
        return math.nan


def _to_int(v: str) -> int:
    try:
        return int(float((v or "").strip()))
    except Exception:
        return -1


def canonical_region(name: str) -> str:
    n = (name or "").strip().lower()
    if "前额" in name or "forehead" in n:
        return "forehead"
    if "眉间" in name or "glabella" in n:
        return "glabella"
    if ("左" in name and "颧" in name) or "left" in n:
        return "left_cheek"
    if ("右" in name and "颧" in name) or "right" in n:
        return "right_cheek"
    return n or "unknown"


def std(vals: List[float]) -> float:
    xs = [x for x in vals if math.isfinite(x)]
    if len(xs) < 2:
        return math.nan
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def mean(vals: List[float]) -> float:
    xs = [x for x in vals if math.isfinite(x)]
    if not xs:
        return math.nan
    return sum(xs) / len(xs)


def load_rows(path: Path) -> List[Row]:
    out: List[Row] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out.append(
                Row(
                    frame_index=_to_int(r.get("frame_index", "")),
                    sec=_to_int(r.get("sec", "")),
                    region_name=r.get("region_name", ""),
                    roi_x=_to_float(r.get("roi_x", "")),
                    roi_y=_to_float(r.get("roi_y", "")),
                    roi_w=_to_float(r.get("roi_w", "")),
                    roi_h=_to_float(r.get("roi_h", "")),
                    face_x=_to_float(r.get("face_x", "")),
                    face_y=_to_float(r.get("face_y", "")),
                    face_w=_to_float(r.get("face_w", "")),
                    face_h=_to_float(r.get("face_h", "")),
                )
            )
    return out


def analyze_sample(csv_path: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows = load_rows(csv_path)
    frames: Dict[int, List[Row]] = defaultdict(list)
    for r in rows:
        if r.frame_index >= 0:
            frames[r.frame_index].append(r)

    total_frames = len(frames)
    if total_frames == 0:
        return [], {
            "sample": csv_path.stem,
            "total_frames": 0,
            "left_right_order_ratio": math.nan,
            "roi_presence_mean": math.nan,
        }

    per_region_rel: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    per_region_face_rel: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    present_frames_by_region: Dict[str, set[int]] = defaultdict(set)

    left_right_checks = 0
    left_before_right = 0

    for frame_idx, frows in frames.items():
        valid = [x for x in frows if x.roi_w > 1e-6 and x.roi_h > 1e-6]
        if not valid:
            continue
        min_x = min(x.roi_x for x in valid)
        min_y = min(x.roi_y for x in valid)
        max_x = max(x.roi_x + x.roi_w for x in valid)
        max_y = max(x.roi_y + x.roi_h for x in valid)
        uw = max(1e-9, max_x - min_x)
        uh = max(1e-9, max_y - min_y)

        left_cx = None
        right_cx = None

        for r in valid:
            reg = canonical_region(r.region_name)
            present_frames_by_region[reg].add(frame_idx)

            cx = r.roi_x + 0.5 * r.roi_w
            cy = r.roi_y + 0.5 * r.roi_h
            rel_cx = (cx - min_x) / uw
            rel_cy = (cy - min_y) / uh
            rel_w = r.roi_w / uw
            rel_h = r.roi_h / uh

            per_region_rel[reg]["rel_cx"].append(rel_cx)
            per_region_rel[reg]["rel_cy"].append(rel_cy)
            per_region_rel[reg]["rel_w"].append(rel_w)
            per_region_rel[reg]["rel_h"].append(rel_h)

            if r.face_w > 1e-9 and r.face_h > 1e-9 and math.isfinite(r.face_x) and math.isfinite(r.face_y):
                face_rel_cx = (cx - r.face_x) / r.face_w
                face_rel_cy = (cy - r.face_y) / r.face_h
                face_rel_w = r.roi_w / r.face_w
                face_rel_h = r.roi_h / r.face_h
                per_region_face_rel[reg]["face_rel_cx"].append(face_rel_cx)
                per_region_face_rel[reg]["face_rel_cy"].append(face_rel_cy)
                per_region_face_rel[reg]["face_rel_w"].append(face_rel_w)
                per_region_face_rel[reg]["face_rel_h"].append(face_rel_h)

            if reg == "left_cheek":
                left_cx = cx
            elif reg == "right_cheek":
                right_cx = cx

        if left_cx is not None and right_cx is not None:
            left_right_checks += 1
            if left_cx < right_cx:
                left_before_right += 1

    region_rows: List[Dict[str, object]] = []
    presence_rates: List[float] = []

    for reg, values in sorted(per_region_rel.items()):
        present = len(present_frames_by_region.get(reg, set()))
        presence = present / max(1, total_frames)
        presence_rates.append(presence)

        rel_cx_series = values.get("rel_cx", [])
        rel_cy_series = values.get("rel_cy", [])
        rel_w_series = values.get("rel_w", [])
        rel_h_series = values.get("rel_h", [])
        face_values = per_region_face_rel.get(reg, {})
        face_rel_cx_series = face_values.get("face_rel_cx", [])
        face_rel_cy_series = face_values.get("face_rel_cy", [])
        face_rel_w_series = face_values.get("face_rel_w", [])
        face_rel_h_series = face_values.get("face_rel_h", [])

        jitter = []
        for i in range(1, min(len(rel_cx_series), len(rel_cy_series))):
            dx = rel_cx_series[i] - rel_cx_series[i - 1]
            dy = rel_cy_series[i] - rel_cy_series[i - 1]
            jitter.append(math.sqrt(dx * dx + dy * dy))

        face_jitter = []
        for i in range(1, min(len(face_rel_cx_series), len(face_rel_cy_series))):
            dx = face_rel_cx_series[i] - face_rel_cx_series[i - 1]
            dy = face_rel_cy_series[i] - face_rel_cy_series[i - 1]
            face_jitter.append(math.sqrt(dx * dx + dy * dy))

        region_rows.append(
            {
                "sample": csv_path.stem,
                "region": reg,
                "presence_frames": present,
                "total_frames": total_frames,
                "presence_rate": presence,
                "face_rel_count": min(len(face_rel_cx_series), len(face_rel_cy_series)),
                "rel_cx_std": std(rel_cx_series),
                "rel_cy_std": std(rel_cy_series),
                "rel_w_std": std(rel_w_series),
                "rel_h_std": std(rel_h_series),
                "center_jitter_mean": mean(jitter),
                "face_rel_cx_std": std(face_rel_cx_series),
                "face_rel_cy_std": std(face_rel_cy_series),
                "face_rel_w_std": std(face_rel_w_series),
                "face_rel_h_std": std(face_rel_h_series),
                "face_center_jitter_mean": mean(face_jitter),
            }
        )

    overall = {
        "sample": csv_path.stem,
        "total_frames": total_frames,
        "left_right_order_checks": left_right_checks,
        "left_right_order_ratio": (left_before_right / left_right_checks) if left_right_checks > 0 else math.nan,
        "roi_presence_mean": mean(presence_rates),
    }
    return region_rows, overall


def write_csv(path: Path, rows: List[Dict[str, object]], cols: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in cols})


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze iOS ROI tracking stability from ios_offline_replay ROI debug CSVs.")
    ap.add_argument("--roi-debug-dir", required=True)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    in_dir = Path(args.roi_debug_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir

    files = sorted(
        p for p in in_dir.glob("*.csv")
        if not p.name.startswith("roi_tracking_")
    )
    if not files:
        raise RuntimeError(f"no ROI debug csv found in: {in_dir}")

    region_rows_all: List[Dict[str, object]] = []
    overall_rows: List[Dict[str, object]] = []

    for p in files:
        region_rows, overall = analyze_sample(p)
        region_rows_all.extend(region_rows)
        overall_rows.append(overall)

    region_csv = out_dir / "roi_tracking_region_summary.csv"
    overall_csv = out_dir / "roi_tracking_overall_summary.csv"

    write_csv(
        region_csv,
        region_rows_all,
        [
            "sample",
            "region",
            "presence_frames",
            "total_frames",
            "presence_rate",
            "face_rel_count",
            "rel_cx_std",
            "rel_cy_std",
            "rel_w_std",
            "rel_h_std",
            "center_jitter_mean",
            "face_rel_cx_std",
            "face_rel_cy_std",
            "face_rel_w_std",
            "face_rel_h_std",
            "face_center_jitter_mean",
        ],
    )
    write_csv(
        overall_csv,
        overall_rows,
        [
            "sample",
            "total_frames",
            "left_right_order_checks",
            "left_right_order_ratio",
            "roi_presence_mean",
        ],
    )

    print(f"[OK] region summary:  {region_csv}")
    print(f"[OK] overall summary: {overall_csv}")


if __name__ == "__main__":
    main()
