#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def _to_float(row: Dict[str, str], key: str) -> float:
    txt = (row.get(key) or "").strip()
    if txt == "":
        return math.nan
    try:
        return float(txt)
    except ValueError:
        return math.nan


def _load_overall_row(summary_csv: Path) -> Dict[str, str]:
    if not summary_csv.exists():
        raise RuntimeError(f"summary csv not found: {summary_csv}")
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("split") or "").strip() == "ALL":
                return row
    raise RuntimeError(f"ALL row not found in: {summary_csv}")


def _run_eval(
    python_exe: str,
    repo_dir: Path,
    out_dir: Path,
    profile: str,
    data_dir: Path,
    roi_mode: str,
    split_file: str,
    split_set: str,
    align_mode: str,
    metrics: str,
    lf_window_sec: float,
    lf_resample_fs: float,
    mp_face_detector_model: str,
    strict_roi: bool,
    use_published: bool,
    disable_cbcr: bool,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        python_exe,
        str(repo_dir / "evaluate_dataset.py"),
        "--data-dir",
        str(data_dir),
        "--out-dir",
        str(out_dir),
        "--roi-mode",
        roi_mode,
        "--split-file",
        split_file,
        "--split-set",
        split_set,
        "--align-mode",
        align_mode,
        "--metrics",
        metrics,
        "--lf-window-sec",
        str(lf_window_sec),
        "--lf-resample-fs",
        str(lf_resample_fs),
        "--config-profile",
        profile,
    ]
    if mp_face_detector_model.strip():
        cmd.extend(["--mp-face-detector-model", mp_face_detector_model.strip()])
    if strict_roi:
        cmd.append("--strict-roi")
    if use_published:
        cmd.append("--use-published")
    if disable_cbcr:
        cmd.append("--disable-cbcr")

    print(f"[RUN] profile={profile} roi={roi_mode} align={align_mode}")
    subprocess.run(cmd, check=True)

    metric_tag = "published" if use_published else "best"
    profile_suffix = "" if profile == "python_latest" else f"_{profile}"
    summary_csv = out_dir / f"rppg_ecg_summary_{metric_tag}_{roi_mode}_{align_mode}{profile_suffix}.csv"
    detail_csv = out_dir / f"rppg_ecg_comparison_{metric_tag}_{roi_mode}_{align_mode}{profile_suffix}.csv"
    row_all = _load_overall_row(summary_csv)
    return {
        "command": cmd,
        "summary_csv": str(summary_csv),
        "detail_csv": str(detail_csv),
        "all_row": row_all,
    }


def _metric_rows(py_row: Dict[str, str], ios_row: Dict[str, str]) -> List[Dict[str, object]]:
    metrics = [
        ("hr_mae", "lower_better"),
        ("hr_rmse", "lower_better"),
        ("hr_mape", "lower_better"),
        ("hr_corr", "higher_better"),
        ("hf_mae", "lower_better"),
        ("lfhf_mae", "lower_better"),
        ("lfratio_mae", "lower_better"),
        ("lf_composite_mae", "lower_better"),
        ("lf_composite_mape", "lower_better"),
        ("lf_composite_corr", "higher_better"),
    ]

    out: List[Dict[str, object]] = []
    for metric, direction in metrics:
        py_v = _to_float(py_row, metric)
        ios_v = _to_float(ios_row, metric)
        if not (math.isfinite(py_v) and math.isfinite(ios_v)):
            pct = math.nan
        elif direction == "lower_better":
            denom = abs(ios_v) if abs(ios_v) > 1e-9 else math.nan
            pct = ((ios_v - py_v) / denom * 100.0) if math.isfinite(denom) else math.nan
        else:
            denom = abs(ios_v) if abs(ios_v) > 1e-9 else math.nan
            pct = ((py_v - ios_v) / denom * 100.0) if math.isfinite(denom) else math.nan
        out.append(
            {
                "metric": metric,
                "direction": direction,
                "python_latest": py_v,
                "ios_like_v413": ios_v,
                "python_vs_ios_change_percent": pct,
            }
        )
    return out


def _write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Compare Python latest profile vs iOS-like profile using same videos and ECG CSV.")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--data-dir", default="/Users/liangwenwang/Downloads/Code/Demo2")
    parser.add_argument("--out-dir", default=str(repo_dir / "results" / "compare_python_ios"))
    parser.add_argument("--roi-mode", choices=["mediapipe", "opencv"], default="opencv")
    parser.add_argument("--split-file", default=str(repo_dir / "splits" / "fixed_holdout_6.txt"))
    parser.add_argument("--split-set", choices=["all", "train", "test"], default="test")
    parser.add_argument("--align-mode", choices=["timestamp", "index"], default="timestamp")
    parser.add_argument("--metrics", default="hr,lf")
    parser.add_argument("--lf-window-sec", type=float, default=30.0)
    parser.add_argument("--lf-resample-fs", type=float, default=4.0)
    parser.add_argument("--mp-face-detector-model", default="")
    parser.add_argument("--strict-roi", action="store_true")
    parser.add_argument("--use-published", action="store_true")
    parser.add_argument("--disable-cbcr", action="store_true")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / run_id
    py_dir = run_dir / "python_latest"
    ios_dir = run_dir / "ios_like_v413"

    py_res = _run_eval(
        python_exe=args.python_exe,
        repo_dir=repo_dir,
        out_dir=py_dir,
        profile="python_latest",
        data_dir=Path(args.data_dir),
        roi_mode=args.roi_mode,
        split_file=args.split_file,
        split_set=args.split_set,
        align_mode=args.align_mode,
        metrics=args.metrics,
        lf_window_sec=args.lf_window_sec,
        lf_resample_fs=args.lf_resample_fs,
        mp_face_detector_model=args.mp_face_detector_model,
        strict_roi=args.strict_roi,
        use_published=args.use_published,
        disable_cbcr=args.disable_cbcr,
    )
    ios_res = _run_eval(
        python_exe=args.python_exe,
        repo_dir=repo_dir,
        out_dir=ios_dir,
        profile="ios_like_v413",
        data_dir=Path(args.data_dir),
        roi_mode=args.roi_mode,
        split_file=args.split_file,
        split_set=args.split_set,
        align_mode=args.align_mode,
        metrics=args.metrics,
        lf_window_sec=args.lf_window_sec,
        lf_resample_fs=args.lf_resample_fs,
        mp_face_detector_model=args.mp_face_detector_model,
        strict_roi=args.strict_roi,
        use_published=args.use_published,
        disable_cbcr=args.disable_cbcr,
    )

    py_all = py_res["all_row"]
    ios_all = ios_res["all_row"]
    assert isinstance(py_all, dict)
    assert isinstance(ios_all, dict)
    metric_rows = _metric_rows(py_all, ios_all)

    compare_csv = run_dir / "python_latest_vs_ios_like_comparison.csv"
    _write_csv(
        compare_csv,
        metric_rows,
        columns=[
            "metric",
            "direction",
            "python_latest",
            "ios_like_v413",
            "python_vs_ios_change_percent",
        ],
    )

    meta = {
        "run_dir": str(run_dir),
        "roi_mode": args.roi_mode,
        "split_file": args.split_file,
        "split_set": args.split_set,
        "align_mode": args.align_mode,
        "metrics": args.metrics,
        "python_latest": py_res,
        "ios_like_v413": ios_res,
        "comparison_csv": str(compare_csv),
    }
    meta_path = run_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")

    def _show(metric: str) -> str:
        row = next((r for r in metric_rows if r["metric"] == metric), None)
        if row is None:
            return "NA"
        py_v = row["python_latest"]
        ios_v = row["ios_like_v413"]
        pct = row["python_vs_ios_change_percent"]
        if not isinstance(py_v, float) or not isinstance(ios_v, float):
            return "NA"
        pct_txt = "NA" if not (isinstance(pct, float) and math.isfinite(pct)) else f"{pct:+.2f}%"
        return f"py={py_v:.4f}, ios_like={ios_v:.4f}, delta={pct_txt}"

    print(f"[OK] run_dir={run_dir}")
    print(f"[OK] comparison_csv={compare_csv}")
    print(f"[ALL] HR_MAE: {_show('hr_mae')}")
    print(f"[ALL] HR_corr: {_show('hr_corr')}")
    print(f"[ALL] LF_composite_MAE: {_show('lf_composite_mae')}")
    print(f"[ALL] LF_composite_corr: {_show('lf_composite_corr')}")


if __name__ == "__main__":
    main()

