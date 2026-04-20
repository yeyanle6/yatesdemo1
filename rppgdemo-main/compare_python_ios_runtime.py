#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from evaluate_dataset import (
    build_split_map,
    compare_to_ecg,
    discover_samples,
    load_ecg_series,
    summarize_metric,
)


def _to_float(text: str) -> Optional[float]:
    txt = text.strip()
    if txt == "":
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def _to_float_nan(text: str) -> float:
    v = _to_float(text)
    return v if v is not None else math.nan


def _write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def _load_all_row(summary_csv: Path) -> Dict[str, str]:
    if not summary_csv.exists():
        raise RuntimeError(f"summary csv not found: {summary_csv}")
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("split") or "").strip() == "ALL" and (row.get("group") or "").strip() == "ALL":
                return row
    raise RuntimeError(f"ALL row not found in: {summary_csv}")


def _load_sample_rows(summary_csv: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = (row.get("group") or "").strip()
            s = (row.get("stem") or "").strip()
            if g == "" or s == "" or (g == "ALL" and s == "ALL"):
                continue
            out[(g, s)] = row
    return out


def _strip_preview_blocks(swift_source: str) -> str:
    """Remove trailing #Preview macro blocks for CLI swiftc compatibility."""
    lines = swift_source.splitlines()
    out: List[str] = []
    skipping = False
    depth = 0
    for line in lines:
        stripped = line.strip()
        if not skipping and stripped.startswith("#Preview"):
            skipping = True
            depth = line.count("{") - line.count("}")
            if depth <= 0:
                skipping = False
                depth = 0
            continue
        if skipping:
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                skipping = False
                depth = 0
            continue
        out.append(line)

    merged = "\n".join(out)
    if swift_source.endswith("\n"):
        merged += "\n"
    return merged


def _prepare_swift_source_for_cli(src: Path, staging_dir: Path) -> Path:
    """Stage source file if it contains unsupported preview macros."""
    text = src.read_text(encoding="utf-8")
    stripped = _strip_preview_blocks(text)
    if stripped == text:
        return src
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged = staging_dir / src.name
    staged.write_text(stripped, encoding="utf-8")
    return staged


def _compile_ios_runner(repo_dir: Path, ios_module_root: Path, bin_path: Path) -> List[str]:
    swift_script = repo_dir / "scripts" / "ios_offline_replay.swift"
    if not swift_script.exists():
        raise RuntimeError(f"missing swift script: {swift_script}")

    staging_dir = bin_path.parent / "_swift_cli_staging"
    gui_coordinator = _prepare_swift_source_for_cli(
        ios_module_root / "Utils" / "RPPGUICoordinator.swift",
        staging_dir=staging_dir,
    )

    files = [
        swift_script,
        ios_module_root / "Services" / "VideoFrameProcessor.swift",
        ios_module_root / "Services" / "RPPGService.swift",
        ios_module_root / "Utils" / "CoordinateTypes.swift",
        ios_module_root / "Utils" / "CoordinateTransformer.swift",
        ios_module_root / "Utils" / "RPPGConstants.swift",
        ios_module_root / "Utils" / "ButterworthFilter.swift",
        ios_module_root / "Services" / "RPPGAlgorithmSupport.swift",
        gui_coordinator,
        ios_module_root / "Utils" / "RPPGSignalProcessor.swift",
        ios_module_root / "Utils" / "AdaptiveROISmoothing.swift",
        ios_module_root / "Utils" / "FaceROITracker.swift",
        ios_module_root / "Views" / "ROIConsistencyValidator.swift",
        ios_module_root / "Utils" / "RPPGProcessor.swift",
    ]
    # Backward compatibility: older iOS branches used CoordinateTransform.swift.
    legacy_coordinate_transform = ios_module_root / "Utils" / "CoordinateTransform.swift"
    if legacy_coordinate_transform.exists():
        files.append(legacy_coordinate_transform)
    for f in files:
        if not f.exists():
            raise RuntimeError(f"missing iOS source: {f}")

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    swift_cache = bin_path.parent / "_swift_module_cache"
    clang_cache = bin_path.parent / "_clang_module_cache"
    swift_cache.mkdir(parents=True, exist_ok=True)
    clang_cache.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["SWIFT_MODULE_CACHE_PATH"] = str(swift_cache)
    env["CLANG_MODULE_CACHE_PATH"] = str(clang_cache)

    cmd = ["swiftc", "-O", "-parse-as-library", *[str(x) for x in files], "-o", str(bin_path)]
    subprocess.run(cmd, check=True, env=env)
    return cmd


def _run_ios_replay(
    bin_path: Path,
    video_path: Path,
    out_csv: Path,
    max_seconds: int,
    ios_preset: str,
    roi_debug_csv: Optional[Path],
    algo_debug_csv: Optional[Path],
    log_path: Path,
) -> List[str]:
    cmd = [str(bin_path), "--video-path", str(video_path), "--out-csv", str(out_csv)]
    if max_seconds >= 0:
        cmd.extend(["--max-seconds", str(max_seconds)])
    cmd.extend(["--preset", ios_preset])
    if roi_debug_csv is not None:
        cmd.extend(["--roi-debug-csv", str(roi_debug_csv)])
    if algo_debug_csv is not None:
        cmd.extend(["--algo-debug-csv", str(algo_debug_csv)])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # Reduce massive per-frame stdout overhead in automation runs.
    env["IOS_REPLAY_SILENT"] = "1"
    with log_path.open("w", encoding="utf-8") as logf:
        subprocess.run(cmd, check=True, stdout=logf, stderr=logf, env=env)
    return cmd


def _load_ios_pred_rows(pred_csv: Path, group: str, stem: str) -> List[Dict[str, object]]:
    if not pred_csv.exists():
        raise RuntimeError(f"missing iOS prediction csv: {pred_csv}")
    out: List[Dict[str, object]] = []
    with pred_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sec_txt = (row.get("sec") or "").strip()
            if sec_txt == "":
                continue
            sec = int(float(sec_txt))
            hr_best = _to_float(row.get("hr_best") or "")
            hr_pub = _to_float(row.get("hr_published") or "")
            conf = _to_float(row.get("confidence") or "")
            sq = _to_float(row.get("signal_quality") or "")
            out.append(
                {
                    "group": group,
                    "stem": stem,
                    "roi_mode_requested": "ios_runtime",
                    "roi_mode": "ios_runtime",
                    "video_path": "",
                    "ecg_csv": "",
                    "video_fps": math.nan,
                    "fs_est": math.nan,
                    "sec": sec,
                    "hr_best": hr_best,
                    "hr_published": hr_pub,
                    "ppi_hr": None,
                    "pnn50": None,
                    "pnn50_reliable": False,
                    "sqi": sq,
                    "frequency_confidence": conf,
                    "snr_db": math.nan,
                    "state": "",
                    "hf_est": None,
                    "lfhf_est": None,
                    "lfratio_est": None,
                }
            )
    return out


def _run_python_eval(
    python_exe: str,
    repo_dir: Path,
    data_dir: Path,
    out_dir: Path,
    roi_mode: str,
    split_file: str,
    split_set: str,
    align_mode: str,
    use_published: bool,
) -> Tuple[List[str], Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
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
        "hr",
        "--config-profile",
        "python_latest",
    ]
    if use_published:
        cmd.append("--use-published")
    subprocess.run(cmd, check=True)

    metric_tag = "published" if use_published else "best"
    summary_csv = out_dir / f"rppg_ecg_summary_{metric_tag}_{roi_mode}_{align_mode}.csv"
    return cmd, summary_csv


def _pct_change(py_v: float, ios_v: float, lower_better: bool) -> float:
    if not (math.isfinite(py_v) and math.isfinite(ios_v)):
        return math.nan
    denom = abs(py_v) if abs(py_v) > 1e-9 else math.nan
    if not math.isfinite(denom):
        return math.nan
    if lower_better:
        return (py_v - ios_v) / denom * 100.0
    return (ios_v - py_v) / denom * 100.0


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run real iOS Swift rPPG offline replay on the same videos, then compare against Python latest."
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--data-dir", default="/Users/liangwenwang/Downloads/Code/Demo2")
    parser.add_argument(
        "--ios-module-root",
        default="/Users/liangwenwang/Downloads/Code/Demo2/HealthApp-feature-Health-2026-02-27-v4.13/HealthApp/HealthApp",
    )
    parser.add_argument("--out-dir", default=str(repo_dir / "results" / "compare_python_ios_runtime"))
    parser.add_argument("--roi-mode", choices=["opencv", "mediapipe"], default="opencv")
    parser.add_argument("--split-file", default=str(repo_dir / "splits" / "fixed_holdout_6.txt"))
    parser.add_argument("--split-set", choices=["all", "train", "test"], default="test")
    parser.add_argument("--align-mode", choices=["timestamp", "index"], default="timestamp")
    parser.add_argument("--use-published", action="store_true")
    parser.add_argument("--max-seconds", type=int, default=-1, help="limit replay duration per video; -1 means full length")
    parser.add_argument("--skip-python-eval", action="store_true", help="only run iOS runtime replay + ECG compare")
    parser.add_argument(
        "--ios-preset",
        choices=["robustMode", "pythonAligned", "ios26Hybrid", "accuracyPriority", "v4Compatible"],
        default="robustMode",
    )
    parser.add_argument("--dump-roi-debug", action="store_true")
    parser.add_argument("--dump-algo-debug", action="store_true")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / run_id
    py_dir = run_dir / "python_latest"
    ios_dir = run_dir / "ios_runtime"
    ios_pred_dir = ios_dir / "pred_csv"
    ios_log_dir = ios_dir / "logs"
    ios_roi_debug_dir = ios_dir / "roi_debug"
    ios_algo_debug_dir = ios_dir / "algo_debug"
    ios_bin = ios_dir / "ios_offline_replay"

    data_dir = Path(args.data_dir)
    samples = discover_samples(data_dir)
    split_map = build_split_map(samples, split_file=args.split_file, holdout_list="")

    selected = []
    for s in samples:
        key = f"{s.group}/{s.stem}"
        split_name = split_map.get(key, "train")
        if args.split_set != "all" and split_name != args.split_set:
            continue
        selected.append((s, split_name))
    if not selected:
        raise RuntimeError("no samples selected by split settings")

    compile_cmd = _compile_ios_runner(repo_dir, Path(args.ios_module_root), ios_bin)

    ios_detail_rows: List[Dict[str, object]] = []
    ios_summary_rows: List[Dict[str, object]] = []
    replay_cmds: Dict[str, List[str]] = {}

    for sample, split_name in selected:
        token = f"{sample.group}/{sample.stem}"
        pred_csv = ios_pred_dir / f"{sample.group}_{sample.stem}.csv"
        log_path = ios_log_dir / f"{sample.group}_{sample.stem}.log"
        roi_debug_csv = (
            ios_roi_debug_dir / f"{sample.group}_{sample.stem}.csv" if args.dump_roi_debug else None
        )
        algo_debug_csv = (
            ios_algo_debug_dir / f"{sample.group}_{sample.stem}.csv" if args.dump_algo_debug else None
        )
        cmd = _run_ios_replay(
            bin_path=ios_bin,
            video_path=sample.video_path,
            out_csv=pred_csv,
            max_seconds=args.max_seconds,
            ios_preset=args.ios_preset,
            roi_debug_csv=roi_debug_csv,
            algo_debug_csv=algo_debug_csv,
            log_path=log_path,
        )
        replay_cmds[token] = cmd

        pred_rows = _load_ios_pred_rows(pred_csv, sample.group, sample.stem)
        ecg = load_ecg_series(sample.ecg_csv_path, align_mode=args.align_mode)
        compared = compare_to_ecg(
            pred_rows=pred_rows,
            ecg=ecg,
            use_published=args.use_published,
            metrics={"hr"},
        )
        for row in compared:
            row["split"] = split_name
            row["config_profile"] = "ios_runtime"
            row["video_path"] = str(sample.video_path)
            row["ecg_csv"] = str(sample.ecg_csv_path)
        ios_detail_rows.extend(compared)

        m = summarize_metric(compared, "ecg_hr", "est_hr")
        ios_summary_rows.append(
            {
                "split": split_name,
                "config_profile": "ios_runtime",
                "group": sample.group,
                "stem": sample.stem,
                "roi_mode_requested": "ios_runtime",
                "roi_mode_used": "ios_runtime",
                "video_path": str(sample.video_path),
                "ecg_csv": str(sample.ecg_csv_path),
                "hr_n": m["n"],
                "hr_mae": m["mae"],
                "hr_rmse": m["rmse"],
                "hr_mape": m["mape"],
                "hr_corr": m["corr"],
            }
        )

    ios_all = summarize_metric(ios_detail_rows, "ecg_hr", "est_hr")
    ios_summary_rows.append(
        {
            "split": "ALL",
            "config_profile": "ios_runtime",
            "group": "ALL",
            "stem": "ALL",
            "roi_mode_requested": "ios_runtime",
            "roi_mode_used": "ios_runtime",
            "video_path": "-",
            "ecg_csv": "-",
            "hr_n": ios_all["n"],
            "hr_mae": ios_all["mae"],
            "hr_rmse": ios_all["rmse"],
            "hr_mape": ios_all["mape"],
            "hr_corr": ios_all["corr"],
        }
    )

    ios_detail_csv = ios_dir / "rppg_ecg_comparison_ios_runtime_hr.csv"
    ios_summary_csv = ios_dir / "rppg_ecg_summary_ios_runtime_hr.csv"
    _write_csv(
        ios_detail_csv,
        ios_detail_rows,
        columns=[
            "split",
            "config_profile",
            "group",
            "stem",
            "sec",
            "ecg_hr",
            "est_hr",
            "rppg_hr",
            "hr_best",
            "hr_published",
            "error_hr",
            "abs_error_hr",
            "ape_percent_hr",
            "ecg_source",
            "ecg_hr_interpolated",
            "video_path",
            "ecg_csv",
        ],
    )
    _write_csv(
        ios_summary_csv,
        ios_summary_rows,
        columns=[
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
        ],
    )

    if args.skip_python_eval:
        run_meta = {
            "run_dir": str(run_dir),
            "ios_preset": args.ios_preset,
            "compile_cmd": compile_cmd,
            "ios_replay_cmds": replay_cmds,
            "ios_roi_debug_dir": str(ios_roi_debug_dir) if args.dump_roi_debug else "",
            "ios_algo_debug_dir": str(ios_algo_debug_dir) if args.dump_algo_debug else "",
            "ios_summary_csv": str(ios_summary_csv),
            "ios_detail_csv": str(ios_detail_csv),
            "python_skipped": True,
        }
        meta_path = run_dir / "run_meta.json"
        meta_path.write_text(json.dumps(run_meta, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"[OK] run_dir={run_dir}")
        print(f"[OK] ios_summary={ios_summary_csv}")
        print(
            f"[ALL][iOS] HR_MAE={ios_all['mae']:.4f} HR_RMSE={ios_all['rmse']:.4f} "
            f"HR_corr={ios_all['corr']:.4f} n={ios_all['n']:.0f}"
        )
        return

    py_cmd, py_summary_csv = _run_python_eval(
        python_exe=args.python_exe,
        repo_dir=repo_dir,
        data_dir=data_dir,
        out_dir=py_dir,
        roi_mode=args.roi_mode,
        split_file=args.split_file,
        split_set=args.split_set,
        align_mode=args.align_mode,
        use_published=args.use_published,
    )

    py_all_row = _load_all_row(py_summary_csv)
    ios_all_row = _load_all_row(ios_summary_csv)

    metrics_rows = []
    metric_specs = [
        ("hr_n", False),
        ("hr_mae", True),
        ("hr_rmse", True),
        ("hr_mape", True),
        ("hr_corr", False),
    ]
    for metric, lower_better in metric_specs:
        py_v = _to_float_nan(py_all_row.get(metric, ""))
        ios_v = _to_float_nan(ios_all_row.get(metric, ""))
        delta = _pct_change(py_v, ios_v, lower_better=lower_better)
        metrics_rows.append(
            {
                "metric": metric,
                "python_latest": py_v,
                "ios_runtime": ios_v,
                "change_percent_vs_python": delta,
                "direction": "lower_better" if lower_better else "higher_better",
            }
        )

    py_sample_rows = _load_sample_rows(py_summary_csv)
    ios_sample_rows = _load_sample_rows(ios_summary_csv)
    per_sample_rows: List[Dict[str, object]] = []
    for key in sorted(set(py_sample_rows.keys()) | set(ios_sample_rows.keys())):
        g, s = key
        py_r = py_sample_rows.get(key, {})
        ios_r = ios_sample_rows.get(key, {})
        py_mae = _to_float_nan(py_r.get("hr_mae", ""))
        ios_mae = _to_float_nan(ios_r.get("hr_mae", ""))
        py_corr = _to_float_nan(py_r.get("hr_corr", ""))
        ios_corr = _to_float_nan(ios_r.get("hr_corr", ""))
        mae_delta = _pct_change(py_mae, ios_mae, lower_better=True)
        corr_delta = _pct_change(py_corr, ios_corr, lower_better=False)
        per_sample_rows.append(
            {
                "group": g,
                "stem": s,
                "python_hr_mae": py_mae,
                "ios_hr_mae": ios_mae,
                "mae_change_percent_vs_python": mae_delta,
                "python_hr_corr": py_corr,
                "ios_hr_corr": ios_corr,
                "corr_change_percent_vs_python": corr_delta,
            }
        )

    metrics_csv = run_dir / "python_vs_ios_runtime_overall.csv"
    per_sample_csv = run_dir / "python_vs_ios_runtime_per_sample.csv"
    _write_csv(
        metrics_csv,
        metrics_rows,
        columns=[
            "metric",
            "direction",
            "python_latest",
            "ios_runtime",
            "change_percent_vs_python",
        ],
    )
    _write_csv(
        per_sample_csv,
        per_sample_rows,
        columns=[
            "group",
            "stem",
            "python_hr_mae",
            "ios_hr_mae",
            "mae_change_percent_vs_python",
            "python_hr_corr",
            "ios_hr_corr",
            "corr_change_percent_vs_python",
        ],
    )

    run_meta = {
        "run_dir": str(run_dir),
        "ios_preset": args.ios_preset,
        "compile_cmd": compile_cmd,
        "python_eval_cmd": py_cmd,
        "ios_replay_cmds": replay_cmds,
        "ios_roi_debug_dir": str(ios_roi_debug_dir) if args.dump_roi_debug else "",
        "ios_algo_debug_dir": str(ios_algo_debug_dir) if args.dump_algo_debug else "",
        "python_summary_csv": str(py_summary_csv),
        "ios_summary_csv": str(ios_summary_csv),
        "python_detail_csv": str(
            py_dir
            / f"rppg_ecg_comparison_{'published' if args.use_published else 'best'}_{args.roi_mode}_{args.align_mode}.csv"
        ),
        "ios_detail_csv": str(ios_detail_csv),
        "overall_comparison_csv": str(metrics_csv),
        "per_sample_comparison_csv": str(per_sample_csv),
    }
    meta_path = run_dir / "run_meta.json"
    meta_path.write_text(json.dumps(run_meta, ensure_ascii=True, indent=2), encoding="utf-8")

    def _show(metric: str) -> str:
        row = next((r for r in metrics_rows if r["metric"] == metric), None)
        if row is None:
            return "NA"
        py_v = row["python_latest"]
        ios_v = row["ios_runtime"]
        d = row["change_percent_vs_python"]
        if not (isinstance(py_v, float) and isinstance(ios_v, float)):
            return "NA"
        d_txt = "NA" if not (isinstance(d, float) and math.isfinite(d)) else f"{d:+.2f}%"
        return f"py={py_v:.4f}, ios={ios_v:.4f}, delta={d_txt}"

    print(f"[OK] run_dir={run_dir}")
    print(f"[OK] overall={metrics_csv}")
    print(f"[OK] per_sample={per_sample_csv}")
    print(f"[ALL] HR_MAE:  {_show('hr_mae')}")
    print(f"[ALL] HR_RMSE: {_show('hr_rmse')}")
    print(f"[ALL] HR_corr: {_show('hr_corr')}")


if __name__ == "__main__":
    main()
