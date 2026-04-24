#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


@dataclass
class Artifact:
    label: str
    src: Path
    dst_rel: Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _write_csv(path: Path, headers: List[str], rows: Iterable[List[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bundle strict rerun CSVs/logs and build traceable catalogs.")
    ap.add_argument(
        "--repo-root",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main",
    )
    ap.add_argument(
        "--bundle-dir",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/strict_20260424_bundle",
    )
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    bundle = Path(args.bundle_dir).resolve()
    bundle.mkdir(parents=True, exist_ok=True)

    data1 = repo / "results/strict_20260424_data1"
    data2 = repo / "results/strict_20260424_data2_noiphone13"
    analysis = repo / "results/strict_20260424_analysis"
    viz = repo / "results/visualizations/rppg_精度向上開発_報告書_6p_v1.html"
    log1 = repo / "results/strict_20260424_data1_stdout.log"
    log2 = repo / "results/strict_20260424_data2_noiphone13_stdout.log"

    artifacts = [
        Artifact("data1_summary_best", data1 / "rppg_ecg_summary_best_opencv_timestamp.csv", Path("data1/rppg_ecg_summary_best_opencv_timestamp.csv")),
        Artifact("data1_summary_published", data1 / "rppg_ecg_summary_published_opencv_timestamp.csv", Path("data1/rppg_ecg_summary_published_opencv_timestamp.csv")),
        Artifact("data1_detail_best", data1 / "rppg_ecg_comparison_best_opencv_timestamp.csv", Path("data1/rppg_ecg_comparison_best_opencv_timestamp.csv")),
        Artifact("data1_detail_published", data1 / "rppg_ecg_comparison_published_opencv_timestamp.csv", Path("data1/rppg_ecg_comparison_published_opencv_timestamp.csv")),
        Artifact("data2_summary_best", data2 / "rppg_ecg_summary_best_opencv_timestamp.csv", Path("data2_no_iphone13/rppg_ecg_summary_best_opencv_timestamp.csv")),
        Artifact("data2_summary_published", data2 / "rppg_ecg_summary_published_opencv_timestamp.csv", Path("data2_no_iphone13/rppg_ecg_summary_published_opencv_timestamp.csv")),
        Artifact("data2_detail_best", data2 / "rppg_ecg_comparison_best_opencv_timestamp.csv", Path("data2_no_iphone13/rppg_ecg_comparison_best_opencv_timestamp.csv")),
        Artifact("data2_detail_published", data2 / "rppg_ecg_comparison_published_opencv_timestamp.csv", Path("data2_no_iphone13/rppg_ecg_comparison_published_opencv_timestamp.csv")),
        Artifact("analysis_overall", analysis / "data1_data2_overall.csv", Path("analysis/data1_data2_overall.csv")),
        Artifact("analysis_per_group", analysis / "data1_data2_per_group.csv", Path("analysis/data1_data2_per_group.csv")),
        Artifact("analysis_per_sample", analysis / "data1_data2_per_sample.csv", Path("analysis/data1_data2_per_sample.csv")),
        Artifact("analysis_md", analysis / "data1_data2_analysis.md", Path("analysis/data1_data2_analysis.md")),
        Artifact("log_data1", log1, Path("logs/strict_20260424_data1_stdout.log")),
        Artifact("log_data2", log2, Path("logs/strict_20260424_data2_noiphone13_stdout.log")),
        Artifact("report_6p", viz, Path("visualizations/rppg_精度向上開発_報告書_6p_v1.html")),
    ]

    for a in artifacts:
        if not a.src.exists():
            raise RuntimeError(f"missing artifact: {a.src}")
        dst = bundle / a.dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a.src, dst)

    all_rows = []
    csv_rows = []
    for a in artifacts:
        dst = bundle / a.dst_rel
        size = dst.stat().st_size
        mtime = _iso(dst.stat().st_mtime)
        sha = _sha256(dst)
        line_cnt = _line_count(dst) if dst.suffix.lower() in {".csv", ".log", ".md", ".html"} else -1
        row = [a.label, str(a.dst_rel), size, line_cnt, mtime, sha]
        all_rows.append(row)
        if dst.suffix.lower() == ".csv":
            csv_rows.append(row)

    _write_csv(
        bundle / "artifact_catalog_20260424.csv",
        ["label", "bundle_path", "bytes", "line_count", "mtime_local", "sha256"],
        all_rows,
    )
    _write_csv(
        bundle / "csv_catalog_20260424.csv",
        ["label", "bundle_path", "bytes", "line_count", "mtime_local", "sha256"],
        csv_rows,
    )

    run_commands = """# Strict rerun commands (2026-04-24)

## Data2 (exclude iPhone13 Pro)
python evaluate_dataset.py \\
  --data-dir /Users/liangwenwang/Downloads/Code/Demo2/Data2_as_Data1 \\
  --out-dir /Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/strict_20260424_data2_noiphone13 \\
  --roi-mode opencv --align-mode timestamp --metrics hr --export-dual-channel \\
  --split-file /Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/splits/data2_no_iphone13_all.txt \\
  --split-set test
log: /Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/strict_20260424_data2_noiphone13_stdout.log

## Data1 (full rerun)
python -u evaluate_dataset.py \\
  --data-dir /Users/liangwenwang/Downloads/Code/Demo2/Data \\
  --out-dir /Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/strict_20260424_data1 \\
  --roi-mode opencv --align-mode timestamp --metrics hr --export-dual-channel \\
  2>&1 | tee /Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/strict_20260424_data1_stdout.log

## Consolidation + report expansion
python scripts/build_data1_data2_strict_summary.py
python scripts/expand_report_v4_with_data_pages.py
python scripts/organize_strict_outputs.py
"""
    (bundle / "run_commands_20260424.md").write_text(run_commands, encoding="utf-8")

    print(f"[OK] bundle dir: {bundle}")
    print(f"[OK] artifact catalog: {bundle / 'artifact_catalog_20260424.csv'}")
    print(f"[OK] csv catalog: {bundle / 'csv_catalog_20260424.csv'}")
    print(f"[OK] run commands: {bundle / 'run_commands_20260424.md'}")


if __name__ == "__main__":
    main()
