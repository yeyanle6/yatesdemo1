#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DATA_PATH_RE = re.compile(r"/Data/([^/]+)/video/([^/.]+)")


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid sweep for publish-gate policy: freq_conf threshold + warmup sec + min qualified points."
    )
    parser.add_argument(
        "--quick9-csv",
        default="results/mae_target_quick9_baseline/rppg_ecg_comparison_best_opencv_timestamp.csv",
        help="Quick dataset per-second comparison CSV.",
    )
    parser.add_argument(
        "--full-csv",
        default="results/reproduce_20260420_full/rppg_ecg_comparison_best_opencv_timestamp.csv",
        help="Full dataset per-second comparison CSV.",
    )
    parser.add_argument(
        "--fc-thresholds",
        default="0.88,0.90,0.92",
        help="Comma-separated frequency confidence thresholds.",
    )
    parser.add_argument(
        "--min-secs",
        default="10,12,15",
        help="Comma-separated minimum warmup seconds.",
    )
    parser.add_argument(
        "--k-values",
        default="3,5,7",
        help="Comma-separated required count of qualified points.",
    )
    parser.add_argument(
        "--out-grid",
        default="results/parameter_compare_publish_gate_grid_quick9_full.csv",
        help="Output CSV for full grid metrics.",
    )
    parser.add_argument(
        "--out-top",
        default="results/parameter_compare_publish_gate_grid_top.csv",
        help="Output CSV for top candidates selected on quick9 and projected to full35.",
    )
    parser.add_argument(
        "--quick9-min-coverage",
        type=float,
        default=0.25,
        help="Candidate filter on quick9 coverage.",
    )
    parser.add_argument(
        "--quick9-min-sample-ratio",
        type=float,
        default=0.75,
        help="Candidate filter on quick9 sample output ratio.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="How many quick9 candidates to keep before projection to full35.",
    )
    return parser.parse_args()


def sample_from_row(row: pd.Series) -> str:
    video_path = str(row.get("video_path", "") or "")
    match = DATA_PATH_RE.search(video_path)
    if match:
        return f"{match.group(1)}/{match.group(2)}"

    group = str(row.get("group", "") or "").strip()
    stem = str(row.get("stem", "") or "").strip()
    if group and stem:
        return f"{group}/{stem}"
    return ""


def load_comparison_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["sec", "ecg_hr", "hr_best", "frequency_confidence"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sample"] = df.apply(sample_from_row, axis=1)
    df = df.dropna(subset=["sample", "sec", "ecg_hr", "hr_best", "frequency_confidence"]).copy()
    df = df[df["sample"] != ""].copy()
    df["sec"] = df["sec"].astype(int)
    return df.sort_values(["sample", "sec"]).reset_index(drop=True)


def evaluate_policy(df: pd.DataFrame, fc_thr: float, min_sec: int, k: int) -> Dict[str, float]:
    d = df.copy()
    qualified = (d["frequency_confidence"] >= fc_thr) & (d["sec"] >= min_sec)
    d["qualified"] = qualified.astype(int)
    d["qualified_cum"] = d.groupby("sample")["qualified"].cumsum()

    accepted_mask = qualified & (d["qualified_cum"] >= k)
    accepted = d[accepted_mask]

    total_n = len(d)
    used_n = len(accepted)
    total_samples = d["sample"].nunique()
    first_secs = accepted.groupby("sample")["sec"].min() if used_n > 0 else pd.Series(dtype=float)
    with_output = len(first_secs)

    if used_n > 0:
        err = accepted["hr_best"] - accepted["ecg_hr"]
        mae = float(err.abs().mean())
        rmse = float(np.sqrt((err**2).mean()))
        bias = float(err.mean())
    else:
        mae = math.nan
        rmse = math.nan
        bias = math.nan

    def q(v: pd.Series, p: float) -> float:
        return float(v.quantile(p)) if len(v) > 0 else math.nan

    return {
        "fc_thr": float(fc_thr),
        "min_sec": int(min_sec),
        "k": int(k),
        "n": int(used_n),
        "coverage": float(used_n / total_n) if total_n > 0 else math.nan,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "samples_with_output": int(with_output),
        "samples_without_output": int(total_samples - with_output),
        "sample_output_ratio": float(with_output / total_samples) if total_samples > 0 else math.nan,
        "first_sec_q25": q(first_secs, 0.25),
        "first_sec_q50": q(first_secs, 0.50),
        "first_sec_q75": q(first_secs, 0.75),
    }


def run_grid(
    df: pd.DataFrame,
    dataset_name: str,
    fc_thresholds: Iterable[float],
    min_secs: Iterable[int],
    k_values: Iterable[int],
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for fc_thr in fc_thresholds:
        for min_sec in min_secs:
            for k in k_values:
                record = evaluate_policy(df, fc_thr=fc_thr, min_sec=min_sec, k=k)
                record["dataset"] = dataset_name
                rows.append(record)
    return pd.DataFrame.from_records(rows)


def choose_top_candidates(
    quick_df: pd.DataFrame,
    full_df: pd.DataFrame,
    min_cov: float,
    min_ratio: float,
    top_k: int,
) -> pd.DataFrame:
    keys = ["fc_thr", "min_sec", "k"]
    q = quick_df[(quick_df["coverage"] >= min_cov) & (quick_df["sample_output_ratio"] >= min_ratio)].copy()
    q = q.sort_values(["mae", "coverage", "first_sec_q50"], ascending=[True, False, True]).head(top_k)

    q_small = q[keys + ["mae", "coverage", "rmse", "sample_output_ratio", "first_sec_q50"]].rename(
        columns={
            "mae": "quick9_mae",
            "coverage": "quick9_cov",
            "rmse": "quick9_rmse",
            "sample_output_ratio": "quick9_sample_ratio",
            "first_sec_q50": "quick9_first_q50",
        }
    )
    f_small = full_df[keys + ["mae", "coverage", "rmse", "sample_output_ratio", "first_sec_q50"]].rename(
        columns={
            "mae": "full35_mae",
            "coverage": "full35_cov",
            "rmse": "full35_rmse",
            "sample_output_ratio": "full35_sample_ratio",
            "first_sec_q50": "full35_first_q50",
        }
    )
    out = q_small.merge(f_small, on=keys, how="left")
    return out.sort_values(["full35_mae", "full35_cov"], ascending=[True, False]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    root = Path(".")

    quick_csv = root / args.quick9_csv
    full_csv = root / args.full_csv
    out_grid = root / args.out_grid
    out_top = root / args.out_top

    fc_thresholds = parse_float_list(args.fc_thresholds)
    min_secs = parse_int_list(args.min_secs)
    k_values = parse_int_list(args.k_values)

    quick_df = load_comparison_csv(quick_csv)
    full_df = load_comparison_csv(full_csv)

    quick_grid = run_grid(quick_df, "quick9", fc_thresholds, min_secs, k_values)
    full_grid = run_grid(full_df, "full35", fc_thresholds, min_secs, k_values)
    all_grid = pd.concat([quick_grid, full_grid], ignore_index=True)

    out_grid.parent.mkdir(parents=True, exist_ok=True)
    out_top.parent.mkdir(parents=True, exist_ok=True)
    all_grid.to_csv(out_grid, index=False)

    top = choose_top_candidates(
        quick_df=quick_grid,
        full_df=full_grid,
        min_cov=args.quick9_min_coverage,
        min_ratio=args.quick9_min_sample_ratio,
        top_k=args.top_k,
    )
    top.to_csv(out_top, index=False)

    print(f"[OK] grid metrics: {out_grid}")
    print(f"[OK] top candidates: {out_top}")
    print("\nTop candidates:")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
