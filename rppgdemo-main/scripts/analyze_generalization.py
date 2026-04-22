#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DATA_PATH_RE = re.compile(r"/Data/([^/]+)/(?:video/)?([^/.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generalization analysis from rPPG-vs-ECG detail CSV.")
    parser.add_argument(
        "--detail-csv",
        default="results/regression_full_publish_fc088/rppg_ecg_comparison_published_opencv_timestamp.csv",
        help="Detail CSV from current configuration.",
    )
    parser.add_argument(
        "--baseline-csv",
        default="results/reproduce_20260420_full/rppg_ecg_comparison_best_opencv_timestamp.csv",
        help="Optional baseline detail CSV for comparison.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/generalization_fc088",
        help="Output directory for report and tables.",
    )
    return parser.parse_args()


def _sample_group_from_video_path(video_path: str) -> tuple[str, str]:
    raw = str(video_path)
    try:
        p = Path(raw)
        parts = list(p.parts)
        if "Data" in parts:
            i = parts.index("Data")
            if i + 1 < len(parts):
                group = parts[i + 1]
                stem = p.stem
                if group and stem:
                    return f"{group}/{stem}", group
    except Exception:
        pass

    match = DATA_PATH_RE.search(raw)
    if match:
        group = match.group(1)
        stem = match.group(2)
        return f"{group}/{stem}", group
    return "", ""


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["sec", "ecg_hr", "rppg_hr", "est_hr"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "rppg_hr" not in df.columns and "est_hr" in df.columns:
        df["rppg_hr"] = df["est_hr"]

    sample_group = df["video_path"].map(_sample_group_from_video_path)
    df["sample"] = sample_group.map(lambda x: x[0])
    df["group_raw"] = sample_group.map(lambda x: x[1])
    # Prefix with "g" to avoid numeric auto-casting collisions (e.g., 001 vs 1).
    df["group_id"] = df["group_raw"].map(lambda x: f"g{x}" if x else "")

    df = df.dropna(subset=["sample", "group_raw", "group_id", "ecg_hr", "rppg_hr"]).copy()
    df = df[df["sample"] != ""].copy()
    if "sec" in df.columns:
        df = df.dropna(subset=["sec"]).copy()
        df["sec"] = df["sec"].astype(int)
    return df


def _corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2:
        return math.nan
    return float(np.corrcoef(a.to_numpy(dtype=float), b.to_numpy(dtype=float))[0, 1])


def summarize_per_sample(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for sample, g in df.groupby("sample", sort=True):
        err = g["rppg_hr"] - g["ecg_hr"]
        ae = err.abs()
        rows.append(
            {
                "sample": sample,
                "group_id": g["group_id"].iloc[0],
                "group_raw": g["group_raw"].iloc[0],
                "n": int(len(g)),
                "mae": float(ae.mean()),
                "rmse": float(np.sqrt((err**2).mean())),
                "corr": _corr(g["ecg_hr"], g["rppg_hr"]),
                "bias": float(err.mean()),
                "p90_abs_error": float(ae.quantile(0.90)),
                "first_sec": float(g["sec"].min()) if "sec" in g.columns else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("mae", ascending=False).reset_index(drop=True)


def summarize_per_group(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for group_id, g in df.groupby("group_id", sort=True):
        err = g["rppg_hr"] - g["ecg_hr"]
        ae = err.abs()
        rows.append(
            {
                "group_id": group_id,
                "group_raw": g["group_raw"].iloc[0],
                "n": int(len(g)),
                "sample_count": int(g["sample"].nunique()),
                "mae": float(ae.mean()),
                "rmse": float(np.sqrt((err**2).mean())),
                "corr": _corr(g["ecg_hr"], g["rppg_hr"]),
                "bias": float(err.mean()),
                "p90_abs_error": float(ae.quantile(0.90)),
            }
        )
    return pd.DataFrame(rows).sort_values("mae", ascending=False).reset_index(drop=True)


def summarize_per_domain(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["domain"] = d["group_raw"].map(lambda g: "legacy_001_003" if g in {"001", "002", "003"} else "new_1_6")
    rows: List[Dict[str, float]] = []
    for domain, g in d.groupby("domain", sort=True):
        err = g["rppg_hr"] - g["ecg_hr"]
        ae = err.abs()
        rows.append(
            {
                "domain": domain,
                "n": int(len(g)),
                "sample_count": int(g["sample"].nunique()),
                "group_count": int(g["group_id"].nunique()),
                "mae": float(ae.mean()),
                "rmse": float(np.sqrt((err**2).mean())),
                "corr": _corr(g["ecg_hr"], g["rppg_hr"]),
                "bias": float(err.mean()),
                "p90_abs_error": float(ae.quantile(0.90)),
            }
        )
    return pd.DataFrame(rows).sort_values("domain").reset_index(drop=True)


def compare_vs_baseline(cur: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    cur_g = summarize_per_group(cur).rename(
        columns={
            "n": "n_cur",
            "sample_count": "sample_count_cur",
            "mae": "mae_cur",
            "rmse": "rmse_cur",
            "corr": "corr_cur",
            "bias": "bias_cur",
            "p90_abs_error": "p90_abs_error_cur",
        }
    )
    base_g = summarize_per_group(base).rename(
        columns={
            "n": "n_base",
            "sample_count": "sample_count_base",
            "mae": "mae_base",
            "rmse": "rmse_base",
            "corr": "corr_base",
            "bias": "bias_base",
            "p90_abs_error": "p90_abs_error_base",
        }
    )
    merged = cur_g.merge(base_g, on=["group_id", "group_raw"], how="outer")
    merged["coverage_ratio"] = merged["n_cur"] / merged["n_base"]
    merged["delta_mae"] = merged["mae_cur"] - merged["mae_base"]
    merged["delta_rmse"] = merged["rmse_cur"] - merged["rmse_base"]
    merged["delta_corr"] = merged["corr_cur"] - merged["corr_base"]
    return merged.sort_values("group_id").reset_index(drop=True)


def render_report(
    out_path: Path,
    cur_overall: Dict[str, float],
    base_overall: Dict[str, float],
    per_domain: pd.DataFrame,
    per_group: pd.DataFrame,
    compare_group: pd.DataFrame,
    worst_samples: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Generalization Report")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(
        f"- Current: n={int(cur_overall['n'])}, MAE={cur_overall['mae']:.3f}, RMSE={cur_overall['rmse']:.3f}, corr={cur_overall['corr']:.3f}"
    )
    lines.append(
        f"- Baseline: n={int(base_overall['n'])}, MAE={base_overall['mae']:.3f}, RMSE={base_overall['rmse']:.3f}, corr={base_overall['corr']:.3f}"
    )
    lines.append(
        f"- Delta: n={int(cur_overall['n']-base_overall['n'])}, MAE={cur_overall['mae']-base_overall['mae']:.3f}, RMSE={cur_overall['rmse']-base_overall['rmse']:.3f}, corr={cur_overall['corr']-base_overall['corr']:.3f}"
    )
    lines.append("")
    lines.append("## Domain Split (legacy 001-003 vs new 1-6)")
    lines.append("")
    lines.append(per_domain.to_markdown(index=False))
    lines.append("")
    lines.append("## Worst Groups by MAE")
    lines.append("")
    lines.append(per_group.head(6).to_markdown(index=False))
    lines.append("")
    lines.append("## Group Delta vs Baseline")
    lines.append("")
    lines.append(compare_group[["group_id", "group_raw", "n_base", "n_cur", "coverage_ratio", "mae_base", "mae_cur", "delta_mae"]].to_markdown(index=False))
    lines.append("")
    lines.append("## Worst Samples by MAE (current)")
    lines.append("")
    lines.append(worst_samples.to_markdown(index=False))
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def overall_stats(df: pd.DataFrame) -> Dict[str, float]:
    err = df["rppg_hr"] - df["ecg_hr"]
    return {
        "n": float(len(df)),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "corr": _corr(df["ecg_hr"], df["rppg_hr"]),
    }


def main() -> None:
    args = parse_args()
    detail_csv = Path(args.detail_csv)
    baseline_csv = Path(args.baseline_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cur_raw = pd.read_csv(detail_csv)
    base_raw = pd.read_csv(baseline_csv)
    cur = _prepare(cur_raw)
    base = _prepare(base_raw)

    per_sample = summarize_per_sample(cur)
    per_group = summarize_per_group(cur)
    per_domain = summarize_per_domain(cur)
    compare_group = compare_vs_baseline(cur, base)

    cur_overall = overall_stats(cur)
    base_overall = overall_stats(base)

    per_sample_path = out_dir / "generalization_per_sample.csv"
    per_group_path = out_dir / "generalization_per_group.csv"
    per_domain_path = out_dir / "generalization_per_domain.csv"
    compare_group_path = out_dir / "generalization_group_vs_baseline.csv"
    report_path = out_dir / "GENERALIZATION_REPORT.md"

    per_sample.to_csv(per_sample_path, index=False)
    per_group.to_csv(per_group_path, index=False)
    per_domain.to_csv(per_domain_path, index=False)
    compare_group.to_csv(compare_group_path, index=False)

    render_report(
        out_path=report_path,
        cur_overall=cur_overall,
        base_overall=base_overall,
        per_domain=per_domain,
        per_group=per_group,
        compare_group=compare_group,
        worst_samples=per_sample.head(10),
    )

    print(f"[OK] per-sample: {per_sample_path}")
    print(f"[OK] per-group: {per_group_path}")
    print(f"[OK] per-domain: {per_domain_path}")
    print(f"[OK] group-vs-baseline: {compare_group_path}")
    print(f"[OK] report: {report_path}")
    print("")
    print(
        f"[OVERALL] current n={int(cur_overall['n'])} mae={cur_overall['mae']:.3f} rmse={cur_overall['rmse']:.3f} corr={cur_overall['corr']:.3f}"
    )
    print(
        f"[OVERALL] baseline n={int(base_overall['n'])} mae={base_overall['mae']:.3f} rmse={base_overall['rmse']:.3f} corr={base_overall['corr']:.3f}"
    )


if __name__ == "__main__":
    main()
