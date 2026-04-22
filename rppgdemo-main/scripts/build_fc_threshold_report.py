#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS = [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94]
WARMUP_MIN_SECS = [5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 60]
WARMUP_THRESHOLDS = [0.80, 0.85, 0.88, 0.90, 0.92]
TIME_TO_K_THRESHOLDS = [0.88, 0.90, 0.92]
TIME_TO_K_KS = [1, 3, 5, 10]
DATA_PATH_RE = re.compile(r"/Data/([^/]+)/video/([^/.]+)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build HTML report for freq_conf threshold tradeoff with per-file HR comparisons."
    )
    parser.add_argument(
        "--comparison-csv",
        default="results/reproduce_20260420_full/rppg_ecg_comparison_best_opencv_timestamp.csv",
        help="Input per-second comparison CSV.",
    )
    parser.add_argument(
        "--matrix-csv-out",
        default="results/parameter_compare_per_dataset_fc_matrix_080_094.csv",
        help="Output per-sample matrix (wide format).",
    )
    parser.add_argument(
        "--matrix-long-csv-out",
        default="results/parameter_compare_per_dataset_fc_matrix_080_094_long.csv",
        help="Output per-sample matrix (long format).",
    )
    parser.add_argument(
        "--per-file-hr-csv-out",
        default="results/parameter_compare_per_file_hr_values_long.csv",
        help="Output per-file HR comparison (long format).",
    )
    parser.add_argument(
        "--warmup-csv-out",
        default="results/parameter_compare_warmup_time_fc.csv",
        help="Output warmup time-vs-accuracy matrix.",
    )
    parser.add_argument(
        "--time-to-k-csv-out",
        default="results/parameter_compare_time_to_k_fc.csv",
        help="Output time-to-k high-confidence points distribution.",
    )
    parser.add_argument(
        "--html-out",
        default="results/visualizations/fc_threshold_report.html",
        help="Output HTML report path.",
    )
    parser.add_argument(
        "--generalization-domain-csv",
        default="results/generalization_fc088/generalization_per_domain.csv",
        help="Optional generalization domain summary CSV.",
    )
    parser.add_argument(
        "--generalization-group-csv",
        default="results/generalization_fc088/generalization_per_group.csv",
        help="Optional generalization group summary CSV.",
    )
    parser.add_argument(
        "--generalization-delta-csv",
        default="results/generalization_fc088/generalization_group_vs_baseline.csv",
        help="Optional generalization group-vs-baseline delta CSV.",
    )
    parser.add_argument(
        "--generalization-sample-csv",
        default="results/generalization_fc088/generalization_per_sample.csv",
        help="Optional generalization per-sample summary CSV.",
    )
    parser.add_argument(
        "--generalization-report-md",
        default="results/generalization_fc088/GENERALIZATION_REPORT.md",
        help="Optional generalization report markdown with overall current/baseline metrics.",
    )
    parser.add_argument(
        "--lag-curve-baseline-csv",
        default="results/lag_analysis/lag_curve_baseline_best.csv",
        help="Optional lag curve CSV for baseline.",
    )
    parser.add_argument(
        "--lag-curve-current-csv",
        default="results/lag_analysis/lag_curve_published_fc088.csv",
        help="Optional lag curve CSV for current/published setting.",
    )
    parser.add_argument(
        "--lag-sample-baseline-csv",
        default="results/lag_analysis/lag_per_sample_baseline_best.csv",
        help="Optional per-sample lag gain CSV for baseline.",
    )
    parser.add_argument(
        "--lag-sample-current-csv",
        default="results/lag_analysis/lag_per_sample_published_fc088.csv",
        help="Optional per-sample lag gain CSV for current/published setting.",
    )
    parser.add_argument(
        "--lag-commonwindow-sample-csv",
        default="results/lag_analysis/lag_commonwindow_per_sample_baseline_best.csv",
        help="Optional fixed-window lag comparison CSV (baseline).",
    )
    parser.add_argument(
        "--lag-diffcorr-sample-csv",
        default="results/lag_analysis/lag_diffcorr_per_sample_baseline_best.csv",
        help="Optional diff-correlation lag summary CSV (baseline).",
    )
    parser.add_argument(
        "--lag-global-diff-csv",
        default="results/lag_analysis/lag_global_diff_curve_baseline_best.csv",
        help="Optional global diff-correlation lag curve CSV (baseline).",
    )
    parser.add_argument(
        "--thresholds",
        default="0.80,0.82,0.84,0.86,0.88,0.90,0.92,0.94",
        help="Comma-separated freq_conf thresholds.",
    )
    return parser.parse_args()


def _sample_from_row(row: pd.Series) -> str:
    video_path = str(row.get("video_path", "") or "")
    match = DATA_PATH_RE.search(video_path)
    if match:
        return f"{match.group(1)}/{match.group(2)}"

    group = str(row.get("group", "") or "").strip()
    stem = str(row.get("stem", "") or "").strip()
    if group and stem:
        return f"{group}/{stem}"
    return ""


def _format_float(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def _format_pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{100.0 * value:.1f}%"


def _compute_stats_table(
    df: pd.DataFrame, thresholds: List[float]
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for sample, g in df.groupby("sample", sort=True):
        g_sorted = g.sort_values("sec")
        n_all = len(g_sorted)
        masks = [("all", np.ones(n_all, dtype=bool))]
        masks.extend([(f"{thr:.2f}", g_sorted["frequency_confidence"].to_numpy() >= thr) for thr in thresholds])

        for threshold_key, mask in masks:
            g_sel = g_sorted.loc[mask]
            n = len(g_sel)
            coverage = float(n) / float(n_all) if n_all > 0 else math.nan
            if n > 0:
                diff = g_sel["hr_best"] - g_sel["ecg_hr"]
                ecg_mean = float(g_sel["ecg_hr"].mean())
                rppg_mean = float(g_sel["hr_best"].mean())
                bias = float(diff.mean())
                mae = float(diff.abs().mean())
                rmse = float(np.sqrt((diff ** 2).mean()))
            else:
                ecg_mean = math.nan
                rppg_mean = math.nan
                bias = math.nan
                mae = math.nan
                rmse = math.nan

            records.append(
                {
                    "sample": sample,
                    "threshold": threshold_key,
                    "n_all": int(n_all),
                    "n": int(n),
                    "coverage": coverage,
                    "ecg_hr_mean": ecg_mean,
                    "rppg_hr_mean": rppg_mean,
                    "bias_hr": bias,
                    "mae": mae,
                    "rmse": rmse,
                }
            )

    return pd.DataFrame.from_records(records)


def _build_wide_matrix(
    stats_long: pd.DataFrame, thresholds: List[float]
) -> pd.DataFrame:
    base = (
        stats_long[stats_long["threshold"] == "all"][["sample", "n_all", "mae"]]
        .rename(columns={"mae": "mae_all"})
        .copy()
    )

    out = base
    for thr in thresholds:
        key = f"{thr:.2f}"
        sub = stats_long[stats_long["threshold"] == key][
            ["sample", "n", "coverage", "mae"]
        ].rename(
            columns={
                "n": f"n_fc_ge_{key}",
                "coverage": f"cov_fc_ge_{key}",
                "mae": f"mae_fc_ge_{key}",
            }
        )
        out = out.merge(sub, on="sample", how="left")
        out[f"delta_mae_fc_ge_{key}"] = out[f"mae_fc_ge_{key}"] - out["mae_all"]

    out = out.sort_values("mae_all", ascending=False).reset_index(drop=True)
    return out


def _compute_overall_rows(
    df: pd.DataFrame, thresholds: List[float]
) -> List[Dict[str, float]]:
    total_n = len(df)
    rows: List[Dict[str, float]] = []
    for thr in thresholds:
        sel = df[df["frequency_confidence"] >= thr]
        n = len(sel)
        coverage = float(n) / float(total_n) if total_n > 0 else math.nan
        if n > 0:
            diff = sel["hr_best"] - sel["ecg_hr"]
            mae = float(diff.abs().mean())
            rmse = float(np.sqrt((diff ** 2).mean()))
            bias = float(diff.mean())
        else:
            mae = math.nan
            rmse = math.nan
            bias = math.nan
        rows.append(
            {
                "threshold": thr,
                "n": n,
                "coverage": coverage,
                "mae": mae,
                "rmse": rmse,
                "bias": bias,
            }
        )
    return rows


def _compute_warmup_rows(
    df: pd.DataFrame, min_secs: List[int], thresholds: List[float]
) -> List[Dict[str, float]]:
    total_n = len(df)
    rows: List[Dict[str, float]] = []
    for min_sec in min_secs:
        dft = df[df["sec"] >= min_sec]
        for thr in thresholds:
            sel = dft[dft["frequency_confidence"] >= thr]
            n = len(sel)
            coverage = float(n) / float(total_n) if total_n > 0 else math.nan
            if n > 0:
                diff = sel["hr_best"] - sel["ecg_hr"]
                mae = float(diff.abs().mean())
                rmse = float(np.sqrt((diff ** 2).mean()))
            else:
                mae = math.nan
                rmse = math.nan
            rows.append(
                {
                    "min_sec": int(min_sec),
                    "threshold": float(thr),
                    "n": int(n),
                    "coverage": coverage,
                    "mae": mae,
                    "rmse": rmse,
                }
            )
    return rows


def _compute_time_to_k_rows(
    df: pd.DataFrame, thresholds: List[float], ks: List[int]
) -> List[Dict[str, float]]:
    all_samples = sorted(df["sample"].unique().tolist())
    total_samples = len(all_samples)
    quantiles = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
    rows: List[Dict[str, float]] = []

    for thr in thresholds:
        dthr = df[df["frequency_confidence"] >= thr]
        sample_secs: Dict[str, List[int]] = {}
        for sample, g in dthr.groupby("sample", sort=True):
            secs = sorted(g["sec"].astype(int).tolist())
            sample_secs[sample] = secs

        for k in ks:
            vals: List[float] = []
            for sample in all_samples:
                secs = sample_secs.get(sample, [])
                if len(secs) >= k:
                    vals.append(float(secs[k - 1]))

            have = len(vals)
            missing = total_samples - have
            if have > 0:
                qv = np.quantile(vals, quantiles)
            else:
                qv = np.array([math.nan] * len(quantiles), dtype=float)

            rows.append(
                {
                    "threshold": float(thr),
                    "k": int(k),
                    "have": int(have),
                    "missing": int(missing),
                    "q0": float(qv[0]) if np.isfinite(qv[0]) else math.nan,
                    "q25": float(qv[1]) if np.isfinite(qv[1]) else math.nan,
                    "q50": float(qv[2]) if np.isfinite(qv[2]) else math.nan,
                    "q75": float(qv[3]) if np.isfinite(qv[3]) else math.nan,
                    "q90": float(qv[4]) if np.isfinite(qv[4]) else math.nan,
                    "q100": float(qv[5]) if np.isfinite(qv[5]) else math.nan,
                }
            )
    return rows


def _build_sample_stats_json(
    stats_long: pd.DataFrame, thresholds: List[float]
) -> str:
    pivot: Dict[str, Dict[str, object]] = {}
    for row in stats_long.itertuples(index=False):
        sample_item = pivot.setdefault(
            row.sample,
            {
                "sample": row.sample,
                "n_all": int(row.n_all),
                "stats": {},
            },
        )
        sample_item["stats"][str(row.threshold)] = {
            "n": int(row.n),
            "coverage": float(row.coverage) if np.isfinite(row.coverage) else None,
            "ecg_mean": float(row.ecg_hr_mean) if np.isfinite(row.ecg_hr_mean) else None,
            "rppg_mean": float(row.rppg_hr_mean) if np.isfinite(row.rppg_hr_mean) else None,
            "bias": float(row.bias_hr) if np.isfinite(row.bias_hr) else None,
            "mae": float(row.mae) if np.isfinite(row.mae) else None,
            "rmse": float(row.rmse) if np.isfinite(row.rmse) else None,
        }

    sorted_samples = sorted(
        pivot.values(),
        key=lambda x: (
            -(x["stats"].get("all", {}).get("mae") if x["stats"].get("all", {}).get("mae") is not None else -1e9),
            x["sample"],
        ),
    )
    return json.dumps(sorted_samples, ensure_ascii=False, separators=(",", ":"))


def _build_series_json(df: pd.DataFrame) -> str:
    payload: Dict[str, Dict[str, List[float]]] = {}
    for sample, g in df.groupby("sample", sort=True):
        g_sorted = g.sort_values("sec")
        payload[sample] = {
            "sec": [int(v) for v in g_sorted["sec"].tolist()],
            "ecg": [round(float(v), 3) for v in g_sorted["ecg_hr"].tolist()],
            "rppg": [round(float(v), 3) for v in g_sorted["hr_best"].tolist()],
            "fc": [round(float(v), 4) for v in g_sorted["frequency_confidence"].tolist()],
        }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _to_json_scalar(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    if pd.isna(value):
        return None
    return str(value)


def _records_from_df(
    df: pd.DataFrame,
    columns: List[str] | None = None,
    sort_by: str | None = None,
    ascending: bool = True,
    limit: int | None = None,
) -> List[Dict[str, object]]:
    if df.empty:
        return []
    out = df.copy()
    if columns is not None:
        keep = [c for c in columns if c in out.columns]
        out = out[keep]
    if sort_by is not None and sort_by in out.columns:
        out = out.sort_values(sort_by, ascending=ascending)
    if limit is not None:
        out = out.head(limit)

    records: List[Dict[str, object]] = []
    for _, row in out.iterrows():
        rec: Dict[str, object] = {}
        for col in out.columns:
            rec[col] = _to_json_scalar(row[col])
        records.append(rec)
    return records


def _weighted_avg(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
    if value_col not in df.columns or weight_col not in df.columns:
        return math.nan
    tmp = df[[value_col, weight_col]].copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp[weight_col] = pd.to_numeric(tmp[weight_col], errors="coerce")
    tmp = tmp.dropna()
    tmp = tmp[tmp[weight_col] > 0]
    if tmp.empty:
        return math.nan
    w = tmp[weight_col].to_numpy(dtype=float)
    v = tmp[value_col].to_numpy(dtype=float)
    return float(np.sum(v * w) / np.sum(w))


def _weighted_rmse_from_group(df: pd.DataFrame, rmse_col: str, n_col: str) -> float:
    if rmse_col not in df.columns or n_col not in df.columns:
        return math.nan
    tmp = df[[rmse_col, n_col]].copy()
    tmp[rmse_col] = pd.to_numeric(tmp[rmse_col], errors="coerce")
    tmp[n_col] = pd.to_numeric(tmp[n_col], errors="coerce")
    tmp = tmp.dropna()
    tmp = tmp[tmp[n_col] > 0]
    if tmp.empty:
        return math.nan
    n = tmp[n_col].to_numpy(dtype=float)
    rmse = tmp[rmse_col].to_numpy(dtype=float)
    return float(np.sqrt(np.sum(n * (rmse ** 2)) / np.sum(n)))


def _parse_generalization_overview(report_md: Path) -> Dict[str, object]:
    if not report_md.exists():
        return {}
    try:
        text = report_md.read_text(encoding="utf-8")
    except Exception:
        return {}

    pattern = re.compile(
        r"-\s*(Current|Baseline):\s*n=(\d+),\s*MAE=([\-0-9.]+),\s*RMSE=([\-0-9.]+),\s*corr=([\-0-9.]+)",
        re.IGNORECASE,
    )
    found: Dict[str, Dict[str, float]] = {}
    for m in pattern.finditer(text):
        key = m.group(1).lower()
        found[key] = {
            "n": float(m.group(2)),
            "mae": float(m.group(3)),
            "rmse": float(m.group(4)),
            "corr": float(m.group(5)),
        }

    current = found.get("current")
    baseline = found.get("baseline")
    if not current or not baseline:
        return {}

    return {
        "n_cur": int(current["n"]),
        "n_base": int(baseline["n"]),
        "mae_cur": current["mae"],
        "mae_base": baseline["mae"],
        "rmse_cur": current["rmse"],
        "rmse_base": baseline["rmse"],
        "corr_cur": current["corr"],
        "corr_base": baseline["corr"],
        "delta_mae": current["mae"] - baseline["mae"],
        "delta_rmse": current["rmse"] - baseline["rmse"],
        "delta_corr": current["corr"] - baseline["corr"],
    }


def _build_generalization_json(
    domain_df: pd.DataFrame,
    group_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    report_md: Path | None = None,
) -> str:
    overview = _parse_generalization_overview(report_md) if report_md is not None else {}
    if not overview and not delta_df.empty:
        n_cur = int(pd.to_numeric(delta_df.get("n_cur"), errors="coerce").fillna(0).sum())
        n_base = int(pd.to_numeric(delta_df.get("n_base"), errors="coerce").fillna(0).sum())
        mae_cur = _weighted_avg(delta_df, "mae_cur", "n_cur")
        mae_base = _weighted_avg(delta_df, "mae_base", "n_base")
        rmse_cur = _weighted_rmse_from_group(delta_df, "rmse_cur", "n_cur")
        rmse_base = _weighted_rmse_from_group(delta_df, "rmse_base", "n_base")
        corr_cur = _weighted_avg(delta_df, "corr_cur", "n_cur")
        corr_base = _weighted_avg(delta_df, "corr_base", "n_base")
        overview = {
            "n_cur": n_cur,
            "n_base": n_base,
            "mae_cur": _to_json_scalar(mae_cur),
            "mae_base": _to_json_scalar(mae_base),
            "rmse_cur": _to_json_scalar(rmse_cur),
            "rmse_base": _to_json_scalar(rmse_base),
            "corr_cur": _to_json_scalar(corr_cur),
            "corr_base": _to_json_scalar(corr_base),
            "delta_mae": _to_json_scalar(mae_cur - mae_base if np.isfinite(mae_cur) and np.isfinite(mae_base) else math.nan),
            "delta_rmse": _to_json_scalar(
                rmse_cur - rmse_base if np.isfinite(rmse_cur) and np.isfinite(rmse_base) else math.nan
            ),
            "delta_corr": _to_json_scalar(
                corr_cur - corr_base if np.isfinite(corr_cur) and np.isfinite(corr_base) else math.nan
            ),
        }

    payload = {
        "overview": overview,
        "domain": _records_from_df(
            domain_df,
            columns=[
                "domain",
                "n",
                "sample_count",
                "group_count",
                "mae",
                "rmse",
                "corr",
                "bias",
                "p90_abs_error",
            ],
            sort_by="mae",
            ascending=False,
        ),
        "group": _records_from_df(
            group_df,
            columns=[
                "group_id",
                "group_raw",
                "n",
                "sample_count",
                "mae",
                "rmse",
                "corr",
                "bias",
                "p90_abs_error",
            ],
            sort_by="mae",
            ascending=False,
        ),
        "delta": _records_from_df(
            delta_df,
            columns=[
                "group_id",
                "group_raw",
                "n_base",
                "n_cur",
                "coverage_ratio",
                "mae_base",
                "mae_cur",
                "delta_mae",
                "rmse_base",
                "rmse_cur",
                "delta_rmse",
                "corr_base",
                "corr_cur",
                "delta_corr",
            ],
            sort_by="delta_mae",
            ascending=True,
        ),
        "samples": _records_from_df(
            sample_df,
            columns=[
                "sample",
                "group_id",
                "group_raw",
                "n",
                "mae",
                "rmse",
                "corr",
                "bias",
                "p90_abs_error",
                "first_sec",
            ],
            sort_by="mae",
            ascending=False,
            limit=30,
        ),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _build_lag_json(
    curve_baseline_df: pd.DataFrame,
    curve_current_df: pd.DataFrame,
    sample_baseline_df: pd.DataFrame,
    sample_current_df: pd.DataFrame,
    commonwindow_df: pd.DataFrame,
    diffcorr_df: pd.DataFrame,
    global_diff_df: pd.DataFrame,
) -> str:
    payload = {
        "curve_baseline": _records_from_df(
            curve_baseline_df,
            columns=["lag", "mae", "rmse", "n"],
            sort_by="lag",
            ascending=True,
        ),
        "curve_current": _records_from_df(
            curve_current_df,
            columns=["lag", "mae", "rmse", "n"],
            sort_by="lag",
            ascending=True,
        ),
        "sample_baseline": _records_from_df(
            sample_baseline_df,
            columns=["sample", "n0", "mae_lag0", "best_lag", "best_mae", "best_n", "mae_gain", "gain_ratio"],
            sort_by="mae_gain",
            ascending=False,
            limit=30,
        ),
        "sample_current": _records_from_df(
            sample_current_df,
            columns=["sample", "n0", "mae_lag0", "best_lag", "best_mae", "best_n", "mae_gain", "gain_ratio"],
            sort_by="mae_gain",
            ascending=False,
            limit=30,
        ),
        "commonwindow": _records_from_df(
            commonwindow_df,
            columns=[
                "sample",
                "n_common",
                "best_lag_mae",
                "best_mae",
                "lag0_mae",
                "mae_gain",
                "best_lag_corr",
                "best_corr",
                "lag0_corr",
                "corr_gain",
            ],
            sort_by="mae_gain",
            ascending=False,
            limit=30,
        ),
        "diffcorr": _records_from_df(
            diffcorr_df,
            columns=[
                "sample",
                "n",
                "lag0_diff_mae",
                "lag0_diff_corr",
                "best_lag_by_diff_corr",
                "best_diff_mae",
                "best_diff_corr",
                "corr_gain",
            ],
            sort_by="corr_gain",
            ascending=False,
            limit=30,
        ),
        "global_diff_curve": _records_from_df(
            global_diff_df,
            columns=["lag", "diff_mae", "diff_corr", "n"],
            sort_by="lag",
            ascending=True,
        ),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _render_html(
    overall_rows: List[Dict[str, float]],
    matrix_wide: pd.DataFrame,
    thresholds: List[float],
    warmup_rows: List[Dict[str, float]],
    time_to_k_rows: List[Dict[str, float]],
    warmup_thresholds: List[float],
    sample_stats_json: str,
    series_json: str,
    generalization_json: str,
    lag_json: str,
) -> str:
    overall_html_rows: List[str] = []
    for row in overall_rows:
        badge = ' <span class="badge">MAE&lt;3</span>' if np.isfinite(row["mae"]) and row["mae"] < 3.0 else ""
        overall_html_rows.append(
            "<tr>"
            f"<td>{row['threshold']:.2f}{badge}</td>"
            f"<td>{int(row['n'])}</td>"
            f"<td>{_format_pct(row['coverage'])}</td>"
            f"<td>{_format_float(row['mae'])}</td>"
            f"<td>{_format_float(row['rmse'])}</td>"
            f"<td>{_format_float(row['bias'])}</td>"
            "</tr>"
        )

    threshold_cells = "\n".join(
        [
            f"<th>n@{thr:.2f}</th><th>cov@{thr:.2f}</th><th>mae@{thr:.2f}</th>"
            for thr in thresholds
        ]
    )

    matrix_rows: List[str] = []
    for _, row in matrix_wide.iterrows():
        cells = [
            f"<td>{html.escape(str(row['sample']))}</td>",
            f"<td>{int(row['n_all'])}</td>",
            f"<td>{_format_float(float(row['mae_all']))}</td>",
        ]
        for thr in thresholds:
            key = f"{thr:.2f}"
            n_col = f"n_fc_ge_{key}"
            c_col = f"cov_fc_ge_{key}"
            m_col = f"mae_fc_ge_{key}"
            n_val = row[n_col]
            c_val = row[c_col]
            m_val = row[m_col]
            cells.extend(
                [
                    f"<td>{'' if pd.isna(n_val) else int(n_val)}</td>",
                    f"<td>{_format_pct(float(c_val)) if pd.notna(c_val) else ''}</td>",
                    f"<td>{_format_float(float(m_val)) if pd.notna(m_val) else ''}</td>",
                ]
            )
        matrix_rows.append("<tr>" + "".join(cells) + "</tr>")

    threshold_options = '<option value="all">all</option>' + "".join(
        [f'<option value="{thr:.2f}">{thr:.2f}</option>' for thr in thresholds]
    )
    sample_options = (
        '<option value="">请选择 sample...</option>'
        + "".join(
            [f'<option value="{html.escape(str(v))}">{html.escape(str(v))}</option>' for v in matrix_wide["sample"].tolist()]
        )
    )
    threshold_js = json.dumps([f"{thr:.2f}" for thr in thresholds], ensure_ascii=False)
    warmup_json = json.dumps(warmup_rows, ensure_ascii=False, separators=(",", ":"))
    time_to_k_json = json.dumps(time_to_k_rows, ensure_ascii=False, separators=(",", ":"))
    warmup_threshold_js = json.dumps([f"{thr:.2f}" for thr in warmup_thresholds], ensure_ascii=False)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>rPPG 参数阈值与逐文件 HR 对比报告</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px; color:#1f2937; }}
h1,h2 {{ margin: 0 0 12px; }}
.section {{ margin: 20px 0 28px; }}
.muted {{ color:#6b7280; font-size: 13px; }}
.grid {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(240px,1fr)); gap:12px; margin: 10px 0 16px; }}
.card {{ border:1px solid #e5e7eb; border-radius:10px; padding:12px; background:#fff; }}
.card .k {{ font-size:12px; color:#6b7280; }}
.table-wrap {{ overflow: auto; max-height: 68vh; border:1px solid #e5e7eb; border-radius:10px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; text-align: right; white-space: nowrap; }}
th {{ background: #f8fafc; position: sticky; top: 0; z-index: 2; }}
td:first-child, th:first-child {{ text-align: left; position: sticky; left: 0; background: #fff; z-index: 3; }}
img {{ max-width: 100%; border:1px solid #e5e7eb; border-radius:10px; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; background:#eff6ff; color:#1d4ed8; }}
input[type="text"], select {{ border:1px solid #d1d5db; border-radius:8px; padding:8px 10px; font-size: 13px; }}
.controls {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:8px 0 10px; }}
.controls label {{ font-size:13px; color:#374151; }}
.plot-wrap {{ border:1px solid #e5e7eb; border-radius:10px; padding:12px; background:#fff; overflow:auto; }}
.legend {{ display:flex; gap:14px; margin:8px 0; font-size:12px; color:#4b5563; }}
.dot {{ display:inline-block; width:10px; height:10px; border-radius:999px; margin-right:4px; }}
.axis-label {{ font-size:11px; fill:#6b7280; }}
</style>
</head>
<body>
<h1>rPPG 参数阈值与逐文件 HR 对比报告（HTML）</h1>
<div class="muted">输入：逐秒对齐 CSV；阈值：freq_conf（0.80~0.94）；支持逐文件 HR 数值和曲线对比。</div>

<div class="section">
  <h2>总体阈值对比</h2>
  <div class="table-wrap"><table><thead><tr><th>threshold</th><th>n</th><th>coverage</th><th>MAE</th><th>RMSE</th><th>Bias(rPPG-ECG)</th></tr></thead><tbody>
  {"".join(overall_html_rows)}
  </tbody></table></div>
</div>

<div class="section">
  <h2>可视化图</h2>
  <div class="grid">
    <div class="card"><div class="k">总体权衡</div><img src="fc_threshold_tradeoff_overall.png" alt="tradeoff"/></div>
    <div class="card"><div class="k">每样本 MAE 热力图</div><img src="fc_threshold_mae_heatmap_per_sample.png" alt="mae heatmap"/></div>
    <div class="card"><div class="k">每样本覆盖率热力图</div><img src="fc_threshold_coverage_heatmap_per_sample.png" alt="coverage heatmap"/></div>
  </div>
</div>

<div class="section">
  <h2>最短时间与精度可视化</h2>
  <div class="muted">目标：看“等待多少秒”能拿到更高精度的 HR。这里按 min_sec 和 freq_conf 阈值联合统计 MAE/覆盖率。</div>
  <div class="controls">
    <label>threshold:
      <select id="warmupThreshold" onchange="renderWarmup()">
        {"".join([f'<option value="{thr:.2f}">{thr:.2f}</option>' for thr in warmup_thresholds])}
      </select>
    </label>
  </div>
  <div id="warmupMeta" class="muted"></div>
  <div class="grid">
    <div class="card">
      <div class="k">MAE vs min_sec</div>
      <div class="plot-wrap"><svg id="warmupMaePlot" width="560" height="300" viewBox="0 0 560 300"></svg></div>
    </div>
    <div class="card">
      <div class="k">Coverage vs min_sec</div>
      <div class="plot-wrap"><svg id="warmupCovPlot" width="560" height="300" viewBox="0 0 560 300"></svg></div>
    </div>
  </div>
  <div class="table-wrap"><table id="warmupTable">
    <thead><tr><th>min_sec</th><th>n</th><th>coverage</th><th>MAE</th><th>RMSE</th></tr></thead>
    <tbody></tbody>
  </table></div>
</div>

<div class="section">
  <h2>达到 K 个高置信点所需秒数（按视频分布）</h2>
  <div class="muted">每个单元是分位数秒数；例如 `q50` 表示 50% 视频在该秒数前可拿到 K 个高置信点。</div>
  <div class="table-wrap"><table id="timeToKTable">
    <thead><tr>
      <th>threshold</th><th>K</th><th>have/miss</th>
      <th>q0</th><th>q25</th><th>q50</th><th>q75</th><th>q90</th><th>q100</th>
    </tr></thead>
    <tbody></tbody>
  </table></div>
</div>

<div class="section">
  <h2>逐样本参数矩阵（MAE / 覆盖率）</h2>
  <div class="muted">筛选 sample（例如 003/1-1、6/6-3）。</div>
  <p><input id="qMatrix" type="text" placeholder="筛选 sample..." oninput="filterMatrixRows()" /></p>
  <div class="table-wrap"><table id="matrix">
    <thead><tr>
      <th>sample</th><th>n_all</th><th>mae_all</th>
      {threshold_cells}
    </tr></thead>
    <tbody>
      {"".join(matrix_rows)}
    </tbody>
  </table></div>
</div>

<div class="section">
  <h2>每一个文件的 HR 测量数值对比</h2>
  <div class="muted">列含义：`ECG均值`、`rPPG均值`、`Bias=rPPG-ECG`、`MAE`、`RMSE`。阈值改变时，按该阈值筛选同一文件内的秒级点。</div>
  <div class="controls">
    <label>threshold:
      <select id="hrThreshold" onchange="renderHrTable(); syncPlotThreshold();">
        {threshold_options}
      </select>
    </label>
    <label>sample 过滤:
      <input id="qHr" type="text" placeholder="输入 sample 关键字..." oninput="renderHrTable()" />
    </label>
  </div>
  <div class="table-wrap"><table id="hrTable">
    <thead><tr>
      <th>sample</th><th>n_all</th><th>n_used</th><th>coverage</th>
      <th>ECG均值</th><th>rPPG均值</th><th>Bias</th><th>MAE</th><th>RMSE</th>
    </tr></thead>
    <tbody></tbody>
  </table></div>
</div>

<div class="section">
  <h2>每文件逐秒 HR 曲线对比</h2>
  <div class="controls">
    <label>sample:
      <select id="plotSample" onchange="drawHrPlot()">{sample_options}</select>
    </label>
    <label>threshold:
      <select id="plotThreshold" onchange="drawHrPlot()">{threshold_options}</select>
    </label>
  </div>
  <div id="plotMeta" class="muted"></div>
  <div class="legend">
    <span><span class="dot" style="background:#9ca3af;"></span>ECG(all points)</span>
    <span><span class="dot" style="background:#2563eb;"></span>ECG(filtered)</span>
    <span><span class="dot" style="background:#ea580c;"></span>rPPG(filtered)</span>
  </div>
  <div class="plot-wrap">
    <svg id="hrPlot" width="960" height="360" viewBox="0 0 960 360" role="img" aria-label="HR plot"></svg>
  </div>
</div>

<div class="section">
  <h2>泛化分析（当前发布策略 vs baseline）</h2>
  <div id="genMeta" class="muted">加载中...</div>
  <div class="grid" id="genCards"></div>

  <h3>Domain 对比</h3>
  <div class="table-wrap"><table id="genDomainTable">
    <thead><tr>
      <th>domain</th><th>n</th><th>sample_count</th><th>group_count</th>
      <th>MAE</th><th>RMSE</th><th>corr</th><th>bias</th><th>p90_abs_error</th>
    </tr></thead>
    <tbody></tbody>
  </table></div>

  <h3>Group 对比（当前策略）</h3>
  <div class="table-wrap"><table id="genGroupTable">
    <thead><tr>
      <th>group_id</th><th>group_raw</th><th>n</th><th>sample_count</th>
      <th>MAE</th><th>RMSE</th><th>corr</th><th>bias</th><th>p90_abs_error</th>
    </tr></thead>
    <tbody></tbody>
  </table></div>

  <h3>Group 相对 Baseline 变化</h3>
  <div class="table-wrap"><table id="genDeltaTable">
    <thead><tr>
      <th>group_id</th><th>group_raw</th><th>n_base</th><th>n_cur</th><th>coverage_ratio</th>
      <th>MAE_base</th><th>MAE_cur</th><th>Delta_MAE</th>
      <th>RMSE_base</th><th>RMSE_cur</th><th>Delta_RMSE</th>
      <th>corr_base</th><th>corr_cur</th><th>Delta_corr</th>
    </tr></thead>
    <tbody></tbody>
  </table></div>

  <h3>Worst Samples（当前策略）</h3>
  <div class="table-wrap"><table id="genSampleTable">
    <thead><tr>
      <th>sample</th><th>group_id</th><th>group_raw</th><th>n</th>
      <th>MAE</th><th>RMSE</th><th>corr</th><th>bias</th><th>p90_abs_error</th><th>first_sec</th>
    </tr></thead>
    <tbody></tbody>
  </table></div>
</div>

<div class="section">
  <h2>ECG-rPPG 位移（lag）分析</h2>
  <div class="controls">
    <label>metric:
      <select id="lagMetric" onchange="renderLagAnalysis()">
        <option value="mae">MAE</option>
        <option value="rmse">RMSE</option>
      </select>
    </label>
  </div>
  <div id="lagMeta" class="muted">加载中...</div>
  <div class="plot-wrap">
    <svg id="lagCurvePlot" width="960" height="340" viewBox="0 0 960 340" role="img" aria-label="Lag curve"></svg>
  </div>

  <h3>Lag 曲线数值（baseline vs current）</h3>
  <div class="table-wrap"><table id="lagCurveTable">
    <thead><tr><th>lag</th><th>baseline_n</th><th>baseline_MAE</th><th>baseline_RMSE</th><th>current_n</th><th>current_MAE</th><th>current_RMSE</th></tr></thead>
    <tbody></tbody>
  </table></div>

  <h3>每样本 lag 改善（baseline）</h3>
  <div class="table-wrap"><table id="lagSampleBaselineTable">
    <thead><tr><th>sample</th><th>n0</th><th>mae_lag0</th><th>best_lag</th><th>best_mae</th><th>best_n</th><th>mae_gain</th><th>gain_ratio</th></tr></thead>
    <tbody></tbody>
  </table></div>

  <h3>每样本 lag 改善（current）</h3>
  <div class="table-wrap"><table id="lagSampleCurrentTable">
    <thead><tr><th>sample</th><th>n0</th><th>mae_lag0</th><th>best_lag</th><th>best_mae</th><th>best_n</th><th>mae_gain</th><th>gain_ratio</th></tr></thead>
    <tbody></tbody>
  </table></div>

  <h3>固定窗口 Lag 扫描（baseline）</h3>
  <div class="table-wrap"><table id="lagCommonWindowTable">
    <thead><tr><th>sample</th><th>n_common</th><th>best_lag_mae</th><th>best_mae</th><th>lag0_mae</th><th>mae_gain</th><th>best_lag_corr</th><th>best_corr</th><th>lag0_corr</th><th>corr_gain</th></tr></thead>
    <tbody></tbody>
  </table></div>

  <h3>Diff-Corr Lag（baseline）</h3>
  <div class="table-wrap"><table id="lagDiffCorrTable">
    <thead><tr><th>sample</th><th>n</th><th>lag0_diff_mae</th><th>lag0_diff_corr</th><th>best_lag_by_diff_corr</th><th>best_diff_mae</th><th>best_diff_corr</th><th>corr_gain</th></tr></thead>
    <tbody></tbody>
  </table></div>

  <h3>Global Diff-Corr 曲线（baseline）</h3>
  <div class="table-wrap"><table id="lagGlobalDiffTable">
    <thead><tr><th>lag</th><th>diff_mae</th><th>diff_corr</th><th>n</th></tr></thead>
    <tbody></tbody>
  </table></div>
</div>

<script>
const SAMPLE_STATS = {sample_stats_json};
const SERIES_DATA = {series_json};
const THRESHOLDS = {threshold_js};
const WARMUP_ROWS = {warmup_json};
const TIME_TO_K_ROWS = {time_to_k_json};
const WARMUP_THRESHOLDS = {warmup_threshold_js};
const GENERALIZATION = {generalization_json};
const LAG_ANALYSIS = {lag_json};

function fmt(v, digits=3) {{
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "";
  return Number(v).toFixed(digits);
}}

function fmtPct(v) {{
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "";
  return (Number(v) * 100).toFixed(1) + "%";
}}

function fmtSigned(v, digits=3) {{
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "";
  const n = Number(v);
  const s = n >= 0 ? "+" : "";
  return s + n.toFixed(digits);
}}

function filterMatrixRows() {{
  const q = (document.getElementById("qMatrix").value || "").toLowerCase().trim();
  const rows = document.querySelectorAll("#matrix tbody tr");
  for (const tr of rows) {{
    const s = (tr.cells[0]?.textContent || "").toLowerCase();
    tr.style.display = (!q || s.includes(q)) ? "" : "none";
  }}
}}

function getThresholdKey(id) {{
  return document.getElementById(id).value || "all";
}}

function drawLinePlot(svgId, points, key, color, yLabel, yAsPct=false) {{
  const svg = document.getElementById(svgId);
  if (!svg || !points.length) {{
    if (svg) svg.innerHTML = "";
    return;
  }}
  const w = 560, h = 300;
  const ml = 48, mr = 18, mt = 12, mb = 34;
  const xVals = points.map(p => Number(p.min_sec));
  const yVals = points.map(p => Number(p[key]));
  const xMin = Math.min(...xVals);
  const xMax = Math.max(...xVals);
  const xSpan = Math.max(1, xMax - xMin);
  let yMin = Math.min(...yVals);
  let yMax = Math.max(...yVals);
  if (yAsPct) {{
    yMin = 0;
    yMax = Math.max(1, yMax * 1.05);
  }} else {{
    const pad = Math.max(0.2, (yMax - yMin) * 0.15);
    yMin = Math.max(0, yMin - pad);
    yMax = yMax + pad;
  }}
  const ySpan = Math.max(1e-9, yMax - yMin);

  const xFn = x => ml + (w - ml - mr) * (x - xMin) / xSpan;
  const yFn = y => mt + (h - mt - mb) * (1 - (y - yMin) / ySpan);

  let path = "";
  for (let i = 0; i < points.length; i += 1) {{
    const x = xFn(Number(points[i].min_sec)).toFixed(2);
    const y = yFn(Number(points[i][key])).toFixed(2);
    path += (i === 0 ? `M ${{x}} ${{y}}` : ` L ${{x}} ${{y}}`);
  }}

  const grid = [];
  for (let i = 0; i <= 5; i += 1) {{
    const yy = mt + (h - mt - mb) * i / 5;
    const val = yMax - (ySpan * i / 5);
    const label = yAsPct ? `${{(val * 100).toFixed(0)}}%` : val.toFixed(2);
    grid.push(`<line x1="${{ml}}" y1="${{yy.toFixed(2)}}" x2="${{w - mr}}" y2="${{yy.toFixed(2)}}" stroke="#eef2f7" stroke-width="1"/>`);
    grid.push(`<text x="${{ml - 8}}" y="${{(yy + 4).toFixed(2)}}" text-anchor="end" class="axis-label">${{label}}</text>`);
  }}

  const xTicks = points.map(p => Number(p.min_sec));
  const xTickHtml = xTicks.map(v => {{
    const xx = xFn(v).toFixed(2);
    return `<text x="${{xx}}" y="${{h - 12}}" text-anchor="middle" class="axis-label">${{v}}</text>`;
  }}).join("");

  const dots = points.map(p => {{
    const xx = xFn(Number(p.min_sec)).toFixed(2);
    const yy = yFn(Number(p[key])).toFixed(2);
    return `<circle cx="${{xx}}" cy="${{yy}}" r="3.2" fill="${{color}}" />`;
  }}).join("");

  svg.innerHTML = `
    <rect x="0" y="0" width="${{w}}" height="${{h}}" fill="#fff"/>
    ${{grid.join("")}}
    <line x1="${{ml}}" y1="${{h - mb}}" x2="${{w - mr}}" y2="${{h - mb}}" stroke="#9ca3af"/>
    <line x1="${{ml}}" y1="${{mt}}" x2="${{ml}}" y2="${{h - mb}}" stroke="#9ca3af"/>
    <text x="${{w / 2}}" y="${{h - 5}}" text-anchor="middle" class="axis-label">min_sec</text>
    <text x="14" y="${{h / 2}}" text-anchor="middle" class="axis-label" transform="rotate(-90 14 ${{h/2}})">${{yLabel}}</text>
    <path d="${{path}}" fill="none" stroke="${{color}}" stroke-width="2"/>
    ${{dots}}
    ${{xTickHtml}}
  `;
}}

function renderWarmup() {{
  const thr = document.getElementById("warmupThreshold").value;
  const rows = WARMUP_ROWS
    .filter(r => Number(r.threshold).toFixed(2) === thr)
    .sort((a, b) => Number(a.min_sec) - Number(b.min_sec));

  drawLinePlot("warmupMaePlot", rows, "mae", "#ea580c", "MAE");
  drawLinePlot("warmupCovPlot", rows, "coverage", "#2563eb", "Coverage", true);

  const tbody = document.querySelector("#warmupTable tbody");
  tbody.innerHTML = rows.map(r => `<tr>
    <td>${{r.min_sec}}</td>
    <td>${{r.n}}</td>
    <td>${{fmtPct(r.coverage)}}</td>
    <td>${{fmt(r.mae)}}</td>
    <td>${{fmt(r.rmse)}}</td>
  </tr>`).join("");

  const hit = rows.find(r => Number(r.mae) < 3);
  const meta = document.getElementById("warmupMeta");
  if (hit) {{
    meta.textContent = `fc>=${{thr}} 时，最早在 min_sec=${{hit.min_sec}} 达到 MAE=${{fmt(hit.mae)}}（coverage=${{fmtPct(hit.coverage)}}）。`;
  }} else {{
    meta.textContent = `fc>=${{thr}} 未达到 MAE<3。`;
  }}
}}

function renderTimeToKTable() {{
  const tbody = document.querySelector("#timeToKTable tbody");
  const rows = TIME_TO_K_ROWS
    .slice()
    .sort((a, b) => Number(a.threshold) - Number(b.threshold) || Number(a.k) - Number(b.k));

  tbody.innerHTML = rows.map(r => `<tr>
    <td>${{Number(r.threshold).toFixed(2)}}</td>
    <td>${{r.k}}</td>
    <td>${{r.have}}/${{r.missing}}</td>
    <td>${{fmt(r.q0, 1)}}</td>
    <td>${{fmt(r.q25, 1)}}</td>
    <td><b>${{fmt(r.q50, 1)}}</b></td>
    <td>${{fmt(r.q75, 1)}}</td>
    <td>${{fmt(r.q90, 1)}}</td>
    <td>${{fmt(r.q100, 1)}}</td>
  </tr>`).join("");
}}

function renderGeneralization() {{
  const meta = document.getElementById("genMeta");
  const cards = document.getElementById("genCards");
  const g = GENERALIZATION || {{}};
  const overview = g.overview || {{}};

  const hasAny =
    (Array.isArray(g.domain) && g.domain.length > 0) ||
    (Array.isArray(g.group) && g.group.length > 0) ||
    (Array.isArray(g.delta) && g.delta.length > 0) ||
    (Array.isArray(g.samples) && g.samples.length > 0);

  if (!hasAny) {{
    if (meta) meta.textContent = "未找到泛化分析 CSV（可通过 --generalization-*-csv 指定）。";
    if (cards) cards.innerHTML = "";
    return;
  }}

  const nCur = overview.n_cur;
  const nBase = overview.n_base;
  if (meta) {{
    meta.textContent =
      `当前策略 n=${{nCur ?? ""}} / baseline n=${{nBase ?? ""}} | ` +
      `MAE ${{fmt(overview.mae_cur)}} vs ${{fmt(overview.mae_base)}} (Δ${{fmtSigned(overview.delta_mae)}}), ` +
      `RMSE ${{fmt(overview.rmse_cur)}} vs ${{fmt(overview.rmse_base)}} (Δ${{fmtSigned(overview.delta_rmse)}}), ` +
      `corr ${{fmt(overview.corr_cur)}} vs ${{fmt(overview.corr_base)}} (Δ${{fmtSigned(overview.delta_corr)}}).`;
  }}

  if (cards) {{
    cards.innerHTML = `
      <div class="card"><div class="k">Current MAE / RMSE / corr</div><div><b>${{fmt(overview.mae_cur)}}</b> / ${{fmt(overview.rmse_cur)}} / ${{fmt(overview.corr_cur)}}</div></div>
      <div class="card"><div class="k">Baseline MAE / RMSE / corr</div><div><b>${{fmt(overview.mae_base)}}</b> / ${{fmt(overview.rmse_base)}} / ${{fmt(overview.corr_base)}}</div></div>
      <div class="card"><div class="k">Delta (Current - Baseline)</div><div><b>${{fmtSigned(overview.delta_mae)}}</b> / ${{fmtSigned(overview.delta_rmse)}} / ${{fmtSigned(overview.delta_corr)}}</div></div>
      <div class="card"><div class="k">Coverage</div><div><b>${{nCur ?? ""}}</b> / ${{nBase ?? ""}} (${{fmtPct((nCur && nBase) ? nCur / nBase : null)}})</div></div>
    `;
  }}

  const domainRows = (g.domain || []).map(r => `<tr>
    <td>${{r.domain ?? ""}}</td>
    <td>${{r.n ?? ""}}</td>
    <td>${{r.sample_count ?? ""}}</td>
    <td>${{r.group_count ?? ""}}</td>
    <td>${{fmt(r.mae)}}</td>
    <td>${{fmt(r.rmse)}}</td>
    <td>${{fmt(r.corr)}}</td>
    <td>${{fmt(r.bias)}}</td>
    <td>${{fmt(r.p90_abs_error)}}</td>
  </tr>`).join("");
  document.querySelector("#genDomainTable tbody").innerHTML = domainRows;

  const groupRows = (g.group || []).map(r => `<tr>
    <td>${{r.group_id ?? ""}}</td>
    <td>${{r.group_raw ?? ""}}</td>
    <td>${{r.n ?? ""}}</td>
    <td>${{r.sample_count ?? ""}}</td>
    <td>${{fmt(r.mae)}}</td>
    <td>${{fmt(r.rmse)}}</td>
    <td>${{fmt(r.corr)}}</td>
    <td>${{fmt(r.bias)}}</td>
    <td>${{fmt(r.p90_abs_error)}}</td>
  </tr>`).join("");
  document.querySelector("#genGroupTable tbody").innerHTML = groupRows;

  const deltaRows = (g.delta || []).map(r => `<tr>
    <td>${{r.group_id ?? ""}}</td>
    <td>${{r.group_raw ?? ""}}</td>
    <td>${{r.n_base ?? ""}}</td>
    <td>${{r.n_cur ?? ""}}</td>
    <td>${{fmtPct(r.coverage_ratio)}}</td>
    <td>${{fmt(r.mae_base)}}</td>
    <td>${{fmt(r.mae_cur)}}</td>
    <td>${{fmtSigned(r.delta_mae)}}</td>
    <td>${{fmt(r.rmse_base)}}</td>
    <td>${{fmt(r.rmse_cur)}}</td>
    <td>${{fmtSigned(r.delta_rmse)}}</td>
    <td>${{fmt(r.corr_base)}}</td>
    <td>${{fmt(r.corr_cur)}}</td>
    <td>${{fmtSigned(r.delta_corr)}}</td>
  </tr>`).join("");
  document.querySelector("#genDeltaTable tbody").innerHTML = deltaRows;

  const sampleRows = (g.samples || []).map(r => `<tr>
    <td>${{r.sample ?? ""}}</td>
    <td>${{r.group_id ?? ""}}</td>
    <td>${{r.group_raw ?? ""}}</td>
    <td>${{r.n ?? ""}}</td>
    <td>${{fmt(r.mae)}}</td>
    <td>${{fmt(r.rmse)}}</td>
    <td>${{fmt(r.corr)}}</td>
    <td>${{fmt(r.bias)}}</td>
    <td>${{fmt(r.p90_abs_error)}}</td>
    <td>${{fmt(r.first_sec, 0)}}</td>
  </tr>`).join("");
  document.querySelector("#genSampleTable tbody").innerHTML = sampleRows;
}}

function buildLagPath(rows, metricKey, xFn, yFn) {{
  if (!rows.length) return "";
  let d = "";
  for (let i = 0; i < rows.length; i += 1) {{
    const x = xFn(Number(rows[i].lag)).toFixed(2);
    const y = yFn(Number(rows[i][metricKey])).toFixed(2);
    d += (i === 0 ? `M ${{x}} ${{y}}` : ` L ${{x}} ${{y}}`);
  }}
  return d;
}}

function drawLagCurvePlot(metricKey) {{
  const svg = document.getElementById("lagCurvePlot");
  const base = (LAG_ANALYSIS.curve_baseline || []).slice().sort((a, b) => Number(a.lag) - Number(b.lag));
  const cur = (LAG_ANALYSIS.curve_current || []).slice().sort((a, b) => Number(a.lag) - Number(b.lag));
  if (!svg) return;
  if (!base.length && !cur.length) {{
    svg.innerHTML = "";
    return;
  }}

  const all = base.concat(cur).filter(r => !Number.isNaN(Number(r[metricKey])));
  const xVals = all.map(r => Number(r.lag));
  const yVals = all.map(r => Number(r[metricKey]));
  if (!xVals.length || !yVals.length) {{
    svg.innerHTML = "";
    return;
  }}

  const w = 960, h = 340;
  const ml = 52, mr = 20, mt = 12, mb = 40;
  const xMin = Math.min(...xVals);
  const xMax = Math.max(...xVals);
  const xSpan = Math.max(1, xMax - xMin);
  let yMin = Math.min(...yVals);
  let yMax = Math.max(...yVals);
  const yPad = Math.max(0.1, (yMax - yMin) * 0.15);
  yMin = Math.max(0, yMin - yPad);
  yMax = yMax + yPad;
  const ySpan = Math.max(1e-9, yMax - yMin);

  const xFn = x => ml + (w - ml - mr) * (x - xMin) / xSpan;
  const yFn = y => mt + (h - mt - mb) * (1 - (y - yMin) / ySpan);

  const basePath = buildLagPath(base, metricKey, xFn, yFn);
  const curPath = buildLagPath(cur, metricKey, xFn, yFn);

  const grid = [];
  for (let i = 0; i <= 5; i += 1) {{
    const yy = mt + (h - mt - mb) * i / 5;
    const val = yMax - (ySpan * i / 5);
    grid.push(`<line x1="${{ml}}" y1="${{yy.toFixed(2)}}" x2="${{w - mr}}" y2="${{yy.toFixed(2)}}" stroke="#eef2f7" stroke-width="1"/>`);
    grid.push(`<text x="${{ml - 8}}" y="${{(yy + 4).toFixed(2)}}" text-anchor="end" class="axis-label">${{val.toFixed(2)}}</text>`);
  }}
  for (let lag = Math.ceil(xMin); lag <= Math.floor(xMax); lag += 2) {{
    const xx = xFn(lag).toFixed(2);
    grid.push(`<line x1="${{xx}}" y1="${{mt}}" x2="${{xx}}" y2="${{h - mb}}" stroke="#f3f4f6" stroke-width="1"/>`);
    grid.push(`<text x="${{xx}}" y="${{h - 10}}" text-anchor="middle" class="axis-label">${{lag}}</text>`);
  }}

  svg.innerHTML = `
    <rect x="0" y="0" width="${{w}}" height="${{h}}" fill="#fff"/>
    ${{grid.join("")}}
    <line x1="${{ml}}" y1="${{h - mb}}" x2="${{w - mr}}" y2="${{h - mb}}" stroke="#9ca3af"/>
    <line x1="${{ml}}" y1="${{mt}}" x2="${{ml}}" y2="${{h - mb}}" stroke="#9ca3af"/>
    <text x="${{w / 2}}" y="${{h - 10}}" text-anchor="middle" class="axis-label">lag (sec)</text>
    <text x="14" y="${{h / 2}}" text-anchor="middle" class="axis-label" transform="rotate(-90 14 ${{h/2}})">${{metricKey.toUpperCase()}}</text>
    <path d="${{basePath}}" fill="none" stroke="#1d4ed8" stroke-width="2.2"/>
    <path d="${{curPath}}" fill="none" stroke="#ea580c" stroke-width="2.2"/>
  `;
}}

function renderLagAnalysis() {{
  const metricKey = (document.getElementById("lagMetric")?.value || "mae").toLowerCase();
  const base = (LAG_ANALYSIS.curve_baseline || []).slice().sort((a, b) => Number(a.lag) - Number(b.lag));
  const cur = (LAG_ANALYSIS.curve_current || []).slice().sort((a, b) => Number(a.lag) - Number(b.lag));
  const meta = document.getElementById("lagMeta");

  if (!base.length && !cur.length) {{
    if (meta) meta.textContent = "未找到 lag 分析 CSV（可通过 --lag-*-csv 指定）。";
    return;
  }}

  drawLagCurvePlot(metricKey);

  const pickMin = rows => {{
    if (!rows.length) return null;
    return rows.reduce((best, curRow) => {{
      const curV = Number(curRow[metricKey]);
      if (!best) return curRow;
      return curV < Number(best[metricKey]) ? curRow : best;
    }}, null);
  }};
  const base0 = base.find(r => Number(r.lag) === 0) || null;
  const cur0 = cur.find(r => Number(r.lag) === 0) || null;
  const baseBest = pickMin(base);
  const curBest = pickMin(cur);
  if (meta) {{
    meta.textContent =
      `metric=${{metricKey.toUpperCase()}} | baseline lag0=${{base0 ? fmt(base0[metricKey]) : ""}}, best=${{baseBest ? fmt(baseBest[metricKey]) : ""}}@lag=${{baseBest ? baseBest.lag : ""}} | ` +
      `current lag0=${{cur0 ? fmt(cur0[metricKey]) : ""}}, best=${{curBest ? fmt(curBest[metricKey]) : ""}}@lag=${{curBest ? curBest.lag : ""}}`;
  }}

  const curveMap = new Map();
  for (const r of base) {{
    const k = String(r.lag);
    curveMap.set(k, {{
      lag: Number(r.lag),
      baseline_n: r.n,
      baseline_mae: r.mae,
      baseline_rmse: r.rmse,
      current_n: null,
      current_mae: null,
      current_rmse: null,
    }});
  }}
  for (const r of cur) {{
    const k = String(r.lag);
    const item = curveMap.get(k) || {{
      lag: Number(r.lag),
      baseline_n: null,
      baseline_mae: null,
      baseline_rmse: null,
      current_n: null,
      current_mae: null,
      current_rmse: null,
    }};
    item.current_n = r.n;
    item.current_mae = r.mae;
    item.current_rmse = r.rmse;
    curveMap.set(k, item);
  }}
  const curveRows = Array.from(curveMap.values()).sort((a, b) => a.lag - b.lag);
  document.querySelector("#lagCurveTable tbody").innerHTML = curveRows.map(r => `<tr>
    <td>${{r.lag}}</td>
    <td>${{r.baseline_n ?? ""}}</td>
    <td>${{fmt(r.baseline_mae)}}</td>
    <td>${{fmt(r.baseline_rmse)}}</td>
    <td>${{r.current_n ?? ""}}</td>
    <td>${{fmt(r.current_mae)}}</td>
    <td>${{fmt(r.current_rmse)}}</td>
  </tr>`).join("");

  const sampleCols = rows => rows.map(r => `<tr>
    <td>${{r.sample ?? ""}}</td>
    <td>${{r.n0 ?? ""}}</td>
    <td>${{fmt(r.mae_lag0)}}</td>
    <td>${{r.best_lag ?? ""}}</td>
    <td>${{fmt(r.best_mae)}}</td>
    <td>${{r.best_n ?? ""}}</td>
    <td>${{fmt(r.mae_gain)}}</td>
    <td>${{fmtPct(r.gain_ratio)}}</td>
  </tr>`).join("");
  document.querySelector("#lagSampleBaselineTable tbody").innerHTML = sampleCols(LAG_ANALYSIS.sample_baseline || []);
  document.querySelector("#lagSampleCurrentTable tbody").innerHTML = sampleCols(LAG_ANALYSIS.sample_current || []);

  document.querySelector("#lagCommonWindowTable tbody").innerHTML = (LAG_ANALYSIS.commonwindow || []).map(r => `<tr>
    <td>${{r.sample ?? ""}}</td>
    <td>${{r.n_common ?? ""}}</td>
    <td>${{r.best_lag_mae ?? ""}}</td>
    <td>${{fmt(r.best_mae)}}</td>
    <td>${{fmt(r.lag0_mae)}}</td>
    <td>${{fmt(r.mae_gain)}}</td>
    <td>${{r.best_lag_corr ?? ""}}</td>
    <td>${{fmt(r.best_corr)}}</td>
    <td>${{fmt(r.lag0_corr)}}</td>
    <td>${{fmt(r.corr_gain)}}</td>
  </tr>`).join("");

  document.querySelector("#lagDiffCorrTable tbody").innerHTML = (LAG_ANALYSIS.diffcorr || []).map(r => `<tr>
    <td>${{r.sample ?? ""}}</td>
    <td>${{r.n ?? ""}}</td>
    <td>${{fmt(r.lag0_diff_mae)}}</td>
    <td>${{fmt(r.lag0_diff_corr)}}</td>
    <td>${{r.best_lag_by_diff_corr ?? ""}}</td>
    <td>${{fmt(r.best_diff_mae)}}</td>
    <td>${{fmt(r.best_diff_corr)}}</td>
    <td>${{fmt(r.corr_gain)}}</td>
  </tr>`).join("");

  document.querySelector("#lagGlobalDiffTable tbody").innerHTML = (LAG_ANALYSIS.global_diff_curve || []).map(r => `<tr>
    <td>${{r.lag ?? ""}}</td>
    <td>${{fmt(r.diff_mae)}}</td>
    <td>${{fmt(r.diff_corr)}}</td>
    <td>${{r.n ?? ""}}</td>
  </tr>`).join("");
}}

function renderHrTable() {{
  const thr = getThresholdKey("hrThreshold");
  const q = (document.getElementById("qHr").value || "").toLowerCase().trim();
  const tbody = document.querySelector("#hrTable tbody");

  const rows = SAMPLE_STATS
    .filter(r => !q || r.sample.toLowerCase().includes(q))
    .map(r => {{
      const s = (r.stats && r.stats[thr]) ? r.stats[thr] : null;
      const mae = s ? s.mae : null;
      return {{ r, s, mae }};
    }})
    .sort((a, b) => {{
      const av = (a.mae === null || Number.isNaN(Number(a.mae))) ? -1e9 : Number(a.mae);
      const bv = (b.mae === null || Number.isNaN(Number(b.mae))) ? -1e9 : Number(b.mae);
      return bv - av;
    }});

  tbody.innerHTML = rows.map(item => {{
    const r = item.r;
    const s = item.s || {{}};
    const nUsed = (s.n === undefined || s.n === null) ? "" : s.n;
    const bias = s.bias;
    const biasText = fmt(bias);
    const biasColor = (bias === null || Number.isNaN(Number(bias))) ? "#111827" : (Number(bias) >= 0 ? "#b91c1c" : "#1d4ed8");
    return `<tr>
      <td>${{r.sample}}</td>
      <td>${{r.n_all}}</td>
      <td>${{nUsed}}</td>
      <td>${{fmtPct(s.coverage)}}</td>
      <td>${{fmt(s.ecg_mean)}}</td>
      <td>${{fmt(s.rppg_mean)}}</td>
      <td style="color:${{biasColor}}">${{biasText}}</td>
      <td>${{fmt(s.mae)}}</td>
      <td>${{fmt(s.rmse)}}</td>
    </tr>`;
  }}).join("");
}}

function syncPlotThreshold() {{
  const cur = getThresholdKey("hrThreshold");
  const plotSel = document.getElementById("plotThreshold");
  if (plotSel.value !== cur) {{
    plotSel.value = cur;
  }}
  drawHrPlot();
}}

function linePath(xs, ys, xFn, yFn) {{
  if (!xs.length) return "";
  let d = `M ${{xFn(xs[0]).toFixed(2)}} ${{yFn(ys[0]).toFixed(2)}}`;
  for (let i = 1; i < xs.length; i += 1) {{
    d += ` L ${{xFn(xs[i]).toFixed(2)}} ${{yFn(ys[i]).toFixed(2)}}`;
  }}
  return d;
}}

function drawHrPlot() {{
  const sample = document.getElementById("plotSample").value;
  const thr = getThresholdKey("plotThreshold");
  const svg = document.getElementById("hrPlot");
  const meta = document.getElementById("plotMeta");
  if (!sample || !SERIES_DATA[sample]) {{
    svg.innerHTML = "";
    meta.textContent = "请选择一个 sample。";
    return;
  }}

  const ser = SERIES_DATA[sample];
  const sec = ser.sec || [];
  const ecg = ser.ecg || [];
  const rppg = ser.rppg || [];
  const fc = ser.fc || [];

  const idx = [];
  const thrNum = (thr === "all") ? null : Number(thr);
  for (let i = 0; i < sec.length; i += 1) {{
    if (thrNum === null || fc[i] >= thrNum) idx.push(i);
  }}

  if (!idx.length) {{
    svg.innerHTML = "";
    meta.textContent = `${{sample}} 在 threshold=${{thr}} 下没有可用点。`;
    return;
  }}

  const w = 960, h = 360;
  const ml = 52, mr = 20, mt = 12, mb = 40;
  const xMin = Math.min(...sec);
  const xMax = Math.max(...sec);
  const secSpan = Math.max(1, xMax - xMin);

  const ecgSel = idx.map(i => ecg[i]);
  const rppgSel = idx.map(i => rppg[i]);
  const yMinRaw = Math.min(...ecgSel, ...rppgSel);
  const yMaxRaw = Math.max(...ecgSel, ...rppgSel);
  const yPad = Math.max(2, (yMaxRaw - yMinRaw) * 0.15);
  const yMin = yMinRaw - yPad;
  const yMax = yMaxRaw + yPad;
  const ySpan = Math.max(1, yMax - yMin);

  const xFn = s => ml + (w - ml - mr) * (s - xMin) / secSpan;
  const yFn = v => mt + (h - mt - mb) * (1 - (v - yMin) / ySpan);

  const secSel = idx.map(i => sec[i]);
  const ecgAllPath = linePath(sec, ecg, xFn, yFn);
  const ecgSelPath = linePath(secSel, ecgSel, xFn, yFn);
  const rppgSelPath = linePath(secSel, rppgSel, xFn, yFn);

  const grid = [];
  const ticks = 5;
  for (let i = 0; i <= ticks; i += 1) {{
    const yy = mt + (h - mt - mb) * i / ticks;
    const val = yMax - (ySpan * i / ticks);
    grid.push(`<line x1="${{ml}}" y1="${{yy.toFixed(2)}}" x2="${{w - mr}}" y2="${{yy.toFixed(2)}}" stroke="#eef2f7" stroke-width="1"/>`);
    grid.push(`<text x="${{ml - 8}}" y="${{(yy + 4).toFixed(2)}}" text-anchor="end" class="axis-label">${{val.toFixed(0)}}</text>`);
  }}

  svg.innerHTML = `
    <rect x="0" y="0" width="${{w}}" height="${{h}}" fill="#ffffff"/>
    ${{grid.join("")}}
    <line x1="${{ml}}" y1="${{h - mb}}" x2="${{w - mr}}" y2="${{h - mb}}" stroke="#9ca3af" />
    <line x1="${{ml}}" y1="${{mt}}" x2="${{ml}}" y2="${{h - mb}}" stroke="#9ca3af" />
    <text x="${{w / 2}}" y="${{h - 10}}" text-anchor="middle" class="axis-label">sec</text>
    <text x="14" y="${{h / 2}}" text-anchor="middle" class="axis-label" transform="rotate(-90 14 ${{h/2}})">HR (BPM)</text>
    <path d="${{ecgAllPath}}" fill="none" stroke="#9ca3af" stroke-width="1.4" opacity="0.6"/>
    <path d="${{ecgSelPath}}" fill="none" stroke="#2563eb" stroke-width="2.0"/>
    <path d="${{rppgSelPath}}" fill="none" stroke="#ea580c" stroke-width="2.0"/>
  `;

  const sampleObj = SAMPLE_STATS.find(x => x.sample === sample);
  const st = sampleObj && sampleObj.stats ? sampleObj.stats[thr] : null;
  const nAll = sampleObj ? sampleObj.n_all : sec.length;
  meta.textContent = `sample=${{sample}} | threshold=${{thr}} | n_used=${{st ? st.n : idx.length}}/${{nAll}} | coverage=${{st ? fmtPct(st.coverage) : ''}} | MAE=${{st ? fmt(st.mae) : ''}} | RMSE=${{st ? fmt(st.rmse) : ''}}`;
}}

renderHrTable();
renderWarmup();
renderTimeToKTable();
renderGeneralization();
renderLagAnalysis();
filterMatrixRows();
(() => {{
  const first = document.getElementById("plotSample");
  if (first && first.options.length > 1) {{
    first.selectedIndex = 1;
  }}
  syncPlotThreshold();
}})();
</script>
</body>
</html>
"""


def main() -> None:
    args = _parse_args()
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    if not thresholds:
        thresholds = DEFAULT_THRESHOLDS

    comparison_csv = Path(args.comparison_csv)
    matrix_csv_out = Path(args.matrix_csv_out)
    matrix_long_csv_out = Path(args.matrix_long_csv_out)
    per_file_hr_csv_out = Path(args.per_file_hr_csv_out)
    warmup_csv_out = Path(args.warmup_csv_out)
    time_to_k_csv_out = Path(args.time_to_k_csv_out)
    html_out = Path(args.html_out)
    generalization_domain_csv = Path(args.generalization_domain_csv)
    generalization_group_csv = Path(args.generalization_group_csv)
    generalization_delta_csv = Path(args.generalization_delta_csv)
    generalization_sample_csv = Path(args.generalization_sample_csv)
    generalization_report_md = Path(args.generalization_report_md)
    lag_curve_baseline_csv = Path(args.lag_curve_baseline_csv)
    lag_curve_current_csv = Path(args.lag_curve_current_csv)
    lag_sample_baseline_csv = Path(args.lag_sample_baseline_csv)
    lag_sample_current_csv = Path(args.lag_sample_current_csv)
    lag_commonwindow_sample_csv = Path(args.lag_commonwindow_sample_csv)
    lag_diffcorr_sample_csv = Path(args.lag_diffcorr_sample_csv)
    lag_global_diff_csv = Path(args.lag_global_diff_csv)

    raw = pd.read_csv(comparison_csv)
    raw["sample"] = raw.apply(_sample_from_row, axis=1)
    for col in ["sec", "ecg_hr", "hr_best", "frequency_confidence"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    df = raw.dropna(subset=["sample", "sec", "ecg_hr", "hr_best", "frequency_confidence"]).copy()
    df = df[df["sample"] != ""].copy()
    df["sec"] = df["sec"].astype(int)

    stats_long = _compute_stats_table(df, thresholds)
    matrix_wide = _build_wide_matrix(stats_long, thresholds)
    overall_rows = _compute_overall_rows(df, thresholds)
    warmup_rows = _compute_warmup_rows(df, WARMUP_MIN_SECS, WARMUP_THRESHOLDS)
    time_to_k_rows = _compute_time_to_k_rows(df, TIME_TO_K_THRESHOLDS, TIME_TO_K_KS)

    for out in [
        matrix_csv_out,
        matrix_long_csv_out,
        per_file_hr_csv_out,
        warmup_csv_out,
        time_to_k_csv_out,
        html_out,
    ]:
        out.parent.mkdir(parents=True, exist_ok=True)

    matrix_wide.to_csv(matrix_csv_out, index=False)
    stats_long.to_csv(matrix_long_csv_out, index=False)
    stats_long.to_csv(per_file_hr_csv_out, index=False)
    pd.DataFrame.from_records(warmup_rows).to_csv(warmup_csv_out, index=False)
    pd.DataFrame.from_records(time_to_k_rows).to_csv(time_to_k_csv_out, index=False)

    sample_stats_json = _build_sample_stats_json(stats_long, thresholds)
    series_json = _build_series_json(df)
    generalization_json = _build_generalization_json(
        domain_df=_read_optional_csv(generalization_domain_csv),
        group_df=_read_optional_csv(generalization_group_csv),
        delta_df=_read_optional_csv(generalization_delta_csv),
        sample_df=_read_optional_csv(generalization_sample_csv),
        report_md=generalization_report_md,
    )
    lag_json = _build_lag_json(
        curve_baseline_df=_read_optional_csv(lag_curve_baseline_csv),
        curve_current_df=_read_optional_csv(lag_curve_current_csv),
        sample_baseline_df=_read_optional_csv(lag_sample_baseline_csv),
        sample_current_df=_read_optional_csv(lag_sample_current_csv),
        commonwindow_df=_read_optional_csv(lag_commonwindow_sample_csv),
        diffcorr_df=_read_optional_csv(lag_diffcorr_sample_csv),
        global_diff_df=_read_optional_csv(lag_global_diff_csv),
    )
    html_text = _render_html(
        overall_rows=overall_rows,
        matrix_wide=matrix_wide,
        thresholds=thresholds,
        warmup_rows=warmup_rows,
        time_to_k_rows=time_to_k_rows,
        warmup_thresholds=WARMUP_THRESHOLDS,
        sample_stats_json=sample_stats_json,
        series_json=series_json,
        generalization_json=generalization_json,
        lag_json=lag_json,
    )
    html_out.write_text(html_text, encoding="utf-8")

    print(f"[OK] matrix wide: {matrix_csv_out}")
    print(f"[OK] matrix long: {matrix_long_csv_out}")
    print(f"[OK] per-file hr: {per_file_hr_csv_out}")
    print(f"[OK] warmup time: {warmup_csv_out}")
    print(f"[OK] time to k: {time_to_k_csv_out}")
    print(f"[OK] html report: {html_out}")


if __name__ == "__main__":
    main()
