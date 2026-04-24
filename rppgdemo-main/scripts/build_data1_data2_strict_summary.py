#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _to_float(v: object) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _metric_from_detail(rows: Iterable[Dict[str, str]]) -> Dict[str, float]:
    pairs: List[Tuple[float, float]] = []
    for r in rows:
        ecg = _to_float(r.get("ecg_hr"))
        est = _to_float(r.get("est_hr"))
        if ecg is None or est is None:
            continue
        pairs.append((ecg, est))
    n = len(pairs)
    if n == 0:
        return {
            "n": 0.0,
            "mae": math.nan,
            "rmse": math.nan,
            "mape": math.nan,
            "accuracy": math.nan,
            "corr": math.nan,
            "bias": math.nan,
        }
    errs = [e - r for r, e in pairs]
    mae = sum(abs(x) for x in errs) / n
    rmse = math.sqrt(sum(x * x for x in errs) / n)
    ape = [abs((e - r) / r) * 100.0 for r, e in pairs if abs(r) > 1e-12]
    mape = sum(ape) / len(ape) if ape else math.nan
    acc = 100.0 - mape if math.isfinite(mape) else math.nan
    bias = sum(errs) / n

    if n < 2:
        corr = math.nan
    else:
        x_mean = sum(r for r, _ in pairs) / n
        y_mean = sum(e for _, e in pairs) / n
        num = sum((r - x_mean) * (e - y_mean) for r, e in pairs)
        den_x = math.sqrt(sum((r - x_mean) ** 2 for r, _ in pairs))
        den_y = math.sqrt(sum((e - y_mean) ** 2 for _, e in pairs))
        corr = num / (den_x * den_y) if den_x > 0 and den_y > 0 else math.nan

    return {
        "n": float(n),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "accuracy": acc,
        "corr": corr,
        "bias": bias,
    }


def _key(r: Dict[str, str]) -> str:
    return f"{r.get('group','')}/{r.get('stem','')}"


def _samples_from_summary(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for r in rows:
        if (r.get("group") or "") == "ALL":
            continue
        if (r.get("stem") or "") == "ALL":
            continue
        out.append(r)
    return out


def _overall_row_from_summary(rows: Iterable[Dict[str, str]]) -> Optional[Dict[str, str]]:
    candidates = [r for r in rows if (r.get("group") or "") == "ALL" and (r.get("stem") or "") == "ALL"]
    for split_name in ("ALL", "TEST_ALL", "TRAIN_ALL", "train", "test"):
        for r in candidates:
            if (r.get("split") or "").upper() == split_name.upper():
                return r
    return candidates[0] if candidates else None


def _fmt(v: float, nd: int = 6) -> str:
    return "" if not math.isfinite(v) else f"{v:.{nd}f}"


def build_dataset_summary(
    *,
    dataset_name: str,
    best_summary: Path,
    pub_summary: Path,
    best_detail: Path,
    pub_detail: Path,
) -> Dict[str, object]:
    b_sum_rows = _read_csv(best_summary)
    p_sum_rows = _read_csv(pub_summary)
    b_det_rows = _read_csv(best_detail)
    p_det_rows = _read_csv(pub_detail)

    overall_sum_b = _overall_row_from_summary(b_sum_rows)
    overall_sum_p = _overall_row_from_summary(p_sum_rows)

    overall_b = _metric_from_detail(b_det_rows)
    overall_p = _metric_from_detail(p_det_rows)

    # per-group from details
    groups = sorted({r.get("group", "") for r in b_det_rows if r.get("group")})
    per_group: List[Dict[str, object]] = []
    for g in groups:
        b_g = _metric_from_detail(r for r in b_det_rows if (r.get("group") or "") == g)
        p_g = _metric_from_detail(r for r in p_det_rows if (r.get("group") or "") == g)
        cov = (p_g["n"] / b_g["n"]) if b_g["n"] > 0 else math.nan
        per_group.append(
            {
                "dataset": dataset_name,
                "group": g,
                "best_n": b_g["n"],
                "best_mae": b_g["mae"],
                "best_rmse": b_g["rmse"],
                "best_mape": b_g["mape"],
                "best_accuracy": b_g["accuracy"],
                "pub_n": p_g["n"],
                "pub_mae": p_g["mae"],
                "pub_rmse": p_g["rmse"],
                "pub_mape": p_g["mape"],
                "pub_accuracy": p_g["accuracy"],
                "coverage": cov,
            }
        )

    # per-sample from summary
    b_s_map = {_key(r): r for r in _samples_from_summary(b_sum_rows)}
    p_s_map = {_key(r): r for r in _samples_from_summary(p_sum_rows)}
    sample_keys = sorted(set(b_s_map.keys()) | set(p_s_map.keys()))
    per_sample: List[Dict[str, object]] = []
    for k in sample_keys:
        b = b_s_map.get(k, {})
        p = p_s_map.get(k, {})
        b_n = _to_float(b.get("hr_n")) or 0.0
        p_n = _to_float(p.get("hr_n")) or 0.0
        b_mae = _to_float(b.get("hr_mae"))
        p_mae = _to_float(p.get("hr_mae"))
        cov = (p_n / b_n) if b_n > 0 else math.nan
        per_sample.append(
            {
                "dataset": dataset_name,
                "group": k.split("/", 1)[0] if "/" in k else "",
                "stem": k.split("/", 1)[1] if "/" in k else k,
                "best_n": b_n,
                "best_mae": b_mae if b_mae is not None else math.nan,
                "best_rmse": _to_float(b.get("hr_rmse")) if _to_float(b.get("hr_rmse")) is not None else math.nan,
                "best_mape": _to_float(b.get("hr_mape")) if _to_float(b.get("hr_mape")) is not None else math.nan,
                "pub_n": p_n,
                "pub_mae": p_mae if p_mae is not None else math.nan,
                "pub_rmse": _to_float(p.get("hr_rmse")) if _to_float(p.get("hr_rmse")) is not None else math.nan,
                "pub_mape": _to_float(p.get("hr_mape")) if _to_float(p.get("hr_mape")) is not None else math.nan,
                "coverage": cov,
                "mae_delta_best_minus_pub": (
                    (b_mae - p_mae)
                    if (b_mae is not None and p_mae is not None and math.isfinite(b_mae) and math.isfinite(p_mae))
                    else math.nan
                ),
            }
        )

    overall = {
        "dataset": dataset_name,
        "best_n": overall_b["n"],
        "best_mae": overall_b["mae"],
        "best_rmse": overall_b["rmse"],
        "best_mape": overall_b["mape"],
        "best_accuracy": overall_b["accuracy"],
        "best_corr": overall_b["corr"],
        "pub_n": overall_p["n"],
        "pub_mae": overall_p["mae"],
        "pub_rmse": overall_p["rmse"],
        "pub_mape": overall_p["mape"],
        "pub_accuracy": overall_p["accuracy"],
        "pub_corr": overall_p["corr"],
        "coverage": (overall_p["n"] / overall_b["n"]) if overall_b["n"] > 0 else math.nan,
        "summary_best_n": (_to_float(overall_sum_b.get("hr_n")) if overall_sum_b else math.nan),
        "summary_pub_n": (_to_float(overall_sum_p.get("hr_n")) if overall_sum_p else math.nan),
        "summary_best_mae": (_to_float(overall_sum_b.get("hr_mae")) if overall_sum_b else math.nan),
        "summary_pub_mae": (_to_float(overall_sum_p.get("hr_mae")) if overall_sum_p else math.nan),
    }
    return {
        "overall": overall,
        "per_group": per_group,
        "per_sample": per_sample,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build strict rerun summary for Data1/Data2")
    ap.add_argument(
        "--data1-dir",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/strict_20260424_data1",
    )
    ap.add_argument(
        "--data2-dir",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/strict_20260424_data2_noiphone13",
    )
    ap.add_argument(
        "--out-dir",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/strict_20260424_analysis",
    )
    args = ap.parse_args()

    data1_dir = Path(args.data1_dir)
    data2_dir = Path(args.data2_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def p(root: Path, name: str) -> Path:
        path = root / name
        if not path.exists():
            raise RuntimeError(f"missing file: {path}")
        return path

    data1 = build_dataset_summary(
        dataset_name="Data1",
        best_summary=p(data1_dir, "rppg_ecg_summary_best_opencv_timestamp.csv"),
        pub_summary=p(data1_dir, "rppg_ecg_summary_published_opencv_timestamp.csv"),
        best_detail=p(data1_dir, "rppg_ecg_comparison_best_opencv_timestamp.csv"),
        pub_detail=p(data1_dir, "rppg_ecg_comparison_published_opencv_timestamp.csv"),
    )
    data2 = build_dataset_summary(
        dataset_name="Data2_no_iPhone13",
        best_summary=p(data2_dir, "rppg_ecg_summary_best_opencv_timestamp.csv"),
        pub_summary=p(data2_dir, "rppg_ecg_summary_published_opencv_timestamp.csv"),
        best_detail=p(data2_dir, "rppg_ecg_comparison_best_opencv_timestamp.csv"),
        pub_detail=p(data2_dir, "rppg_ecg_comparison_published_opencv_timestamp.csv"),
    )

    overall_rows = [data1["overall"], data2["overall"]]
    group_rows = list(data1["per_group"]) + list(data2["per_group"])
    sample_rows = list(data1["per_sample"]) + list(data2["per_sample"])

    overall_csv = out_dir / "data1_data2_overall.csv"
    with overall_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "best_n",
                "best_mae",
                "best_rmse",
                "best_mape",
                "best_accuracy",
                "best_corr",
                "pub_n",
                "pub_mae",
                "pub_rmse",
                "pub_mape",
                "pub_accuracy",
                "pub_corr",
                "coverage",
                "summary_best_n",
                "summary_pub_n",
                "summary_best_mae",
                "summary_pub_mae",
            ]
        )
        for r in overall_rows:
            w.writerow(
                [
                    r["dataset"],
                    int(r["best_n"]),
                    _fmt(r["best_mae"]),
                    _fmt(r["best_rmse"]),
                    _fmt(r["best_mape"]),
                    _fmt(r["best_accuracy"]),
                    _fmt(r["best_corr"]),
                    int(r["pub_n"]),
                    _fmt(r["pub_mae"]),
                    _fmt(r["pub_rmse"]),
                    _fmt(r["pub_mape"]),
                    _fmt(r["pub_accuracy"]),
                    _fmt(r["pub_corr"]),
                    _fmt(r["coverage"]),
                    _fmt(r["summary_best_n"]),
                    _fmt(r["summary_pub_n"]),
                    _fmt(r["summary_best_mae"]),
                    _fmt(r["summary_pub_mae"]),
                ]
            )

    per_group_csv = out_dir / "data1_data2_per_group.csv"
    with per_group_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "group",
                "best_n",
                "best_mae",
                "best_rmse",
                "best_mape",
                "best_accuracy",
                "pub_n",
                "pub_mae",
                "pub_rmse",
                "pub_mape",
                "pub_accuracy",
                "coverage",
            ]
        )
        for r in group_rows:
            w.writerow(
                [
                    r["dataset"],
                    r["group"],
                    int(r["best_n"]),
                    _fmt(r["best_mae"]),
                    _fmt(r["best_rmse"]),
                    _fmt(r["best_mape"]),
                    _fmt(r["best_accuracy"]),
                    int(r["pub_n"]),
                    _fmt(r["pub_mae"]),
                    _fmt(r["pub_rmse"]),
                    _fmt(r["pub_mape"]),
                    _fmt(r["pub_accuracy"]),
                    _fmt(r["coverage"]),
                ]
            )

    per_sample_csv = out_dir / "data1_data2_per_sample.csv"
    with per_sample_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "group",
                "stem",
                "best_n",
                "best_mae",
                "best_rmse",
                "best_mape",
                "pub_n",
                "pub_mae",
                "pub_rmse",
                "pub_mape",
                "coverage",
                "mae_delta_best_minus_pub",
            ]
        )
        for r in sample_rows:
            w.writerow(
                [
                    r["dataset"],
                    r["group"],
                    r["stem"],
                    int(r["best_n"]),
                    _fmt(r["best_mae"]),
                    _fmt(r["best_rmse"]),
                    _fmt(r["best_mape"]),
                    int(r["pub_n"]),
                    _fmt(r["pub_mae"]),
                    _fmt(r["pub_rmse"]),
                    _fmt(r["pub_mape"]),
                    _fmt(r["coverage"]),
                    _fmt(r["mae_delta_best_minus_pub"]),
                ]
            )

    # human-readable analysis summary
    md = out_dir / "data1_data2_analysis.md"
    d1 = data1["overall"]
    d2 = data2["overall"]
    md.write_text(
        "\n".join(
            [
                "# Data1 / Data2 Strict Rerun Analysis",
                "",
                "## Overall (from detail recomputation)",
                "",
                f"- Data1 best: n={int(d1['best_n'])}, MAE={_fmt(d1['best_mae'])}, RMSE={_fmt(d1['best_rmse'])}, MAPE={_fmt(d1['best_mape'])}%, Acc={_fmt(d1['best_accuracy'])}%, corr={_fmt(d1['best_corr'])}",
                f"- Data1 published: n={int(d1['pub_n'])}, MAE={_fmt(d1['pub_mae'])}, RMSE={_fmt(d1['pub_rmse'])}, MAPE={_fmt(d1['pub_mape'])}%, Acc={_fmt(d1['pub_accuracy'])}%, corr={_fmt(d1['pub_corr'])}, coverage={_fmt(d1['coverage'])}",
                "",
                f"- Data2(no iPhone13) best: n={int(d2['best_n'])}, MAE={_fmt(d2['best_mae'])}, RMSE={_fmt(d2['best_rmse'])}, MAPE={_fmt(d2['best_mape'])}%, Acc={_fmt(d2['best_accuracy'])}%, corr={_fmt(d2['best_corr'])}",
                f"- Data2(no iPhone13) published: n={int(d2['pub_n'])}, MAE={_fmt(d2['pub_mae'])}, RMSE={_fmt(d2['pub_rmse'])}, MAPE={_fmt(d2['pub_mape'])}%, Acc={_fmt(d2['pub_accuracy'])}%, corr={_fmt(d2['pub_corr'])}, coverage={_fmt(d2['coverage'])}",
                "",
                "## Files",
                "",
                f"- overall csv: {overall_csv}",
                f"- per-group csv: {per_group_csv}",
                f"- per-sample csv: {per_sample_csv}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[OK] overall: {overall_csv}")
    print(f"[OK] per_group: {per_group_csv}")
    print(f"[OK] per_sample: {per_sample_csv}")
    print(f"[OK] md: {md}")


if __name__ == "__main__":
    main()
