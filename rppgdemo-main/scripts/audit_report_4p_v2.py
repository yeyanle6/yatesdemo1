#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


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


def _fmt(v: Optional[float], nd: int = 3) -> str:
    if v is None or not math.isfinite(v):
        return "-"
    return f"{v:.{nd}f}"


def _pct(v: Optional[float], nd: int = 1) -> str:
    if v is None or not math.isfinite(v):
        return "-"
    return f"{v * 100:.{nd}f}%"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@dataclass
class Metric:
    n: int
    mae: float
    rmse: float
    mape: float
    accuracy: float
    corr: float
    ecg_mean: float
    est_mean: float
    bias: float


def _compute_metric(rows: Sequence[Dict[str, str]]) -> Metric:
    pairs: List[Tuple[float, float]] = []
    for r in rows:
        ecg = _to_float(r.get("ecg_hr"))
        est = _to_float(r.get("est_hr"))
        if ecg is None or est is None:
            continue
        pairs.append((ecg, est))

    if not pairs:
        return Metric(
            n=0,
            mae=math.nan,
            rmse=math.nan,
            mape=math.nan,
            accuracy=math.nan,
            corr=math.nan,
            ecg_mean=math.nan,
            est_mean=math.nan,
            bias=math.nan,
        )

    ecg_vals = [p[0] for p in pairs]
    est_vals = [p[1] for p in pairs]
    errs = [e - r for r, e in pairs]
    abs_err = [abs(x) for x in errs]
    sq_err = [x * x for x in errs]
    ape = [abs((e - r) / r) * 100.0 for r, e in pairs if abs(r) > 1e-12]

    n = len(pairs)
    mae = sum(abs_err) / n
    rmse = math.sqrt(sum(sq_err) / n)
    mape = sum(ape) / len(ape) if ape else math.nan
    acc = 100.0 - mape if math.isfinite(mape) else math.nan
    ecg_mean = sum(ecg_vals) / n
    est_mean = sum(est_vals) / n
    bias = sum(errs) / n

    # Pearson correlation
    if n < 2:
        corr = math.nan
    else:
        xm = ecg_mean
        ym = est_mean
        num = sum((x - xm) * (y - ym) for x, y in pairs)
        den_x = math.sqrt(sum((x - xm) ** 2 for x in ecg_vals))
        den_y = math.sqrt(sum((y - ym) ** 2 for y in est_vals))
        corr = num / (den_x * den_y) if den_x > 0 and den_y > 0 else math.nan

    return Metric(
        n=n,
        mae=mae,
        rmse=rmse,
        mape=mape,
        accuracy=acc,
        corr=corr,
        ecg_mean=ecg_mean,
        est_mean=est_mean,
        bias=bias,
    )


@dataclass
class Claim:
    name: str
    value: Optional[float]
    unit: str


def _extract_report_claims(report_html: str) -> Dict[str, Claim]:
    claims: Dict[str, Claim] = {}

    # Slide 3 KPI cards (order fixed in template)
    kpi_vals = re.findall(
        r'<div class="kpi-card[^"]*">\s*<div class="label">([^<]+)</div>\s*<div class="val">([0-9.]+)',
        report_html,
        flags=re.S,
    )
    for label, value in kpi_vals:
        key = label.strip()
        claims[f"kpi:{key}"] = Claim(name=key, value=_to_float(value), unit="")

    # Data2 sample seconds in slide 2
    m_data2_n = re.search(r"best\s+([0-9]+)\s*秒\s*/\s*published\s+([0-9]+)\s*秒", report_html)
    if m_data2_n:
        claims["data2:best_n"] = Claim("Data2 best n", float(m_data2_n.group(1)), "sec")
        claims["data2:pub_n"] = Claim("Data2 published n", float(m_data2_n.group(2)), "sec")

    # Device rows in slide 3 table
    def _extract_device_metrics(name_pat: str) -> Optional[Tuple[float, float, float, float, float]]:
        row_m = re.search(rf"<tr>\s*<td class=\"name\">[^<]*{name_pat}.*?</tr>", report_html, flags=re.S)
        if not row_m:
            return None
        row = row_m.group(0)
        vals = re.findall(r"<td class=\"c-pub\">([0-9.]+)%?</td>", row)
        if len(vals) < 5:
            return None
        return (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4]))

    lenovo_vals = _extract_device_metrics("Lenovo")
    if lenovo_vals:
        claims["dev:g103:files"] = Claim("Lenovo files", lenovo_vals[0], "count")
        claims["dev:g103:mae"] = Claim("Lenovo MAE", lenovo_vals[1], "bpm")
        claims["dev:g103:rmse"] = Claim("Lenovo RMSE", lenovo_vals[2], "bpm")
        claims["dev:g103:mape"] = Claim("Lenovo MAPE", lenovo_vals[3], "%")
        claims["dev:g103:coverage"] = Claim("Lenovo coverage", lenovo_vals[4], "%")

    iphone_vals = _extract_device_metrics("iPhone\\s*16e")
    if iphone_vals:
        claims["dev:g102:files"] = Claim("iPhone16e files", iphone_vals[0], "count")
        claims["dev:g102:mae"] = Claim("iPhone16e MAE", iphone_vals[1], "bpm")
        claims["dev:g102:rmse"] = Claim("iPhone16e RMSE", iphone_vals[2], "bpm")
        claims["dev:g102:mape"] = Claim("iPhone16e MAPE", iphone_vals[3], "%")
        claims["dev:g102:coverage"] = Claim("iPhone16e coverage", iphone_vals[4], "%")

    m_improve_count = re.search(r"([0-9]+)\s*本中\s*([0-9]+)\s*本", report_html)
    if m_improve_count:
        claims["data2:improve_total"] = Claim("Compared files", float(m_improve_count.group(1)), "count")
        claims["data2:improve_hit"] = Claim("Published better files", float(m_improve_count.group(2)), "count")

    # Data1 claims in slide1
    m_data1_n = re.search(r"サンプル総数</th><td class=\"num\">([0-9,]+)\s*秒", report_html)
    if m_data1_n:
        claims["data1:n"] = Claim("Data1 total seconds", float(m_data1_n.group(1).replace(",", "")), "sec")
    m_fc88 = re.search(
        r"公開閾値 \(fc ≥ 0\.88\)</th><td class=\"num\">MAE ([0-9.]+) bpm .* Accuracy ([0-9.]+)%",
        report_html,
    )
    if m_fc88:
        claims["data1:fc88:mae"] = Claim("Data1 fc>=0.88 MAE", float(m_fc88.group(1)), "bpm")
        claims["data1:fc88:acc"] = Claim("Data1 fc>=0.88 Accuracy", float(m_fc88.group(2)), "%")
    m_cov = re.search(r"カバー率</th><td class=\"num\">([0-9.]+)% \(([0-9,]+)\s*秒\)", report_html)
    if m_cov:
        claims["data1:fc88:coverage"] = Claim("Data1 fc>=0.88 coverage", float(m_cov.group(1)), "%")
        claims["data1:fc88:n"] = Claim("Data1 fc>=0.88 n", float(m_cov.group(2).replace(",", "")), "sec")
    m_fc94 = re.search(
        r"最良閾値 \(fc ≥ 0\.94\)</th><td class=\"num\">MAE ([0-9.]+) bpm .* Accuracy ([0-9.]+)%",
        report_html,
    )
    if m_fc94:
        claims["data1:fc94:mae"] = Claim("Data1 fc>=0.94 MAE", float(m_fc94.group(1)), "bpm")
        claims["data1:fc94:acc"] = Claim("Data1 fc>=0.94 Accuracy", float(m_fc94.group(2)), "%")

    return claims


def _parse_fc_threshold_rows(fc_html: str, threshold: str) -> Optional[Dict[str, float]]:
    # Row shape:
    # <tr><td>0.88</td><td>2818</td><td>69.9%</td><td>3.036</td>...<td>95.595%</td>...
    pattern = rf"<tr><td>{re.escape(threshold)}(?:\s*<span[^>]*>.*?</span>)?</td><td>([0-9]+)</td><td>([0-9.]+)%</td><td>([0-9.]+)</td><td>([0-9.]+)</td><td>([0-9.]+)</td><td>([0-9.]+)%</td>"
    m = re.search(pattern, fc_html)
    if not m:
        return None
    return {
        "n": float(m.group(1)),
        "coverage_pct": float(m.group(2)),
        "mae": float(m.group(3)),
        "rmse": float(m.group(4)),
        "mape_pct": float(m.group(5)),
        "accuracy_pct": float(m.group(6)),
    }


def _filter_rows(rows: Iterable[Dict[str, str]], groups: Sequence[str]) -> List[Dict[str, str]]:
    gs = set(groups)
    return [r for r in rows if (r.get("group") or "") in gs]


def _sample_metrics(summary_rows: Sequence[Dict[str, str]], groups: Sequence[str]) -> Dict[Tuple[str, str], Dict[str, float]]:
    gs = set(groups)
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in summary_rows:
        g = r.get("group") or ""
        s = r.get("stem") or ""
        if g not in gs:
            continue
        if g == "ALL" or s == "ALL":
            continue
        out[(g, s)] = {
            "n": _to_float(r.get("hr_n")) or 0.0,
            "mae": _to_float(r.get("hr_mae")) if _to_float(r.get("hr_mae")) is not None else math.nan,
            "rmse": _to_float(r.get("hr_rmse")) if _to_float(r.get("hr_rmse")) is not None else math.nan,
            "mape": _to_float(r.get("hr_mape")) if _to_float(r.get("hr_mape")) is not None else math.nan,
            "corr": _to_float(r.get("hr_corr")) if _to_float(r.get("hr_corr")) is not None else math.nan,
        }
    return out


def _metric_from_summary_weighted(summary_rows: Sequence[Dict[str, str]], groups: Sequence[str]) -> Metric:
    by_sample = _sample_metrics(summary_rows, groups)
    n_total = int(sum(v["n"] for v in by_sample.values()))
    if n_total <= 0:
        return Metric(0, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)

    mae = sum(v["mae"] * v["n"] for v in by_sample.values() if math.isfinite(v["mae"])) / n_total
    rmse_sq = sum((v["rmse"] ** 2) * v["n"] for v in by_sample.values() if math.isfinite(v["rmse"])) / n_total
    rmse = math.sqrt(rmse_sq) if rmse_sq >= 0 else math.nan
    mape = sum(v["mape"] * v["n"] for v in by_sample.values() if math.isfinite(v["mape"])) / n_total
    # corr is not weighted in a valid statistical way across files; keep NaN here on purpose.
    return Metric(
        n=n_total,
        mae=mae,
        rmse=rmse,
        mape=mape,
        accuracy=100.0 - mape if math.isfinite(mape) else math.nan,
        corr=math.nan,
        ecg_mean=math.nan,
        est_mean=math.nan,
        bias=math.nan,
    )


def _safe_diff(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None:
        return math.inf
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    return abs(a - b)


def _truth_checks(
    claims: Dict[str, Claim],
    data2_best: Metric,
    data2_pub: Metric,
    data2_cov: float,
    group_best: Dict[str, Metric],
    group_pub: Dict[str, Metric],
    group_cov: Dict[str, float],
    improve_hit: int,
    improve_total: int,
    data1_fc88: Dict[str, float],
    data1_fc94: Dict[str, float],
) -> List[Dict[str, object]]:
    checks: List[Dict[str, object]] = []

    def add(name: str, claim: Optional[float], truth: Optional[float], tol: float, unit: str) -> None:
        diff = _safe_diff(claim, truth)
        ok = diff <= tol
        checks.append(
            {
                "name": name,
                "claim": claim,
                "truth": truth,
                "diff": diff,
                "tol": tol,
                "unit": unit,
                "ok": ok,
            }
        )

    # Data2 overall claims
    add("Data2 published MAE", claims.get("kpi:HR MAE").value if claims.get("kpi:HR MAE") else None, data2_pub.mae, 0.05, "bpm")
    add("Data2 published RMSE", claims.get("kpi:HR RMSE").value if claims.get("kpi:HR RMSE") else None, data2_pub.rmse, 0.05, "bpm")
    add("Data2 published Accuracy", claims.get("kpi:Accuracy (100 − MAPE)").value if claims.get("kpi:Accuracy (100 − MAPE)") else None, data2_pub.accuracy, 0.15, "%")
    cov_claim = claims.get("kpi:カバー率 (published/best)")
    add("Data2 coverage", cov_claim.value if cov_claim else None, data2_cov * 100.0, 0.2, "%")
    add("Data2 best n", claims.get("data2:best_n").value if claims.get("data2:best_n") else None, float(data2_best.n), 0.0, "sec")
    add("Data2 published n", claims.get("data2:pub_n").value if claims.get("data2:pub_n") else None, float(data2_pub.n), 0.0, "sec")

    # Device claims
    add("g103 MAE", claims.get("dev:g103:mae").value if claims.get("dev:g103:mae") else None, group_pub["103"].mae, 0.05, "bpm")
    add("g103 RMSE", claims.get("dev:g103:rmse").value if claims.get("dev:g103:rmse") else None, group_pub["103"].rmse, 0.05, "bpm")
    add("g103 MAPE", claims.get("dev:g103:mape").value if claims.get("dev:g103:mape") else None, group_pub["103"].mape, 0.15, "%")
    add("g103 coverage", claims.get("dev:g103:coverage").value if claims.get("dev:g103:coverage") else None, group_cov["103"] * 100.0, 0.2, "%")

    add("g102 MAE", claims.get("dev:g102:mae").value if claims.get("dev:g102:mae") else None, group_pub["102"].mae, 0.05, "bpm")
    add("g102 RMSE", claims.get("dev:g102:rmse").value if claims.get("dev:g102:rmse") else None, group_pub["102"].rmse, 0.05, "bpm")
    add("g102 MAPE", claims.get("dev:g102:mape").value if claims.get("dev:g102:mape") else None, group_pub["102"].mape, 0.15, "%")
    add("g102 coverage", claims.get("dev:g102:coverage").value if claims.get("dev:g102:coverage") else None, group_cov["102"] * 100.0, 0.2, "%")

    add("Improved files count", claims.get("data2:improve_hit").value if claims.get("data2:improve_hit") else None, float(improve_hit), 0.0, "count")
    add("Compared files count", claims.get("data2:improve_total").value if claims.get("data2:improve_total") else None, float(improve_total), 0.0, "count")

    # Data1 references
    add("Data1 total seconds", claims.get("data1:n").value if claims.get("data1:n") else None, 3302.0, 0.0, "sec")
    add("Data1 fc0.88 MAE", claims.get("data1:fc88:mae").value if claims.get("data1:fc88:mae") else None, data1_fc88.get("mae"), 0.05, "bpm")
    add("Data1 fc0.88 Accuracy", claims.get("data1:fc88:acc").value if claims.get("data1:fc88:acc") else None, data1_fc88.get("accuracy_pct"), 0.15, "%")
    add("Data1 fc0.88 coverage", claims.get("data1:fc88:coverage").value if claims.get("data1:fc88:coverage") else None, data1_fc88.get("coverage_pct"), 0.2, "%")
    add("Data1 fc0.88 n", claims.get("data1:fc88:n").value if claims.get("data1:fc88:n") else None, data1_fc88.get("n"), 0.0, "sec")
    add("Data1 fc0.94 MAE", claims.get("data1:fc94:mae").value if claims.get("data1:fc94:mae") else None, data1_fc94.get("mae"), 0.05, "bpm")
    add("Data1 fc0.94 Accuracy", claims.get("data1:fc94:acc").value if claims.get("data1:fc94:acc") else None, data1_fc94.get("accuracy_pct"), 0.15, "%")

    return checks


def _add_internal_consistency_checks(
    checks: List[Dict[str, object]],
    *,
    detail_scope_best: Metric,
    detail_scope_pub: Metric,
    detail_all_best: Metric,
    detail_all_pub: Metric,
    summary_scope_best: Metric,
    summary_scope_pub: Metric,
    summary_all_best: Metric,
    summary_all_pub: Metric,
) -> None:
    def add(name: str, a: float, b: float, tol: float, unit: str) -> None:
        diff = _safe_diff(a, b)
        checks.append(
            {
                "name": name,
                "claim": a,
                "truth": b,
                "diff": diff,
                "tol": tol,
                "unit": unit,
                "ok": diff <= tol,
            }
        )

    # Cross-check detail aggregation vs summary weighted aggregation.
    add("Consistency scope best n", float(detail_scope_best.n), float(summary_scope_best.n), 0.0, "sec")
    add("Consistency scope pub n", float(detail_scope_pub.n), float(summary_scope_pub.n), 0.0, "sec")
    add("Consistency scope best MAE", detail_scope_best.mae, summary_scope_best.mae, 1e-6, "bpm")
    add("Consistency scope pub MAE", detail_scope_pub.mae, summary_scope_pub.mae, 1e-6, "bpm")
    add("Consistency scope best RMSE", detail_scope_best.rmse, summary_scope_best.rmse, 1e-6, "bpm")
    add("Consistency scope pub RMSE", detail_scope_pub.rmse, summary_scope_pub.rmse, 1e-6, "bpm")
    add("Consistency scope best MAPE", detail_scope_best.mape, summary_scope_best.mape, 1e-6, "%")
    add("Consistency scope pub MAPE", detail_scope_pub.mape, summary_scope_pub.mape, 1e-6, "%")

    add("Consistency all best n", float(detail_all_best.n), float(summary_all_best.n), 0.0, "sec")
    add("Consistency all pub n", float(detail_all_pub.n), float(summary_all_pub.n), 0.0, "sec")
    add("Consistency all best MAE", detail_all_best.mae, summary_all_best.mae, 1e-6, "bpm")
    add("Consistency all pub MAE", detail_all_pub.mae, summary_all_pub.mae, 1e-6, "bpm")
    add("Consistency all best RMSE", detail_all_best.rmse, summary_all_best.rmse, 1e-6, "bpm")
    add("Consistency all pub RMSE", detail_all_pub.rmse, summary_all_pub.rmse, 1e-6, "bpm")
    add("Consistency all best MAPE", detail_all_best.mape, summary_all_best.mape, 1e-6, "%")
    add("Consistency all pub MAPE", detail_all_pub.mape, summary_all_pub.mape, 1e-6, "%")


def _build_html(
    *,
    checks: Sequence[Dict[str, object]],
    scope_metrics: Dict[str, Metric],
    all_metrics: Dict[str, Metric],
    scope_coverage: float,
    all_coverage: float,
    sample_rows: Sequence[Dict[str, object]],
    source_files: Sequence[Path],
) -> str:
    ok_count = sum(1 for c in checks if c["ok"])
    total_count = len(checks)
    verdict = "PASS" if ok_count == total_count else "PARTIAL"

    src_rows = []
    for p in source_files:
        if not p.exists():
            continue
        src_rows.append(
            "<tr>"
            f"<td>{html.escape(str(p))}</td>"
            f"<td>{p.stat().st_size}</td>"
            f"<td>{datetime.fromtimestamp(p.stat().st_mtime).isoformat(sep=' ', timespec='seconds')}</td>"
            f"<td><code>{_sha256(p)[:16]}...</code></td>"
            "</tr>"
        )

    check_rows = []
    for c in checks:
        cls = "ok" if c["ok"] else "bad"
        claim = c["claim"]
        truth = c["truth"]
        diff = c["diff"]
        check_rows.append(
            "<tr>"
            f"<td>{html.escape(str(c['name']))}</td>"
            f"<td>{_fmt(claim, 3) if isinstance(claim, float) else '-'}</td>"
            f"<td>{_fmt(truth, 3) if isinstance(truth, float) else '-'}</td>"
            f"<td>{_fmt(diff, 3) if isinstance(diff, float) and math.isfinite(diff) else '-'}</td>"
            f"<td>{_fmt(float(c['tol']), 3)}</td>"
            f"<td>{html.escape(str(c['unit']))}</td>"
            f"<td class=\"{cls}\">{'OK' if c['ok'] else 'MISMATCH'}</td>"
            "</tr>"
        )

    def metric_block(title: str, m_best: Metric, m_pub: Metric, cov: float) -> str:
        return (
            "<div class='metric'>"
            f"<h3>{html.escape(title)}</h3>"
            "<table><thead><tr><th>channel</th><th>n</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>Accuracy</th><th>corr</th><th>bias</th></tr></thead><tbody>"
            f"<tr><td>best</td><td>{m_best.n}</td><td>{_fmt(m_best.mae,3)}</td><td>{_fmt(m_best.rmse,3)}</td><td>{_fmt(m_best.mape,3)}%</td><td>{_fmt(m_best.accuracy,3)}%</td><td>{_fmt(m_best.corr,3)}</td><td>{_fmt(m_best.bias,3)}</td></tr>"
            f"<tr><td>published</td><td>{m_pub.n}</td><td>{_fmt(m_pub.mae,3)}</td><td>{_fmt(m_pub.rmse,3)}</td><td>{_fmt(m_pub.mape,3)}%</td><td>{_fmt(m_pub.accuracy,3)}%</td><td>{_fmt(m_pub.corr,3)}</td><td>{_fmt(m_pub.bias,3)}</td></tr>"
            f"<tr><td colspan='8'>coverage: {_pct(cov,1)} ({m_pub.n}/{m_best.n})</td></tr>"
            "</tbody></table></div>"
        )

    sample_table_rows = []
    for r in sample_rows:
        cls = "ok" if r["improved"] else "bad"
        sample_table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(r['sample']))}</td>"
            f"<td>{int(r['best_n'])}</td>"
            f"<td>{_fmt(r['best_mae'],3)}</td>"
            f"<td>{int(r['pub_n'])}</td>"
            f"<td>{_fmt(r['pub_mae'],3)}</td>"
            f"<td>{_fmt(r['delta_mae'],3)}</td>"
            f"<td class='{cls}'>{'YES' if r['improved'] else 'NO'}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>rPPG 4p_v2 数据真实性二次审计</title>
  <style>
    body {{ font-family: "Avenir Next","PingFang SC","Noto Sans CJK JP",sans-serif; background:#f4f7f9; color:#0f172a; margin:0; }}
    .wrap {{ max-width:1300px; margin:20px auto; padding:0 16px 28px; }}
    .panel {{ background:#fff; border:1px solid #dbe5ea; border-radius:12px; padding:14px; margin-bottom:14px; }}
    h1 {{ margin:0 0 8px; font-size:24px; }}
    h2 {{ margin:0 0 8px; font-size:18px; }}
    h3 {{ margin:0 0 6px; font-size:15px; }}
    .kpi {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px; }}
    .k {{ border:1px solid #d8e3e9; border-radius:10px; padding:10px; background:#fbfefe; }}
    .k .v {{ font-size:22px; font-weight:700; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; }}
    th,td {{ border:1px solid #e2eaee; padding:6px 8px; text-align:right; white-space:nowrap; }}
    th:first-child, td:first-child {{ text-align:left; }}
    thead th {{ background:#f2f8fa; }}
    .ok {{ color:#0f766e; font-weight:700; }}
    .bad {{ color:#b91c1c; font-weight:700; }}
    .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
    code {{ background:#f3f6f8; padding:1px 5px; border-radius:6px; }}
    @media (max-width:980px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h1>rPPG 4p_v2 数据真实性二次审计</h1>
      <div>审计时间: {datetime.now().isoformat(sep=' ', timespec='seconds')}</div>
      <div>结论: <span class="{'ok' if verdict == 'PASS' else 'bad'}">{verdict}</span> ({ok_count}/{total_count} checks passed)</div>
    </div>

    <div class="panel">
      <h2>输入文件快照</h2>
      <table>
        <thead><tr><th>path</th><th>size(bytes)</th><th>mtime</th><th>sha256</th></tr></thead>
        <tbody>{''.join(src_rows)}</tbody>
      </table>
    </div>

    <div class="panel">
      <h2>核心复算结果</h2>
      <div class="grid2">
        {metric_block("报告口径 (Data2: g102+g103)", scope_metrics['best'], scope_metrics['pub'], scope_coverage)}
        {metric_block("全量口径 (Data2: g101+g102+g103)", all_metrics['best'], all_metrics['pub'], all_coverage)}
      </div>
    </div>

    <div class="panel">
      <h2>声明值 vs 复算值</h2>
      <table>
        <thead><tr><th>item</th><th>claim</th><th>truth</th><th>|diff|</th><th>tol</th><th>unit</th><th>status</th></tr></thead>
        <tbody>{''.join(check_rows)}</tbody>
      </table>
    </div>

    <div class="panel">
      <h2>逐文件二次核验 (g102+g103)</h2>
      <table>
        <thead><tr><th>sample</th><th>best n</th><th>best MAE</th><th>pub n</th><th>pub MAE</th><th>best-pub</th><th>pub better?</th></tr></thead>
        <tbody>{''.join(sample_table_rows)}</tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit 4p_v2 report claims against raw CSV data.")
    ap.add_argument(
        "--report-html",
        default="/Users/liangwenwang/Downloads/rppg_精度向上開発_報告書_4p_v2.html",
    )
    ap.add_argument(
        "--fc-threshold-html",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/visualizations/fc_threshold_report.html",
    )
    ap.add_argument(
        "--best-summary-csv",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/data2_as_data1_eval_20260423_rerun/rppg_ecg_summary_best_opencv_timestamp.csv",
    )
    ap.add_argument(
        "--pub-summary-csv",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/data2_as_data1_eval_20260423_rerun/rppg_ecg_summary_published_opencv_timestamp.csv",
    )
    ap.add_argument(
        "--best-detail-csv",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/data2_as_data1_eval_20260423_rerun/rppg_ecg_comparison_best_opencv_timestamp.csv",
    )
    ap.add_argument(
        "--pub-detail-csv",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/data2_as_data1_eval_20260423_rerun/rppg_ecg_comparison_published_opencv_timestamp.csv",
    )
    ap.add_argument(
        "--out-html",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/visualizations/report_4p_v2_truth_audit.html",
    )
    ap.add_argument(
        "--out-csv",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/visualizations/report_4p_v2_truth_checks.csv",
    )
    args = ap.parse_args()

    report_html_path = Path(args.report_html)
    fc_html_path = Path(args.fc_threshold_html)
    best_summary_path = Path(args.best_summary_csv)
    pub_summary_path = Path(args.pub_summary_csv)
    best_detail_path = Path(args.best_detail_csv)
    pub_detail_path = Path(args.pub_detail_csv)

    for p in [report_html_path, fc_html_path, best_summary_path, pub_summary_path, best_detail_path, pub_detail_path]:
        if not p.exists():
            raise RuntimeError(f"missing file: {p}")

    report_html = report_html_path.read_text(encoding="utf-8")
    fc_html = fc_html_path.read_text(encoding="utf-8")

    claims = _extract_report_claims(report_html)

    best_summary = _read_csv(best_summary_path)
    pub_summary = _read_csv(pub_summary_path)
    best_detail = _read_csv(best_detail_path)
    pub_detail = _read_csv(pub_detail_path)

    scope_groups = ["102", "103"]
    all_groups = ["101", "102", "103"]

    scope_best_rows = _filter_rows(best_detail, scope_groups)
    scope_pub_rows = _filter_rows(pub_detail, scope_groups)
    all_best_rows = _filter_rows(best_detail, all_groups)
    all_pub_rows = _filter_rows(pub_detail, all_groups)

    scope_best = _compute_metric(scope_best_rows)
    scope_pub = _compute_metric(scope_pub_rows)
    all_best = _compute_metric(all_best_rows)
    all_pub = _compute_metric(all_pub_rows)

    scope_best_summary = _metric_from_summary_weighted(best_summary, scope_groups)
    scope_pub_summary = _metric_from_summary_weighted(pub_summary, scope_groups)
    all_best_summary = _metric_from_summary_weighted(best_summary, all_groups)
    all_pub_summary = _metric_from_summary_weighted(pub_summary, all_groups)

    scope_coverage = (scope_pub.n / scope_best.n) if scope_best.n > 0 else math.nan
    all_coverage = (all_pub.n / all_best.n) if all_best.n > 0 else math.nan

    group_best: Dict[str, Metric] = {}
    group_pub: Dict[str, Metric] = {}
    group_cov: Dict[str, float] = {}
    for g in scope_groups:
        gb = _compute_metric(_filter_rows(best_detail, [g]))
        gp = _compute_metric(_filter_rows(pub_detail, [g]))
        group_best[g] = gb
        group_pub[g] = gp
        group_cov[g] = (gp.n / gb.n) if gb.n > 0 else math.nan

    best_by_sample = _sample_metrics(best_summary, scope_groups)
    pub_by_sample = _sample_metrics(pub_summary, scope_groups)
    samples = sorted(set(best_by_sample.keys()) | set(pub_by_sample.keys()))

    sample_rows: List[Dict[str, object]] = []
    improve_hit = 0
    improve_total = 0
    for k in samples:
        b = best_by_sample.get(k, {})
        p = pub_by_sample.get(k, {})
        best_n = float(b.get("n", 0.0))
        pub_n = float(p.get("n", 0.0))
        best_mae = float(b.get("mae", math.nan))
        pub_mae = float(p.get("mae", math.nan))
        improved = False
        if best_n > 0 and pub_n > 0 and math.isfinite(best_mae) and math.isfinite(pub_mae):
            improve_total += 1
            improved = pub_mae < best_mae
            if improved:
                improve_hit += 1

        sample_rows.append(
            {
                "sample": f"{k[0]}/{k[1]}",
                "best_n": best_n,
                "best_mae": best_mae,
                "pub_n": pub_n,
                "pub_mae": pub_mae,
                "delta_mae": (best_mae - pub_mae) if (math.isfinite(best_mae) and math.isfinite(pub_mae)) else math.nan,
                "improved": improved,
            }
        )

    data1_fc88 = _parse_fc_threshold_rows(fc_html, "0.88") or {}
    data1_fc94 = _parse_fc_threshold_rows(fc_html, "0.94") or {}

    checks = _truth_checks(
        claims=claims,
        data2_best=scope_best,
        data2_pub=scope_pub,
        data2_cov=scope_coverage,
        group_best=group_best,
        group_pub=group_pub,
        group_cov=group_cov,
        improve_hit=improve_hit,
        improve_total=improve_total,
        data1_fc88=data1_fc88,
        data1_fc94=data1_fc94,
    )
    _add_internal_consistency_checks(
        checks,
        detail_scope_best=scope_best,
        detail_scope_pub=scope_pub,
        detail_all_best=all_best,
        detail_all_pub=all_pub,
        summary_scope_best=scope_best_summary,
        summary_scope_pub=scope_pub_summary,
        summary_all_best=all_best_summary,
        summary_all_pub=all_pub_summary,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "claim", "truth", "abs_diff", "tolerance", "unit", "status"])
        for c in checks:
            status = "OK" if c["ok"] else "MISMATCH"
            w.writerow(
                [
                    c["name"],
                    c["claim"],
                    c["truth"],
                    c["diff"] if math.isfinite(c["diff"]) else "",
                    c["tol"],
                    c["unit"],
                    status,
                ]
            )

    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    html_doc = _build_html(
        checks=checks,
        scope_metrics={"best": scope_best, "pub": scope_pub},
        all_metrics={"best": all_best, "pub": all_pub},
        scope_coverage=scope_coverage,
        all_coverage=all_coverage,
        sample_rows=sample_rows,
        source_files=[report_html_path, fc_html_path, best_summary_path, pub_summary_path, best_detail_path, pub_detail_path],
    )
    out_html.write_text(html_doc, encoding="utf-8")

    ok_n = sum(1 for c in checks if c["ok"])
    print(f"[OK] checks: {ok_n}/{len(checks)}")
    print(f"[OK] html: {out_html}")
    print(f"[OK] csv: {out_csv}")

    # Extra consistency numbers for terminal log
    print(
        "[SCOPE g102+g103] "
        f"best_n={scope_best.n}, pub_n={scope_pub.n}, cov={scope_coverage:.4f}, "
        f"best_mae={scope_best.mae:.4f}, pub_mae={scope_pub.mae:.4f}, "
        f"best_rmse={scope_best.rmse:.4f}, pub_rmse={scope_pub.rmse:.4f}, "
        f"best_mape={scope_best.mape:.4f}, pub_mape={scope_pub.mape:.4f}"
    )
    print(
        "[ALL g101+g102+g103] "
        f"best_n={all_best.n}, pub_n={all_pub.n}, cov={all_coverage:.4f}, "
        f"best_mae={all_best.mae:.4f}, pub_mae={all_pub.mae:.4f}, "
        f"best_rmse={all_best.rmse:.4f}, pub_rmse={all_pub.rmse:.4f}, "
        f"best_mape={all_best.mape:.4f}, pub_mape={all_pub.mape:.4f}"
    )


if __name__ == "__main__":
    main()
