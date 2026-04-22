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
        "--html-out",
        default="results/visualizations/fc_threshold_report.html",
        help="Output HTML report path.",
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


def _render_html(
    overall_rows: List[Dict[str, float]],
    matrix_wide: pd.DataFrame,
    thresholds: List[float],
    sample_stats_json: str,
    series_json: str,
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

<script>
const SAMPLE_STATS = {sample_stats_json};
const SERIES_DATA = {series_json};
const THRESHOLDS = {threshold_js};

function fmt(v, digits=3) {{
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "";
  return Number(v).toFixed(digits);
}}

function fmtPct(v) {{
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "";
  return (Number(v) * 100).toFixed(1) + "%";
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
    html_out = Path(args.html_out)

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

    for out in [matrix_csv_out, matrix_long_csv_out, per_file_hr_csv_out, html_out]:
        out.parent.mkdir(parents=True, exist_ok=True)

    matrix_wide.to_csv(matrix_csv_out, index=False)
    stats_long.to_csv(matrix_long_csv_out, index=False)
    stats_long.to_csv(per_file_hr_csv_out, index=False)

    sample_stats_json = _build_sample_stats_json(stats_long, thresholds)
    series_json = _build_series_json(df)
    html_text = _render_html(
        overall_rows=overall_rows,
        matrix_wide=matrix_wide,
        thresholds=thresholds,
        sample_stats_json=sample_stats_json,
        series_json=series_json,
    )
    html_out.write_text(html_text, encoding="utf-8")

    print(f"[OK] matrix wide: {matrix_csv_out}")
    print(f"[OK] matrix long: {matrix_long_csv_out}")
    print(f"[OK] per-file hr: {per_file_hr_csv_out}")
    print(f"[OK] html report: {html_out}")


if __name__ == "__main__":
    main()
