#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _to_float(v: object) -> Optional[float]:
    if v is None:
        return None
    txt = str(v).strip()
    if txt == "" or txt.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _metrics_from_rows(rows: List[Dict[str, str]]) -> Dict[str, float]:
    vals: List[Tuple[float, float]] = []
    for r in rows:
        ref = _to_float(r.get("ecg_hr"))
        est = _to_float(r.get("est_hr"))
        if ref is None or est is None:
            continue
        vals.append((ref, est))

    if not vals:
        return {
            "n": 0.0,
            "mae": math.nan,
            "rmse": math.nan,
            "corr": math.nan,
            "ecg_mean": math.nan,
            "est_mean": math.nan,
            "bias": math.nan,
        }

    ref = np.array([v[0] for v in vals], dtype=np.float64)
    est = np.array([v[1] for v in vals], dtype=np.float64)
    err = est - ref
    corr = float(np.corrcoef(ref, est)[0, 1]) if len(vals) >= 2 else math.nan
    return {
        "n": float(len(vals)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "corr": corr,
        "ecg_mean": float(np.mean(ref)),
        "est_mean": float(np.mean(est)),
        "bias": float(np.mean(err)),
    }


def _extract_overall(summary_rows: List[Dict[str, str]]) -> Dict[str, float]:
    candidates = [r for r in summary_rows if r.get("group") == "ALL" and r.get("stem") == "ALL"]
    target = None
    for split_name in ("ALL", "TRAIN_ALL", "TEST_ALL"):
        for r in candidates:
            if (r.get("split") or "").upper() == split_name:
                target = r
                break
        if target is not None:
            break
    if target is None and candidates:
        target = candidates[0]
    if target is None:
        return {"n": 0.0, "mae": math.nan, "rmse": math.nan, "corr": math.nan}
    return {
        "n": float(_to_float(target.get("hr_n")) or 0.0),
        "mae": float(_to_float(target.get("hr_mae")) or math.nan),
        "rmse": float(_to_float(target.get("hr_rmse")) or math.nan),
        "corr": float(_to_float(target.get("hr_corr")) or math.nan),
    }


def _sample_rows_only(summary_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [r for r in summary_rows if r.get("group") != "ALL" and r.get("stem") != "ALL"]


def _fmt(v: Optional[float], nd: int = 3) -> str:
    if v is None or not math.isfinite(v):
        return "NaN"
    return f"{v:.{nd}f}"


def _safe_ratio(a: float, b: float) -> float:
    if b <= 0:
        return math.nan
    return a / b


def _parse_token_set(text: str) -> set[str]:
    tokens = [x.strip() for x in (text or "").split(",")]
    return {x for x in tokens if x}


def _condition_sort_key(v: str) -> Tuple[int, str]:
    s = (v or "").strip().upper()
    if s.startswith("C"):
        num = s[1:]
        if num.isdigit():
            return int(num), s
    return 10**9, s


def build_payload(
    best_summary: List[Dict[str, str]],
    pub_summary: List[Dict[str, str]],
    best_detail: List[Dict[str, str]],
    pub_detail: List[Dict[str, str]],
    manifest_rows: List[Dict[str, str]],
    include_devices: set[str],
    exclude_devices: set[str],
) -> Dict[str, object]:
    best_samples = _sample_rows_only(best_summary)
    pub_samples = _sample_rows_only(pub_summary)

    manifest_by_key = {
        f"{r.get('group','')}/{r.get('stem','')}": r
        for r in manifest_rows
    }

    best_by_key_all = {f"{r.get('group','')}/{r.get('stem','')}": r for r in best_samples}
    pub_by_key_all = {f"{r.get('group','')}/{r.get('stem','')}": r for r in pub_samples}
    all_keys_raw = sorted(set(best_by_key_all.keys()) | set(pub_by_key_all.keys()))

    def _device_ok(key: str) -> bool:
        dev = (manifest_by_key.get(key, {}) or {}).get("device", "")
        if include_devices and dev not in include_devices:
            return False
        if exclude_devices and dev in exclude_devices:
            return False
        return True

    all_keys = [k for k in all_keys_raw if _device_ok(k)]
    key_set = set(all_keys)

    best_by_key = {k: v for k, v in best_by_key_all.items() if k in key_set}
    pub_by_key = {k: v for k, v in pub_by_key_all.items() if k in key_set}

    best_detail = [r for r in best_detail if f"{r.get('group','')}/{r.get('stem','')}" in key_set]
    pub_detail = [r for r in pub_detail if f"{r.get('group','')}/{r.get('stem','')}" in key_set]
    best_overall = _metrics_from_rows(best_detail)
    pub_overall = _metrics_from_rows(pub_detail)

    best_detail_by_key: Dict[str, List[Dict[str, str]]] = {}
    for r in best_detail:
        k = f"{r.get('group','')}/{r.get('stem','')}"
        best_detail_by_key.setdefault(k, []).append(r)

    pub_detail_by_key: Dict[str, List[Dict[str, str]]] = {}
    for r in pub_detail:
        k = f"{r.get('group','')}/{r.get('stem','')}"
        pub_detail_by_key.setdefault(k, []).append(r)

    sample_table: List[Dict[str, object]] = []
    for k in all_keys:
        b = best_by_key.get(k, {})
        p = pub_by_key.get(k, {})
        m = manifest_by_key.get(k, {})

        best_n = float(_to_float(b.get("hr_n")) or 0.0)
        pub_n = float(_to_float(p.get("hr_n")) or 0.0)
        cov = _safe_ratio(pub_n, best_n)

        best_hr = _metrics_from_rows(best_detail_by_key.get(k, []))
        pub_hr = _metrics_from_rows(pub_detail_by_key.get(k, []))

        sample_table.append(
            {
                "key": k,
                "group": k.split("/", 1)[0] if "/" in k else "",
                "stem": k.split("/", 1)[1] if "/" in k else k,
                "device": m.get("device", ""),
                "condition": m.get("condition", ""),
                "best_n": best_n,
                "best_mae": _to_float(b.get("hr_mae")),
                "best_rmse": _to_float(b.get("hr_rmse")),
                "best_corr": _to_float(b.get("hr_corr")),
                "pub_n": pub_n,
                "pub_mae": _to_float(p.get("hr_mae")),
                "pub_rmse": _to_float(p.get("hr_rmse")),
                "pub_corr": _to_float(p.get("hr_corr")),
                "coverage": cov,
                "ecg_mean": best_hr["ecg_mean"] if math.isfinite(best_hr["ecg_mean"]) else pub_hr["ecg_mean"],
                "best_mean": best_hr["est_mean"],
                "pub_mean": pub_hr["est_mean"],
                "best_bias": best_hr["bias"],
                "pub_bias": pub_hr["bias"],
            }
        )

    group_rows: List[Dict[str, object]] = []
    groups = sorted({(r.get("group") or "") for r in sample_table if (r.get("group") or "") != ""})
    for g in groups:
        b_rows = [r for r in best_detail if r.get("group") == g]
        p_rows = [r for r in pub_detail if r.get("group") == g]
        b = _metrics_from_rows(b_rows)
        p = _metrics_from_rows(p_rows)
        group_rows.append(
            {
                "group": g,
                "device": next((r.get("device", "") for r in sample_table if r.get("group") == g), ""),
                "best_n": b["n"],
                "best_mae": b["mae"],
                "best_rmse": b["rmse"],
                "best_corr": b["corr"],
                "pub_n": p["n"],
                "pub_mae": p["mae"],
                "pub_rmse": p["rmse"],
                "pub_corr": p["corr"],
                "coverage": _safe_ratio(p["n"], b["n"]),
            }
        )

    # Per-second series payload for C1~C6 comparison plots.
    best_rows_by_key: Dict[str, List[Dict[str, str]]] = {}
    for r in best_detail:
        k = f"{r.get('group','')}/{r.get('stem','')}"
        best_rows_by_key.setdefault(k, []).append(r)
    pub_rows_by_key: Dict[str, List[Dict[str, str]]] = {}
    for r in pub_detail:
        k = f"{r.get('group','')}/{r.get('stem','')}"
        pub_rows_by_key.setdefault(k, []).append(r)

    series_rows: List[Dict[str, object]] = []
    condition_set: set[str] = set()
    for k in all_keys:
        m = manifest_by_key.get(k, {})
        cond = str(m.get("condition", "") or "")
        dev = str(m.get("device", "") or "")
        grp = k.split("/", 1)[0] if "/" in k else ""
        stem = k.split("/", 1)[1] if "/" in k else k
        if cond:
            condition_set.add(cond)

        sec_map: Dict[int, Dict[str, object]] = {}
        for r in best_rows_by_key.get(k, []):
            sec_v = _to_float(r.get("sec"))
            if sec_v is None:
                continue
            sec = int(sec_v)
            rec = sec_map.setdefault(sec, {"sec": sec, "ecg": None, "best": None, "published": None})
            ecg = _to_float(r.get("ecg_hr"))
            if rec["ecg"] is None and ecg is not None:
                rec["ecg"] = ecg
            rec["best"] = _to_float(r.get("est_hr"))
        for r in pub_rows_by_key.get(k, []):
            sec_v = _to_float(r.get("sec"))
            if sec_v is None:
                continue
            sec = int(sec_v)
            rec = sec_map.setdefault(sec, {"sec": sec, "ecg": None, "best": None, "published": None})
            ecg = _to_float(r.get("ecg_hr"))
            if rec["ecg"] is None and ecg is not None:
                rec["ecg"] = ecg
            rec["published"] = _to_float(r.get("est_hr"))

        points = [sec_map[s] for s in sorted(sec_map.keys())]
        if not points:
            continue
        series_rows.append(
            {
                "key": k,
                "group": grp,
                "stem": stem,
                "device": dev,
                "condition": cond,
                "points": points,
            }
        )

    condition_order = sorted(condition_set, key=_condition_sort_key)

    payload = {
        "overall": {
            "best": best_overall,
            "published": pub_overall,
            "coverage": _safe_ratio(float(pub_overall["n"]), float(best_overall["n"])),
        },
        "groups": group_rows,
        "samples": sample_table,
        "series": series_rows,
        "meta": {
            "n_samples": len(sample_table),
            "source": "Data2_as_Data1",
            "include_devices": sorted(include_devices),
            "exclude_devices": sorted(exclude_devices),
            "conditions": condition_order,
        },
    }
    return payload


def render_html(payload: Dict[str, object], title: str) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --bg0: #f4f7f9;
      --bg1: #ffffff;
      --ink: #102023;
      --muted: #5b6d71;
      --line: #d6e0e3;
      --primary: #0b7a75;
      --accent: #e58a00;
      --good: #1f9d55;
      --warn: #b7791f;
      --bad: #c53030;
      --shadow: 0 8px 24px rgba(16, 32, 35, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "PingFang SC", "Hiragino Sans", "Noto Sans CJK JP", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 8% -10%, #d7ecea 0%, transparent 35%),
        radial-gradient(circle at 92% -12%, #ffe3b8 0%, transparent 35%),
        var(--bg0);
      min-height: 100vh;
    }}
    .layout {{
      max-width: 1540px;
      margin: 20px auto 40px;
      padding: 0 14px;
      display: grid;
      grid-template-columns: 240px 1fr;
      gap: 16px;
      align-items: start;
    }}
    .sidebar {{
      position: sticky;
      top: 14px;
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: var(--shadow);
      padding: 10px;
      max-height: calc(100vh - 28px);
      overflow: auto;
    }}
    .sidebar h3 {{
      margin: 4px 6px 10px;
      font-size: 13px;
      color: #244b4f;
      letter-spacing: .04em;
      text-transform: uppercase;
    }}
    .side-link {{
      display: block;
      text-decoration: none;
      color: #244347;
      border: 1px solid transparent;
      border-radius: 9px;
      padding: 8px 10px;
      font-size: 13px;
      margin-bottom: 6px;
      background: #f7fbfb;
    }}
    .side-link:hover {{
      border-color: #bfe3e2;
      background: #ecf7f7;
      color: #0d4b53;
    }}
    .main {{
      min-width: 0;
    }}
    .head {{
      background: linear-gradient(135deg, #0d4b53, #0b7a75 62%, #1f9d55);
      color: #f5fffe;
      border-radius: 18px;
      padding: 20px 22px;
      box-shadow: var(--shadow);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .head h1 {{
      margin: 0;
      font-size: clamp(20px, 3vw, 32px);
      line-height: 1.2;
      letter-spacing: .02em;
    }}
    .head .sub {{
      margin-top: 6px;
      color: #d7fffa;
      font-size: 14px;
    }}
    .lang {{
      display: flex;
      gap: 8px;
      align-items: center;
      background: rgba(255,255,255,.12);
      border: 1px solid rgba(255,255,255,.26);
      border-radius: 12px;
      padding: 8px 10px;
    }}
    .lang select {{
      border: none;
      border-radius: 8px;
      padding: 6px 8px;
      color: #0d4b53;
      background: #f8fffe;
      font-weight: 600;
    }}
    .grid {{
      margin-top: 16px;
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}
    .card {{
      background: var(--bg1);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: var(--shadow);
      padding: 14px 14px 12px;
    }}
    .card .label {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .08em;
    }}
    .card .value {{
      margin-top: 6px;
      font-size: 26px;
      font-weight: 700;
      line-height: 1.1;
    }}
    .card .note {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
    }}
    .panel {{
      margin-top: 16px;
      background: var(--bg1);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: var(--shadow);
      padding: 14px;
    }}
    .panel h2 {{
      margin: 0 0 10px;
      font-size: 20px;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      min-width: 980px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child,
    th:nth-child(2), td:nth-child(2),
    th:nth-child(3), td:nth-child(3) {{
      text-align: left;
    }}
    thead th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f3fbfb;
      color: #14363a;
      font-weight: 700;
    }}
    tr:hover td {{ background: #f8fdfd; }}
    .muted {{ color: var(--muted); }}
    .good {{ color: var(--good); font-weight: 700; }}
    .warn {{ color: var(--warn); font-weight: 700; }}
    .bad {{ color: var(--bad); font-weight: 700; }}
    .toolbar {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin: 0 0 10px;
    }}
    .toolbar input {{
      border: 1px solid var(--line);
      border-radius: 9px;
      padding: 7px 10px;
      min-width: 260px;
      font-size: 13px;
    }}
    .chip {{
      display: inline-block;
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 3px 8px;
      font-size: 11px;
      color: #35595c;
      background: #eef8f8;
    }}
    .foot {{
      margin-top: 12px;
      color: var(--muted);
      font-size: 12px;
    }}
    .viz-grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(440px, 1fr));
    }}
    .viz-card {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px;
      background: #fbfefe;
    }}
    .viz-title {{
      margin: 0 0 6px;
      color: #234b50;
      font-size: 14px;
      font-weight: 700;
    }}
    .viz-controls {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 8px;
      align-items: center;
    }}
    .viz-controls label {{
      font-size: 12px;
      color: var(--muted);
    }}
    .viz-controls select,
    .viz-controls input[type="range"] {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 5px 8px;
      background: #fff;
      color: #12393d;
      font-size: 12px;
    }}
    .viz-controls .tiny {{
      font-size: 12px;
      color: #12393d;
      min-width: 42px;
      text-align: right;
      font-weight: 700;
    }}
    svg.plot {{
      width: 100%;
      height: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
    }}
    .axis-label {{
      font-size: 11px;
      fill: #5b6d71;
    }}
    .bar-label {{
      font-size: 11px;
      fill: #183e42;
    }}
    .bar-value {{
      font-size: 11px;
      fill: #0d4b53;
      font-weight: 700;
    }}
    @media (max-width: 1100px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        position: static;
        max-height: none;
      }}
      .side-nav-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 6px;
      }}
      .side-link {{
        margin-bottom: 0;
      }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h3 id="sideTitle"></h3>
      <div class="side-nav-grid">
        <a class="side-link" id="lnkOverview" href="#sec-overview"></a>
        <a class="side-link" id="lnkKpi" href="#sec-kpi"></a>
        <a class="side-link" id="lnkGroup" href="#sec-group"></a>
        <a class="side-link" id="lnkSample" href="#sec-sample"></a>
        <a class="side-link" id="lnkViz" href="#sec-viz"></a>
        <a class="side-link" id="lnkTimeseries" href="#sec-timeseries"></a>
        <a class="side-link" id="lnkParam" href="#sec-param"></a>
      </div>
    </aside>
    <main class="main">
    <header class="head" id="sec-overview">
      <div>
        <h1 id="title"></h1>
        <div class="sub" id="subtitle"></div>
      </div>
      <div class="lang">
        <label id="langLabel" for="langSel"></label>
        <select id="langSel">
          <option value="zh">中文</option>
          <option value="en">English</option>
          <option value="ja">日本語</option>
        </select>
      </div>
    </header>

    <section class="panel" id="sec-kpi">
      <h2 id="kpiTitle"></h2>
      <section class="grid" id="kpiGrid"></section>
      <div class="foot" id="kpiFoot"></div>
    </section>

    <section class="panel" id="sec-group">
      <h2 id="groupTitle"></h2>
      <div class="table-wrap">
        <table id="groupTable">
          <thead></thead>
          <tbody></tbody>
        </table>
      </div>
    </section>

    <section class="panel" id="sec-sample">
      <h2 id="sampleTitle"></h2>
      <div class="toolbar">
        <input id="sampleFilter" />
      </div>
      <div class="table-wrap">
        <table id="sampleTable">
          <thead></thead>
          <tbody></tbody>
        </table>
      </div>
      <div class="foot" id="footNote"></div>
    </section>

    <section class="panel" id="sec-viz">
      <h2 id="vizTitle"></h2>
      <div class="viz-controls">
        <label id="vizMetricLabel" for="vizMetric"></label>
        <select id="vizMetric"></select>
        <label id="vizGroupLabel" for="vizGroup"></label>
        <select id="vizGroup"></select>
        <label id="vizTopNLabel" for="vizTopN"></label>
        <input id="vizTopN" type="range" min="5" max="30" step="1" value="12" />
        <span id="vizTopNVal" class="tiny">12</span>
      </div>
      <div class="viz-grid">
        <div class="viz-card">
          <h3 class="viz-title" id="vizDeviceTitle"></h3>
          <svg id="devicePlot" class="plot" viewBox="0 0 900 340" role="img" aria-label="device plot"></svg>
        </div>
        <div class="viz-card">
          <h3 class="viz-title" id="vizSampleTitle"></h3>
          <svg id="samplePlot" class="plot" viewBox="0 0 900 520" role="img" aria-label="sample plot"></svg>
        </div>
      </div>
    </section>

    <section class="panel" id="sec-timeseries">
      <h2 id="tsTitle"></h2>
      <div class="viz-controls">
        <label id="tsCondLabel" for="tsCond"></label>
        <select id="tsCond"></select>
        <label id="tsMetricLabel" for="tsMetric"></label>
        <select id="tsMetric"></select>
        <label id="tsChannelLabel" for="tsChannel"></label>
        <select id="tsChannel"></select>
      </div>
      <div class="viz-card">
        <h3 class="viz-title" id="tsPlotTitle"></h3>
        <svg id="tsPlot" class="plot" viewBox="0 0 1100 480" role="img" aria-label="timeseries plot"></svg>
      </div>
      <div class="foot" id="tsFoot"></div>
    </section>

    <section class="panel" id="sec-param">
      <h2 id="paramTitle"></h2>
      <div class="table-wrap">
        <table id="paramTable">
          <thead></thead>
          <tbody></tbody>
        </table>
      </div>
    </section>
    </main>
  </div>

  <script>
    const DATA = {data_json};
    const I18N = {{
      zh: {{
        title: "Data2 RPPG 评估报告（Data1 格式）",
        subtitle: "对比通道：best 与 published；展示整体、分设备、分文件指标",
        lang: "语言",
        sideTitle: "导航",
        sideOverview: "总览",
        sideKpi: "KPI 指标",
        sideGroup: "分设备",
        sideSample: "分文件",
        sideViz: "交互图表",
        sideTimeseries: "C1-C6 逐秒变化",
        sideParam: "参数解释",
        kpiTitle: "核心指标概览",
        kpiFoot: "KPI 用于第一眼判断：精度、覆盖率、样本可用性。",
        kBestMae: "整体 MAE (best)",
        kPubMae: "整体 MAE (published)",
        kCoverage: "覆盖率 (published/best)",
        kSamples: "样本数",
        groupTitle: "分设备汇总",
        sampleTitle: "分文件对比",
        search: "筛选：group/stem/device/condition",
        foot: "注：corr 在短窗口和低波动 HR 条件下不稳定，MAE/RMSE 更具可比性。",
        gCols: ["Group","Device","best n","best MAE","best RMSE","best corr","pub n","pub MAE","pub RMSE","pub corr","Coverage"],
        sCols: ["Group","Stem","Device","Cond","best n","best MAE","best RMSE","pub n","pub MAE","pub RMSE","Coverage","ECG mean","best mean","pub mean","best bias","pub bias"],
        vizTitle: "交互可视化",
        vizMetric: "指标",
        vizGroup: "组别",
        vizTopN: "Top N 样本",
        vizDeviceTitle: "设备级指标对比",
        vizSampleTitle: "样本级指标条形图",
        tsTitle: "C1-C6 每秒数据变化对比",
        tsCond: "条件",
        tsMetric: "曲线类型",
        tsChannel: "估计通道",
        tsPlotTitle: "逐秒趋势图",
        tsFoot: "黑色虚线为 ECG；彩线为各设备 RPPG。可按条件 C1~C6 对比。",
        tsAllConds: "全部条件",
        tsMetricOpts: {{
          hr: "HR 曲线 (ECG vs RPPG)",
          abs_err: "绝对误差曲线 |RPPG-ECG|"
        }},
        tsChannelOpts: {{
          best: "best",
          published: "published"
        }},
        paramTitle: "参数解释（定义 / 单位 / 解读）",
        pCols: ["参数", "定义", "单位/公式", "如何解读"],
        metricOpts: {{
          best_mae: "best MAE",
          pub_mae: "published MAE",
          coverage: "Coverage",
          best_rmse: "best RMSE",
          pub_rmse: "published RMSE",
          best_corr: "best corr",
          pub_corr: "published corr"
        }},
        allGroups: "全部",
        noData: "当前筛选无数据",
        params: [
          ["group", "数据分组ID（映射到设备）", "字符串", "用于区分不同设备或采集组。"],
          ["stem", "文件样本名（视频/ECG 对）", "字符串", "唯一定位单个测试文件。"],
          ["device", "采集设备", "字符串", "例如 iphone16e / lenovo。"],
          ["condition", "实验条件编号", "C1~C6", "对应同一批次同步采集条件。"],
          ["best n", "best 通道有效对齐点数", "计数", "越高表示可用估计越多。"],
          ["pub n", "published 通道有效对齐点数", "计数", "经过发布门控后剩余点数。"],
          ["best MAE", "best 通道平均绝对误差", "BPM", "越低越好；<=3 通常较优。"],
          ["published MAE", "published 通道平均绝对误差", "BPM", "越低越好；通常比 best 更稳。"],
          ["best RMSE", "best 通道均方根误差", "BPM", "对大误差更敏感，越低越好。"],
          ["published RMSE", "published 通道均方根误差", "BPM", "越低越好。"],
          ["best corr", "best 通道相关系数", "[-1,1]", "越接近 1 越一致；短窗口可能不稳定。"],
          ["published corr", "published 通道相关系数", "[-1,1]", "同上。"],
          ["Coverage", "发布覆盖率", "pub n / best n", "越高表示门控后保留数据越多。"],
          ["ECG mean", "参考 ECG 平均 HR", "BPM", "该样本 ECG 基线。"],
          ["best mean", "best 平均估计 HR", "BPM", "与 ECG mean 的差可观察系统偏差。"],
          ["pub mean", "published 平均估计 HR", "BPM", "发布结果的平均值。"],
          ["best bias", "best 平均偏差", "mean(est-ecg)", "正值=估计偏高，负值=估计偏低。"],
          ["pub bias", "published 平均偏差", "mean(est-ecg)", "同上。"]
        ],
      }},
      en: {{
        title: "Data2 RPPG Evaluation Report (Data1-style)",
        subtitle: "Channels compared: best vs published; includes overall, per-device, and per-file metrics.",
        lang: "Language",
        sideTitle: "Navigation",
        sideOverview: "Overview",
        sideKpi: "KPI",
        sideGroup: "By Device",
        sideSample: "By File",
        sideViz: "Interactive Charts",
        sideTimeseries: "C1-C6 Per-Second",
        sideParam: "Glossary",
        kpiTitle: "KPI Overview",
        kpiFoot: "A quick read on accuracy, coverage, and usable sample quality.",
        kBestMae: "Overall MAE (best)",
        kPubMae: "Overall MAE (published)",
        kCoverage: "Coverage (published/best)",
        kSamples: "Samples",
        groupTitle: "Per-Device Summary",
        sampleTitle: "Per-File Comparison",
        search: "Filter: group/stem/device/condition",
        foot: "Note: corr can be unstable in short windows or low-HR-variance segments; MAE/RMSE are more comparable.",
        gCols: ["Group","Device","best n","best MAE","best RMSE","best corr","pub n","pub MAE","pub RMSE","pub corr","Coverage"],
        sCols: ["Group","Stem","Device","Cond","best n","best MAE","best RMSE","pub n","pub MAE","pub RMSE","Coverage","ECG mean","best mean","pub mean","best bias","pub bias"],
        vizTitle: "Interactive Visualization",
        vizMetric: "Metric",
        vizGroup: "Group",
        vizTopN: "Top N",
        vizDeviceTitle: "Device-level Metric Comparison",
        vizSampleTitle: "Sample-level Bar Chart",
        tsTitle: "C1-C6 Per-Second Trend Comparison",
        tsCond: "Condition",
        tsMetric: "Curve Type",
        tsChannel: "Estimate Channel",
        tsPlotTitle: "Per-Second Trend Plot",
        tsFoot: "Black dashed line: ECG; colored lines: device RPPG. Compare by condition C1~C6.",
        tsAllConds: "ALL Conditions",
        tsMetricOpts: {{
          hr: "HR Curves (ECG vs RPPG)",
          abs_err: "Absolute Error |RPPG-ECG|"
        }},
        tsChannelOpts: {{
          best: "best",
          published: "published"
        }},
        paramTitle: "Parameter Glossary (Definition / Unit / Interpretation)",
        pCols: ["Parameter", "Definition", "Unit/Formula", "Interpretation"],
        metricOpts: {{
          best_mae: "best MAE",
          pub_mae: "published MAE",
          coverage: "Coverage",
          best_rmse: "best RMSE",
          pub_rmse: "published RMSE",
          best_corr: "best corr",
          pub_corr: "published corr"
        }},
        allGroups: "ALL",
        noData: "No data for current filter",
        params: [
          ["group", "Dataset group ID (mapped to device)", "string", "Used to separate devices or acquisition groups."],
          ["stem", "Sample file key (video/ECG pair)", "string", "Uniquely identifies a test sample."],
          ["device", "Capture device", "string", "For example iphone16e / lenovo."],
          ["condition", "Experiment condition ID", "C1~C6", "Synchronized condition index across devices."],
          ["best n", "Valid aligned points in best channel", "count", "Higher means more usable estimates."],
          ["pub n", "Valid aligned points in published channel", "count", "Points left after publish gating."],
          ["best MAE", "Mean Absolute Error of best channel", "BPM", "Lower is better; <=3 is usually strong."],
          ["published MAE", "Mean Absolute Error of published channel", "BPM", "Lower is better; often more stable."],
          ["best RMSE", "Root Mean Squared Error of best", "BPM", "More sensitive to large errors."],
          ["published RMSE", "Root Mean Squared Error of published", "BPM", "Lower is better."],
          ["best corr", "Correlation of best channel", "[-1,1]", "Closer to 1 means stronger agreement."],
          ["published corr", "Correlation of published channel", "[-1,1]", "Same as above."],
          ["Coverage", "Published coverage", "pub n / best n", "Higher means more points survive gating."],
          ["ECG mean", "Mean ECG HR", "BPM", "ECG baseline for that sample."],
          ["best mean", "Mean estimated HR in best", "BPM", "Compare with ECG mean for systematic drift."],
          ["pub mean", "Mean estimated HR in published", "BPM", "Published-channel average."],
          ["best bias", "Average bias in best", "mean(est-ecg)", "Positive=over-estimation, negative=under."],
          ["pub bias", "Average bias in published", "mean(est-ecg)", "Same as above."]
        ],
      }},
      ja: {{
        title: "Data2 RPPG 評価レポート（Data1互換）",
        subtitle: "best と published を比較し、全体・デバイス別・ファイル別の指標を表示します。",
        lang: "言語",
        sideTitle: "ナビゲーション",
        sideOverview: "概要",
        sideKpi: "KPI",
        sideGroup: "デバイス別",
        sideSample: "ファイル別",
        sideViz: "インタラクティブ図表",
        sideTimeseries: "C1-C6 秒次推移",
        sideParam: "用語集",
        kpiTitle: "KPI 概要",
        kpiFoot: "精度・カバレッジ・有効サンプル性を素早く確認します。",
        kBestMae: "全体 MAE (best)",
        kPubMae: "全体 MAE (published)",
        kCoverage: "カバレッジ (published/best)",
        kSamples: "サンプル数",
        groupTitle: "デバイス別集計",
        sampleTitle: "ファイル別比較",
        search: "絞り込み: group/stem/device/condition",
        foot: "注: corr は短時間・低変動HRでは不安定になりやすく、MAE/RMSE の方が比較しやすいです。",
        gCols: ["Group","Device","best n","best MAE","best RMSE","best corr","pub n","pub MAE","pub RMSE","pub corr","Coverage"],
        sCols: ["Group","Stem","Device","Cond","best n","best MAE","best RMSE","pub n","pub MAE","pub RMSE","Coverage","ECG mean","best mean","pub mean","best bias","pub bias"],
        vizTitle: "インタラクティブ可視化",
        vizMetric: "指標",
        vizGroup: "グループ",
        vizTopN: "上位 N サンプル",
        vizDeviceTitle: "デバイス別指標比較",
        vizSampleTitle: "サンプル別バー表示",
        tsTitle: "C1-C6 秒ごとのデータ推移比較",
        tsCond: "条件",
        tsMetric: "曲線タイプ",
        tsChannel: "推定チャネル",
        tsPlotTitle: "秒次トレンド図",
        tsFoot: "黒破線は ECG、色線は各デバイスの RPPG。C1~C6 条件で比較できます。",
        tsAllConds: "全条件",
        tsMetricOpts: {{
          hr: "HR 曲線 (ECG vs RPPG)",
          abs_err: "絶対誤差 |RPPG-ECG|"
        }},
        tsChannelOpts: {{
          best: "best",
          published: "published"
        }},
        paramTitle: "パラメータ説明（定義 / 単位 / 解釈）",
        pCols: ["パラメータ", "定義", "単位/式", "解釈"],
        metricOpts: {{
          best_mae: "best MAE",
          pub_mae: "published MAE",
          coverage: "Coverage",
          best_rmse: "best RMSE",
          pub_rmse: "published RMSE",
          best_corr: "best corr",
          pub_corr: "published corr"
        }},
        allGroups: "全体",
        noData: "現在の条件にデータがありません",
        params: [
          ["group", "データグループID（デバイス対応）", "文字列", "デバイス/収録群の区別に使用。"],
          ["stem", "サンプル識別子（動画/ECG 対）", "文字列", "単一サンプルを一意に特定。"],
          ["device", "収録デバイス", "文字列", "例: iphone16e / lenovo。"],
          ["condition", "実験条件ID", "C1~C6", "デバイス間で同期した条件番号。"],
          ["best n", "best チャンネル有効点数", "件数", "大きいほど利用可能推定点が多い。"],
          ["pub n", "published チャンネル有効点数", "件数", "公開ゲート通過後の点数。"],
          ["best MAE", "best の平均絶対誤差", "BPM", "小さいほど良い（目安 <=3）。"],
          ["published MAE", "published の平均絶対誤差", "BPM", "小さいほど良い。"],
          ["best RMSE", "best の二乗平均平方根誤差", "BPM", "大外れ誤差に敏感。"],
          ["published RMSE", "published の二乗平均平方根誤差", "BPM", "小さいほど良い。"],
          ["best corr", "best の相関係数", "[-1,1]", "1 に近いほど一致。"],
          ["published corr", "published の相関係数", "[-1,1]", "同上。"],
          ["Coverage", "公開カバレッジ", "pub n / best n", "高いほどゲート後の残存率が高い。"],
          ["ECG mean", "ECG 平均HR", "BPM", "そのサンプルの ECG 基準値。"],
          ["best mean", "best 平均推定HR", "BPM", "ECG平均との差で系統偏差を確認。"],
          ["pub mean", "published 平均推定HR", "BPM", "公開チャンネル平均。"],
          ["best bias", "best 平均バイアス", "mean(est-ecg)", "正=高め推定、負=低め推定。"],
          ["pub bias", "published 平均バイアス", "mean(est-ecg)", "同上。"]
        ],
      }}
    }};

    function fmt(v, nd = 3) {{
      if (v === null || v === undefined || Number.isNaN(Number(v))) return "NaN";
      return Number(v).toFixed(nd);
    }}
    function pct(v, nd = 1) {{
      if (v === null || v === undefined || Number.isNaN(Number(v))) return "NaN";
      return (Number(v) * 100).toFixed(nd) + "%";
    }}
    function clsMae(v) {{
      if (v === null || v === undefined || Number.isNaN(Number(v))) return "muted";
      const x = Number(v);
      if (x <= 3) return "good";
      if (x <= 6) return "warn";
      return "bad";
    }}
    function clsCov(v) {{
      if (v === null || v === undefined || Number.isNaN(Number(v))) return "muted";
      const x = Number(v);
      if (x >= 0.75) return "good";
      if (x >= 0.45) return "warn";
      return "bad";
    }}

    function setTableHead(tableId, cols) {{
      const tr = "<tr>" + cols.map(c => `<th>${{c}}</th>`).join("") + "</tr>";
      document.querySelector(`#${{tableId}} thead`).innerHTML = tr;
    }}

    function metricFormat(metric, value) {{
      if (metric === "coverage") return pct(value);
      return fmt(value);
    }}

    function barColor(metric) {{
      if (metric === "coverage") return "#0b7a75";
      if (metric.includes("corr")) return "#2b6cb0";
      if (metric.includes("rmse")) return "#d97706";
      return "#0d9488";
    }}

    function drawBarPlot(svgId, rows, metricKey, labelFn, noDataText) {{
      const svg = document.getElementById(svgId);
      const w = 900, h = svgId === "samplePlot" ? 520 : 340;
      const m = {{ left: 180, right: 46, top: 32, bottom: 30 }};
      const vals = rows
        .map(r => Number(r[metricKey]))
        .filter(v => Number.isFinite(v));

      if (!rows.length || !vals.length) {{
        svg.innerHTML = `<text x="${{w/2}}" y="${{h/2}}" text-anchor="middle" class="axis-label">${{noDataText}}</text>`;
        return;
      }}
      const maxV = Math.max(...vals, metricKey === "coverage" ? 1 : 0.01);
      const plotW = w - m.left - m.right;
      const plotH = h - m.top - m.bottom;
      const rowH = plotH / rows.length;
      const color = barColor(metricKey);

      const bars = rows.map((r, i) => {{
        const v = Number(r[metricKey]);
        const vv = Number.isFinite(v) ? Math.max(v, 0) : 0;
        const y = m.top + i * rowH + rowH * 0.14;
        const bh = Math.max(12, rowH * 0.72);
        const bw = (vv / maxV) * plotW;
        const label = labelFn(r);
        return `
          <text x="8" y="${{(y + bh * 0.74).toFixed(2)}}" class="bar-label">${{label}}</text>
          <rect x="${{m.left}}" y="${{y.toFixed(2)}}" width="${{bw.toFixed(2)}}" height="${{bh.toFixed(2)}}" fill="${{color}}" opacity="0.86" rx="4" />
          <text x="${{(m.left + bw + 8).toFixed(2)}}" y="${{(y + bh * 0.74).toFixed(2)}}" class="bar-value">${{metricFormat(metricKey, vv)}}</text>
        `;
      }}).join("");

      const ticks = [];
      for (let i = 0; i <= 4; i++) {{
        const v = (maxV * i) / 4;
        const x = m.left + (plotW * i) / 4;
        ticks.push(`<line x1="${{x.toFixed(2)}}" y1="${{m.top}}" x2="${{x.toFixed(2)}}" y2="${{(h - m.bottom)}}" stroke="#e3edef" />`);
        ticks.push(`<text x="${{x.toFixed(2)}}" y="${{h - 8}}" text-anchor="middle" class="axis-label">${{metricFormat(metricKey, v)}}</text>`);
      }}

      svg.innerHTML = `
        <rect x="0" y="0" width="${{w}}" height="${{h}}" fill="transparent" />
        <line x1="${{m.left}}" y1="${{m.top}}" x2="${{m.left}}" y2="${{h - m.bottom}}" stroke="#c8d8dc" />
        <line x1="${{m.left}}" y1="${{h - m.bottom}}" x2="${{w - m.right}}" y2="${{h - m.bottom}}" stroke="#c8d8dc" />
        ${{ticks.join("")}}
        ${{bars}}
      `;
    }}

    const DEVICE_PALETTE = ["#0f766e", "#2563eb", "#ea580c", "#7c3aed", "#059669", "#d946ef", "#0891b2", "#be123c"];
    function deviceColor(name) {{
      const s = String(name || "");
      let h = 0;
      for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
      return DEVICE_PALETTE[h % DEVICE_PALETTE.length];
    }}

    function linePath(points, xFn, yFn) {{
      const valid = points.filter(p => Number.isFinite(p.x) && Number.isFinite(p.y));
      if (!valid.length) return "";
      return valid.map((p, i) => `${{i === 0 ? "M" : "L"}}${{xFn(p.x).toFixed(2)}},${{yFn(p.y).toFixed(2)}}`).join(" ");
    }}

    function drawTimeSeriesPlot(seriesRows, cond, metric, channel, noDataText) {{
      const svg = document.getElementById("tsPlot");
      const w = 1100, h = 480;
      const m = {{ left: 58, right: 210, top: 28, bottom: 42 }};
      const plotW = w - m.left - m.right;
      const plotH = h - m.top - m.bottom;

      const picked = seriesRows.filter(r => cond === "ALL" ? true : r.condition === cond);
      if (!picked.length) {{
        svg.innerHTML = `<text x="${{w/2}}" y="${{h/2}}" text-anchor="middle" class="axis-label">${{noDataText}}</text>`;
        return;
      }}

      let xMin = Infinity, xMax = -Infinity;
      let yMin = Infinity, yMax = -Infinity;

      function pushY(v) {{
        if (Number.isFinite(v)) {{
          yMin = Math.min(yMin, v);
          yMax = Math.max(yMax, v);
        }}
      }}

      const deviceLines = picked.map(r => {{
        const pts = (r.points || []).map(p => {{
          const sec = Number(p.sec);
          const ecg = Number(p.ecg);
          const pred = Number(channel === "best" ? p.best : p.published);
          xMin = Math.min(xMin, sec);
          xMax = Math.max(xMax, sec);
          let y = NaN;
          if (metric === "abs_err") {{
            if (Number.isFinite(ecg) && Number.isFinite(pred)) y = Math.abs(pred - ecg);
          }} else {{
            if (Number.isFinite(pred)) y = pred;
          }}
          pushY(y);
          return {{ x: sec, y, ecg }};
        }});
        return {{
          key: r.key,
          group: r.group,
          device: r.device,
          condition: r.condition,
          color: deviceColor(r.device || r.group),
          points: pts,
        }};
      }});

      // ECG reference (mean by second across picked rows), only for HR curves.
      let ecgLine = [];
      if (metric === "hr") {{
        const bySec = new Map();
        for (const r of picked) {{
          for (const p of (r.points || [])) {{
            const sec = Number(p.sec);
            const ecg = Number(p.ecg);
            if (!Number.isFinite(sec) || !Number.isFinite(ecg)) continue;
            const arr = bySec.get(sec) || [];
            arr.push(ecg);
            bySec.set(sec, arr);
          }}
        }}
        ecgLine = Array.from(bySec.entries())
          .sort((a, b) => a[0] - b[0])
          .map(([sec, arr]) => {{
            const y = arr.reduce((s, x) => s + x, 0) / Math.max(arr.length, 1);
            pushY(y);
            return {{ x: sec, y }};
          }});
      }}

      if (!Number.isFinite(xMin) || !Number.isFinite(xMax) || !Number.isFinite(yMin) || !Number.isFinite(yMax)) {{
        svg.innerHTML = `<text x="${{w/2}}" y="${{h/2}}" text-anchor="middle" class="axis-label">${{noDataText}}</text>`;
        return;
      }}
      if (xMax <= xMin) xMax = xMin + 1;
      if (yMax <= yMin) {{
        const pad0 = Math.max(1, Math.abs(yMax) * 0.1);
        yMin -= pad0;
        yMax += pad0;
      }}
      const yPad = (yMax - yMin) * 0.08;
      yMin -= yPad;
      yMax += yPad;

      const xFn = x => m.left + (x - xMin) / (xMax - xMin) * plotW;
      const yFn = y => m.top + (yMax - y) / (yMax - yMin) * plotH;

      const grid = [];
      for (let i = 0; i <= 6; i++) {{
        const xv = xMin + (xMax - xMin) * i / 6;
        const xx = xFn(xv).toFixed(2);
        grid.push(`<line x1="${{xx}}" y1="${{m.top}}" x2="${{xx}}" y2="${{h - m.bottom}}" stroke="#edf4f5" />`);
        grid.push(`<text x="${{xx}}" y="${{h - 10}}" text-anchor="middle" class="axis-label">${{Math.round(xv)}}s</text>`);
      }}
      for (let i = 0; i <= 5; i++) {{
        const yv = yMin + (yMax - yMin) * i / 5;
        const yy = yFn(yv).toFixed(2);
        grid.push(`<line x1="${{m.left}}" y1="${{yy}}" x2="${{w - m.right}}" y2="${{yy}}" stroke="#edf4f5" />`);
        grid.push(`<text x="${{m.left - 8}}" y="${{(Number(yy) + 4).toFixed(2)}}" text-anchor="end" class="axis-label">${{fmt(yv,1)}}</text>`);
      }}

      const ecgPath = metric === "hr"
        ? `<path d="${{linePath(ecgLine, xFn, yFn)}}" fill="none" stroke="#111827" stroke-width="2.1" stroke-dasharray="6 5" opacity="0.9" />`
        : "";

      const lines = deviceLines.map(l => {{
        const d = linePath(l.points, xFn, yFn);
        if (!d) return "";
        const tail = [...l.points].reverse().find(p => Number.isFinite(p.y));
        const lx = tail ? xFn(tail.x) + 6 : (w - m.right - 4);
        const ly = tail ? yFn(tail.y) + 3 : (m.top + 12);
        return `
          <path d="${{d}}" fill="none" stroke="${{l.color}}" stroke-width="2" opacity="0.92" />
          <text x="${{lx.toFixed(2)}}" y="${{ly.toFixed(2)}}" class="axis-label" fill="${{l.color}}">${{l.group}}</text>
        `;
      }}).join("");

      const legend = [];
      let ly = m.top + 8;
      if (metric === "hr") {{
        legend.push(`<line x1="${{w - m.right + 10}}" y1="${{ly}}" x2="${{w - m.right + 40}}" y2="${{ly}}" stroke="#111827" stroke-width="2.1" stroke-dasharray="6 5" />`);
        legend.push(`<text x="${{w - m.right + 48}}" y="${{ly + 4}}" class="axis-label">ECG</text>`);
        ly += 18;
      }}
      for (const l of deviceLines) {{
        legend.push(`<line x1="${{w - m.right + 10}}" y1="${{ly}}" x2="${{w - m.right + 40}}" y2="${{ly}}" stroke="${{l.color}}" stroke-width="2.5" />`);
        legend.push(`<text x="${{w - m.right + 48}}" y="${{ly + 4}}" class="axis-label">${{l.group}} · ${{l.device || "-"}}</text>`);
        ly += 18;
      }}

      svg.innerHTML = `
        <rect x="0" y="0" width="${{w}}" height="${{h}}" fill="transparent" />
        <line x1="${{m.left}}" y1="${{m.top}}" x2="${{m.left}}" y2="${{h - m.bottom}}" stroke="#c8d8dc" />
        <line x1="${{m.left}}" y1="${{h - m.bottom}}" x2="${{w - m.right}}" y2="${{h - m.bottom}}" stroke="#c8d8dc" />
        ${{grid.join("")}}
        ${{ecgPath}}
        ${{lines}}
        ${{legend.join("")}}
        <text x="${{(m.left + plotW/2).toFixed(2)}}" y="${{h - 10}}" text-anchor="middle" class="axis-label">sec</text>
      `;
    }}

    function render(lang) {{
      const t = I18N[lang] || I18N.zh;
      document.documentElement.lang = lang === "ja" ? "ja" : (lang === "en" ? "en" : "zh");
      document.getElementById("title").textContent = t.title;
      document.getElementById("subtitle").textContent = t.subtitle;
      document.getElementById("langLabel").textContent = t.lang;
      document.getElementById("sideTitle").textContent = t.sideTitle;
      document.getElementById("lnkOverview").textContent = t.sideOverview;
      document.getElementById("lnkKpi").textContent = t.sideKpi;
      document.getElementById("lnkGroup").textContent = t.sideGroup;
      document.getElementById("lnkSample").textContent = t.sideSample;
      document.getElementById("lnkViz").textContent = t.sideViz;
      document.getElementById("lnkTimeseries").textContent = t.sideTimeseries;
      document.getElementById("lnkParam").textContent = t.sideParam;
      document.getElementById("kpiTitle").textContent = t.kpiTitle;
      document.getElementById("kpiFoot").textContent = t.kpiFoot;
      document.getElementById("groupTitle").textContent = t.groupTitle;
      document.getElementById("sampleTitle").textContent = t.sampleTitle;
      document.getElementById("sampleFilter").placeholder = t.search;
      document.getElementById("footNote").textContent = t.foot;
      document.getElementById("vizTitle").textContent = t.vizTitle;
      document.getElementById("vizMetricLabel").textContent = t.vizMetric;
      document.getElementById("vizGroupLabel").textContent = t.vizGroup;
      document.getElementById("vizTopNLabel").textContent = t.vizTopN;
      document.getElementById("vizDeviceTitle").textContent = t.vizDeviceTitle;
      document.getElementById("vizSampleTitle").textContent = t.vizSampleTitle;
      document.getElementById("tsTitle").textContent = t.tsTitle;
      document.getElementById("tsCondLabel").textContent = t.tsCond;
      document.getElementById("tsMetricLabel").textContent = t.tsMetric;
      document.getElementById("tsChannelLabel").textContent = t.tsChannel;
      document.getElementById("tsPlotTitle").textContent = t.tsPlotTitle;
      document.getElementById("tsFoot").textContent = t.tsFoot;
      document.getElementById("paramTitle").textContent = t.paramTitle;

      const k = DATA.overall;
      const cards = [
        {{ label: t.kBestMae, value: fmt(k.best.mae), note: `n=${{Math.round(k.best.n)}} | RMSE=${{fmt(k.best.rmse)}} | corr=${{fmt(k.best.corr)}}`, cls: clsMae(k.best.mae) }},
        {{ label: t.kPubMae, value: fmt(k.published.mae), note: `n=${{Math.round(k.published.n)}} | RMSE=${{fmt(k.published.rmse)}} | corr=${{fmt(k.published.corr)}}`, cls: clsMae(k.published.mae) }},
        {{ label: t.kCoverage, value: pct(k.coverage), note: `${{Math.round(k.published.n)}} / ${{Math.round(k.best.n)}}`, cls: clsCov(k.coverage) }},
        {{ label: t.kSamples, value: String(DATA.meta.n_samples), note: DATA.meta.source, cls: "" }},
      ];
      document.getElementById("kpiGrid").innerHTML = cards.map(c => `
        <div class="card">
          <div class="label">${{c.label}}</div>
          <div class="value ${{c.cls}}">${{c.value}}</div>
          <div class="note">${{c.note}}</div>
        </div>`).join("");

      setTableHead("groupTable", t.gCols);
      const gBody = DATA.groups.map(r => `
        <tr>
          <td><span class="chip">${{r.group}}</span></td>
          <td>${{r.device || "-"}}</td>
          <td>${{Math.round(r.best_n)}}</td>
          <td class="${{clsMae(r.best_mae)}}">${{fmt(r.best_mae)}}</td>
          <td>${{fmt(r.best_rmse)}}</td>
          <td>${{fmt(r.best_corr)}}</td>
          <td>${{Math.round(r.pub_n)}}</td>
          <td class="${{clsMae(r.pub_mae)}}">${{fmt(r.pub_mae)}}</td>
          <td>${{fmt(r.pub_rmse)}}</td>
          <td>${{fmt(r.pub_corr)}}</td>
          <td class="${{clsCov(r.coverage)}}">${{pct(r.coverage)}}</td>
        </tr>`).join("");
      document.querySelector("#groupTable tbody").innerHTML = gBody;

      setTableHead("sampleTable", t.sCols);
      const filterText = (document.getElementById("sampleFilter").value || "").trim().toLowerCase();
      const filtered = DATA.samples.filter(r => {{
        if (!filterText) return true;
        const blob = `${{r.group}} ${{r.stem}} ${{r.device}} ${{r.condition}}`.toLowerCase();
        return blob.includes(filterText);
      }});
      const sBody = filtered.map(r => `
        <tr>
          <td>${{r.group}}</td>
          <td>${{r.stem}}</td>
          <td>${{r.device || "-"}}</td>
          <td>${{r.condition || "-"}}</td>
          <td>${{Math.round(r.best_n)}}</td>
          <td class="${{clsMae(r.best_mae)}}">${{fmt(r.best_mae)}}</td>
          <td>${{fmt(r.best_rmse)}}</td>
          <td>${{Math.round(r.pub_n)}}</td>
          <td class="${{clsMae(r.pub_mae)}}">${{fmt(r.pub_mae)}}</td>
          <td>${{fmt(r.pub_rmse)}}</td>
          <td class="${{clsCov(r.coverage)}}">${{pct(r.coverage)}}</td>
          <td>${{fmt(r.ecg_mean, 2)}}</td>
          <td>${{fmt(r.best_mean, 2)}}</td>
          <td>${{fmt(r.pub_mean, 2)}}</td>
          <td>${{fmt(r.best_bias, 2)}}</td>
          <td>${{fmt(r.pub_bias, 2)}}</td>
        </tr>`).join("");
      document.querySelector("#sampleTable tbody").innerHTML = sBody;

      // Visualization controls
      const metricSel = document.getElementById("vizMetric");
      const groupSel = document.getElementById("vizGroup");
      const topNInput = document.getElementById("vizTopN");
      const topNVal = document.getElementById("vizTopNVal");
      const metricKeys = Object.keys(t.metricOpts);
      const prevMetric = metricSel.value || metricSel.dataset.last || "pub_mae";
      metricSel.innerHTML = metricKeys
        .map(km => `<option value="${{km}}">${{t.metricOpts[km]}}</option>`)
        .join("");
      metricSel.value = metricKeys.includes(prevMetric) ? prevMetric : "pub_mae";
      metricSel.dataset.last = metricSel.value;
      const groups = Array.from(new Set(DATA.samples.map(r => r.group))).sort();
      const prevGroup = groupSel.value || groupSel.dataset.last || "ALL";
      groupSel.innerHTML = [`<option value="ALL">${{t.allGroups}}</option>`]
        .concat(groups.map(g => `<option value="${{g}}">${{g}}</option>`))
        .join("");
      groupSel.value = (prevGroup === "ALL" || groups.includes(prevGroup)) ? prevGroup : "ALL";
      groupSel.dataset.last = groupSel.value;
      if (!topNInput.dataset.inited) {{
        topNInput.value = "12";
        topNInput.dataset.inited = "1";
      }}
      topNVal.textContent = topNInput.value;

      const metric = metricSel.value || "pub_mae";
      const pickGroup = groupSel.value || "ALL";
      const topN = Math.max(5, Math.min(30, Number(topNInput.value || 12)));

      const devRows = DATA.groups
        .slice()
        .sort((a, b) => Number(b[metric]) - Number(a[metric]));
      drawBarPlot("devicePlot", devRows, metric, r => `${{r.group}} · ${{r.device || "-"}}`, t.noData);

      const sampleRows = filtered
        .filter(r => pickGroup === "ALL" ? true : r.group === pickGroup)
        .slice()
        .sort((a, b) => Number(b[metric]) - Number(a[metric]))
        .slice(0, topN);
      drawBarPlot("samplePlot", sampleRows, metric, r => `${{r.group}}/${{r.stem}}`, t.noData);

      // C1~C6 per-second trend controls
      const tsCondSel = document.getElementById("tsCond");
      const tsMetricSel = document.getElementById("tsMetric");
      const tsChannelSel = document.getElementById("tsChannel");

      const conditions = (DATA.meta.conditions || []).slice();
      const prevTsCond = tsCondSel.value || tsCondSel.dataset.last || "ALL";
      tsCondSel.innerHTML = [`<option value="ALL">${{t.tsAllConds}}</option>`]
        .concat(conditions.map(c => `<option value="${{c}}">${{c}}</option>`))
        .join("");
      tsCondSel.value = (prevTsCond === "ALL" || conditions.includes(prevTsCond)) ? prevTsCond : "ALL";
      tsCondSel.dataset.last = tsCondSel.value;

      const tsMetricOpts = t.tsMetricOpts || {{ hr: "HR", abs_err: "|err|" }};
      const tsMetricKeys = Object.keys(tsMetricOpts);
      const prevTsMetric = tsMetricSel.value || tsMetricSel.dataset.last || "hr";
      tsMetricSel.innerHTML = tsMetricKeys.map(km => `<option value="${{km}}">${{tsMetricOpts[km]}}</option>`).join("");
      tsMetricSel.value = tsMetricKeys.includes(prevTsMetric) ? prevTsMetric : "hr";
      tsMetricSel.dataset.last = tsMetricSel.value;

      const tsChannelOpts = t.tsChannelOpts || {{ best: "best", published: "published" }};
      const tsChannelKeys = Object.keys(tsChannelOpts);
      const prevTsChannel = tsChannelSel.value || tsChannelSel.dataset.last || "published";
      tsChannelSel.innerHTML = tsChannelKeys.map(kc => `<option value="${{kc}}">${{tsChannelOpts[kc]}}</option>`).join("");
      tsChannelSel.value = tsChannelKeys.includes(prevTsChannel) ? prevTsChannel : "published";
      tsChannelSel.dataset.last = tsChannelSel.value;

      drawTimeSeriesPlot(
        DATA.series || [],
        tsCondSel.value || "ALL",
        tsMetricSel.value || "hr",
        tsChannelSel.value || "published",
        t.noData
      );

      // Parameter glossary
      setTableHead("paramTable", t.pCols);
      document.querySelector("#paramTable tbody").innerHTML = (t.params || []).map(row => `
        <tr>
          <td><span class="chip">${{row[0]}}</span></td>
          <td>${{row[1]}}</td>
          <td>${{row[2]}}</td>
          <td>${{row[3]}}</td>
        </tr>
      `).join("");
    }}

    const langSel = document.getElementById("langSel");
    const sampleFilter = document.getElementById("sampleFilter");
    const vizMetric = document.getElementById("vizMetric");
    const vizGroup = document.getElementById("vizGroup");
    const vizTopN = document.getElementById("vizTopN");
    const tsCond = document.getElementById("tsCond");
    const tsMetric = document.getElementById("tsMetric");
    const tsChannel = document.getElementById("tsChannel");
    langSel.addEventListener("change", () => render(langSel.value));
    sampleFilter.addEventListener("input", () => render(langSel.value));
    vizMetric.addEventListener("change", () => render(langSel.value));
    vizGroup.addEventListener("change", () => render(langSel.value));
    vizTopN.addEventListener("input", () => render(langSel.value));
    tsCond.addEventListener("change", () => render(langSel.value));
    tsMetric.addEventListener("change", () => render(langSel.value));
    tsChannel.addEventListener("change", () => render(langSel.value));
    render("zh");
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HTML report for Data2_as_Data1 evaluation results")
    parser.add_argument(
        "--best-summary-csv",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/data2_as_data1_eval_20260423_rerun/rppg_ecg_summary_best_opencv_timestamp.csv",
    )
    parser.add_argument(
        "--published-summary-csv",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/data2_as_data1_eval_20260423_rerun/rppg_ecg_summary_published_opencv_timestamp.csv",
    )
    parser.add_argument(
        "--best-detail-csv",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/data2_as_data1_eval_20260423_rerun/rppg_ecg_comparison_best_opencv_timestamp.csv",
    )
    parser.add_argument(
        "--published-detail-csv",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/data2_as_data1_eval_20260423_rerun/rppg_ecg_comparison_published_opencv_timestamp.csv",
    )
    parser.add_argument(
        "--manifest-csv",
        default="/Users/liangwenwang/Downloads/Code/Demo2/Data2_as_Data1/manifest_data2_as_data1.csv",
    )
    parser.add_argument(
        "--out-html",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/visualizations/data2_eval_report.html",
    )
    parser.add_argument(
        "--include-devices",
        default="",
        help="comma-separated device names to include (exact match from manifest)",
    )
    parser.add_argument(
        "--exclude-devices",
        default="",
        help="comma-separated device names to exclude (exact match from manifest)",
    )
    args = parser.parse_args()

    best_summary_path = Path(args.best_summary_csv)
    pub_summary_path = Path(args.published_summary_csv)
    best_detail_path = Path(args.best_detail_csv)
    pub_detail_path = Path(args.published_detail_csv)
    manifest_path = Path(args.manifest_csv)
    out_html = Path(args.out_html)

    for p in [best_summary_path, pub_summary_path, best_detail_path, pub_detail_path]:
        if not p.exists():
            raise RuntimeError(f"required CSV not found: {p}")

    best_summary = _read_csv(best_summary_path)
    pub_summary = _read_csv(pub_summary_path)
    best_detail = _read_csv(best_detail_path)
    pub_detail = _read_csv(pub_detail_path)
    manifest = _read_csv(manifest_path) if manifest_path.exists() else []
    include_devices = _parse_token_set(args.include_devices)
    exclude_devices = _parse_token_set(args.exclude_devices)

    payload = build_payload(
        best_summary=best_summary,
        pub_summary=pub_summary,
        best_detail=best_detail,
        pub_detail=pub_detail,
        manifest_rows=manifest,
        include_devices=include_devices,
        exclude_devices=exclude_devices,
    )
    html = render_html(payload, "Data2 RPPG Evaluation Report")

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"[OK] report: {out_html}")


if __name__ == "__main__":
    main()
