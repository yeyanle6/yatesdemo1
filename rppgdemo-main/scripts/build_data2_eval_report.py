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

    payload = {
        "overall": {
            "best": best_overall,
            "published": pub_overall,
            "coverage": _safe_ratio(float(pub_overall["n"]), float(best_overall["n"])),
        },
        "groups": group_rows,
        "samples": sample_table,
        "meta": {
            "n_samples": len(sample_table),
            "source": "Data2_as_Data1",
            "include_devices": sorted(include_devices),
            "exclude_devices": sorted(exclude_devices),
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
    .wrap {{
      max-width: 1280px;
      margin: 24px auto 40px;
      padding: 0 16px;
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
  </style>
</head>
<body>
  <div class="wrap">
    <header class="head">
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

    <section class="grid" id="kpiGrid"></section>

    <section class="panel">
      <h2 id="groupTitle"></h2>
      <div class="table-wrap">
        <table id="groupTable">
          <thead></thead>
          <tbody></tbody>
        </table>
      </div>
    </section>

    <section class="panel">
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
  </div>

  <script>
    const DATA = {data_json};
    const I18N = {{
      zh: {{
        title: "Data2 RPPG 评估报告（Data1 格式）",
        subtitle: "对比通道：best 与 published；展示整体、分设备、分文件指标",
        lang: "语言",
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
      }},
      en: {{
        title: "Data2 RPPG Evaluation Report (Data1-style)",
        subtitle: "Channels compared: best vs published; includes overall, per-device, and per-file metrics.",
        lang: "Language",
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
      }},
      ja: {{
        title: "Data2 RPPG 評価レポート（Data1互換）",
        subtitle: "best と published を比較し、全体・デバイス別・ファイル別の指標を表示します。",
        lang: "言語",
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

    function render(lang) {{
      const t = I18N[lang] || I18N.zh;
      document.documentElement.lang = lang === "ja" ? "ja" : (lang === "en" ? "en" : "zh");
      document.getElementById("title").textContent = t.title;
      document.getElementById("subtitle").textContent = t.subtitle;
      document.getElementById("langLabel").textContent = t.lang;
      document.getElementById("groupTitle").textContent = t.groupTitle;
      document.getElementById("sampleTitle").textContent = t.sampleTitle;
      document.getElementById("sampleFilter").placeholder = t.search;
      document.getElementById("footNote").textContent = t.foot;

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
    }}

    const langSel = document.getElementById("langSel");
    const sampleFilter = document.getElementById("sampleFilter");
    langSel.addEventListener("change", () => render(langSel.value));
    sampleFilter.addEventListener("input", () => render(langSel.value));
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
