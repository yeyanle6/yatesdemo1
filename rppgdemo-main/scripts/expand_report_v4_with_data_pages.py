#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import math
import re
from pathlib import Path
from typing import Dict, List


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return math.nan


def _fmt(v: float, nd: int = 2) -> str:
    if not math.isfinite(v):
        return "-"
    return f"{v:.{nd}f}"


def _pct_ratio(v: float, nd: int = 1) -> str:
    if not math.isfinite(v):
        return "-"
    return f"{v * 100:.{nd}f}%"


def _pct(v: float, nd: int = 1) -> str:
    if not math.isfinite(v):
        return "-"
    return f"{v:.{nd}f}%"


def _find_dataset(rows: List[Dict[str, str]], name: str) -> Dict[str, str]:
    for r in rows:
        if (r.get("dataset") or "") == name:
            return r
    raise RuntimeError(f"dataset not found in overall csv: {name}")


def _make_group_rows(per_group: List[Dict[str, str]], dataset: str) -> str:
    rows = [r for r in per_group if (r.get("dataset") or "") == dataset]
    rows.sort(key=lambda r: (r.get("group") or ""))
    out = []
    for r in rows:
        out.append(
            "<tr>"
            f"<td class=\"name\">{html.escape(r.get('group') or '-')}</td>"
            f"<td class=\"c-pub\">{int(float(r.get('best_n') or 0))}</td>"
            f"<td class=\"c-pub\">{_fmt(_to_float(r.get('best_mae') or ''), 2)}</td>"
            f"<td class=\"c-pub\">{_fmt(_to_float(r.get('best_accuracy') or ''), 1)}%</td>"
            f"<td class=\"c-pub\">{int(float(r.get('pub_n') or 0))}</td>"
            f"<td class=\"c-pub\">{_fmt(_to_float(r.get('pub_mae') or ''), 2)}</td>"
            f"<td class=\"c-pub\">{_fmt(_to_float(r.get('pub_accuracy') or ''), 1)}%</td>"
            f"<td class=\"c-pub\">{_pct_ratio(_to_float(r.get('coverage') or ''), 1)}</td>"
            "</tr>"
        )
    return "".join(out)


def _make_sample_rows(per_sample: List[Dict[str, str]], dataset: str, top_n: int = 6) -> str:
    rows = [r for r in per_sample if (r.get("dataset") or "") == dataset]
    rows = [r for r in rows if math.isfinite(_to_float(r.get("pub_mae") or ""))]
    rows.sort(key=lambda r: _to_float(r.get("pub_mae") or ""), reverse=True)
    out = []
    for r in rows[:top_n]:
        out.append(
            "<tr>"
            f"<td class=\"name\">{html.escape((r.get('group') or '') + '/' + (r.get('stem') or ''))}</td>"
            f"<td class=\"c-pub\">{int(float(r.get('best_n') or 0))}</td>"
            f"<td class=\"c-pub\">{_fmt(_to_float(r.get('best_mae') or ''), 2)}</td>"
            f"<td class=\"c-pub\">{int(float(r.get('pub_n') or 0))}</td>"
            f"<td class=\"c-pub\">{_fmt(_to_float(r.get('pub_mae') or ''), 2)}</td>"
            f"<td class=\"c-pub\">{_pct_ratio(_to_float(r.get('coverage') or ''), 1)}</td>"
            f"<td class=\"small\">{_fmt(_to_float(r.get('mae_delta_best_minus_pub') or ''), 2)}</td>"
            "</tr>"
        )
    return "".join(out)


def _build_slide_5(data1: Dict[str, str], data1_group_rows: str, data1_sample_rows: str) -> str:
    return f"""
<!-- ================= SLIDE 5 ================= -->
<section class="slide">
  <div class="slide-head">
    <div class="slide-title"><span class="num">05</span>Data1 詳細レポート (Strict Rerun)</div>
    <div class="slide-meta">Data1 Detailed Metrics &middot; 5 / 6</div>
  </div>

  <h3 class="sub" style="margin-top:2px">Data1 全体結果 (再計算, best/published 両通道)</h3>
  <div class="kpi-grid">
    <div class="kpi-card accent">
      <div class="label"><span class="jp">best 通道</span>MAE / Accuracy</div>
      <div class="val">{_fmt(_to_float(data1['best_mae']),2)}<span class="unit">bpm</span></div>
      <div class="prev">Accuracy: {_fmt(_to_float(data1['best_accuracy']),1)}% &middot; n={int(float(data1['best_n']))}</div>
      <div class="kpi-note">RMSE={_fmt(_to_float(data1['best_rmse']),2)} &middot; MAPE={_fmt(_to_float(data1['best_mape']),2)}%</div>
    </div>
    <div class="kpi-card accent">
      <div class="label"><span class="jp">published 通道</span>MAE / Accuracy</div>
      <div class="val">{_fmt(_to_float(data1['pub_mae']),2)}<span class="unit">bpm</span></div>
      <div class="prev">Accuracy: {_fmt(_to_float(data1['pub_accuracy']),1)}% &middot; n={int(float(data1['pub_n']))}</div>
      <div class="kpi-note">RMSE={_fmt(_to_float(data1['pub_rmse']),2)} &middot; MAPE={_fmt(_to_float(data1['pub_mape']),2)}%</div>
    </div>
    <div class="kpi-card accent">
      <div class="label"><span class="jp">coverage</span>published / best</div>
      <div class="val">{_pct_ratio(_to_float(data1['coverage']),1)}</div>
      <div class="prev">n = {int(float(data1['pub_n']))} / {int(float(data1['best_n']))}</div>
      <div class="kpi-note">summary 一致性: best MAE={_fmt(_to_float(data1['summary_best_mae']),3)}, pub MAE={_fmt(_to_float(data1['summary_pub_mae']),3)}</div>
    </div>
  </div>

  <h3 class="sub">Data1 group 别结果</h3>
  <table class="dev-table">
    <thead>
      <tr>
        <th style="width:20%">group</th>
        <th>best n</th>
        <th>best MAE</th>
        <th>best Accuracy</th>
        <th>pub n</th>
        <th>pub MAE</th>
        <th>pub Accuracy</th>
        <th>Coverage</th>
      </tr>
    </thead>
    <tbody>
      {data1_group_rows}
    </tbody>
  </table>

  <h3 class="sub">Data1 误差较大样本 (pub MAE Top)</h3>
  <table class="dev-table">
    <thead>
      <tr>
        <th style="width:34%">sample</th>
        <th>best n</th>
        <th>best MAE</th>
        <th>pub n</th>
        <th>pub MAE</th>
        <th>Coverage</th>
        <th>best-pub</th>
      </tr>
    </thead>
    <tbody>
      {data1_sample_rows}
    </tbody>
  </table>

  <div class="slide-foot"><span>出典: strict_20260424_data1 (re-run)</span><span>p. 5 / 6</span></div>
</section>
"""


def _build_slide_6(data2: Dict[str, str], data2_group_rows: str, data2_sample_rows: str) -> str:
    return f"""
<!-- ================= SLIDE 6 ================= -->
<section class="slide">
  <div class="slide-head">
    <div class="slide-title"><span class="num">06</span>Data2 詳細レポート (iPhone13 Pro 除外)</div>
    <div class="slide-meta">Data2 Detailed Metrics &middot; 6 / 6</div>
  </div>

  <h3 class="sub" style="margin-top:2px">Data2 全体结果 (group 102/103 のみ)</h3>
  <div class="kpi-grid">
    <div class="kpi-card accent">
      <div class="label"><span class="jp">best 通道</span>MAE / Accuracy</div>
      <div class="val">{_fmt(_to_float(data2['best_mae']),2)}<span class="unit">bpm</span></div>
      <div class="prev">Accuracy: {_fmt(_to_float(data2['best_accuracy']),1)}% &middot; n={int(float(data2['best_n']))}</div>
      <div class="kpi-note">RMSE={_fmt(_to_float(data2['best_rmse']),2)} &middot; MAPE={_fmt(_to_float(data2['best_mape']),2)}%</div>
    </div>
    <div class="kpi-card accent">
      <div class="label"><span class="jp">published 通道</span>MAE / Accuracy</div>
      <div class="val">{_fmt(_to_float(data2['pub_mae']),2)}<span class="unit">bpm</span></div>
      <div class="prev">Accuracy: {_fmt(_to_float(data2['pub_accuracy']),1)}% &middot; n={int(float(data2['pub_n']))}</div>
      <div class="kpi-note">RMSE={_fmt(_to_float(data2['pub_rmse']),2)} &middot; MAPE={_fmt(_to_float(data2['pub_mape']),2)}%</div>
    </div>
    <div class="kpi-card accent">
      <div class="label"><span class="jp">coverage</span>published / best</div>
      <div class="val">{_pct_ratio(_to_float(data2['coverage']),1)}</div>
      <div class="prev">n = {int(float(data2['pub_n']))} / {int(float(data2['best_n']))}</div>
      <div class="kpi-note">限定条件: iPhone13 Pro (group101) は評価対象から除外</div>
    </div>
  </div>

  <h3 class="sub">Data2 group 别结果 (102: iPhone16e / 103: Lenovo)</h3>
  <table class="dev-table">
    <thead>
      <tr>
        <th style="width:20%">group</th>
        <th>best n</th>
        <th>best MAE</th>
        <th>best Accuracy</th>
        <th>pub n</th>
        <th>pub MAE</th>
        <th>pub Accuracy</th>
        <th>Coverage</th>
      </tr>
    </thead>
    <tbody>
      {data2_group_rows}
    </tbody>
  </table>

  <h3 class="sub">Data2 误差较大样本 (pub MAE Top)</h3>
  <table class="dev-table">
    <thead>
      <tr>
        <th style="width:34%">sample</th>
        <th>best n</th>
        <th>best MAE</th>
        <th>pub n</th>
        <th>pub MAE</th>
        <th>Coverage</th>
        <th>best-pub</th>
      </tr>
    </thead>
    <tbody>
      {data2_sample_rows}
    </tbody>
  </table>

  <div class="slide-foot"><span>出典: strict_20260424_data2_noiphone13 (re-run)</span><span>p. 6 / 6 &middot; End</span></div>
</section>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Expand 4p_v4 report into 6 pages with Data1/Data2 detail pages")
    ap.add_argument(
        "--input-html",
        default="/Users/liangwenwang/Downloads/rppg_精度向上開発_報告書_4p_v4.html",
    )
    ap.add_argument(
        "--overall-csv",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/strict_20260424_analysis/data1_data2_overall.csv",
    )
    ap.add_argument(
        "--per-group-csv",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/strict_20260424_analysis/data1_data2_per_group.csv",
    )
    ap.add_argument(
        "--per-sample-csv",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/strict_20260424_analysis/data1_data2_per_sample.csv",
    )
    ap.add_argument(
        "--output-html",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/visualizations/rppg_精度向上開発_報告書_6p_v1.html",
    )
    args = ap.parse_args()

    in_html = Path(args.input_html)
    overall_csv = Path(args.overall_csv)
    group_csv = Path(args.per_group_csv)
    sample_csv = Path(args.per_sample_csv)
    out_html = Path(args.output_html)

    for p in [in_html, overall_csv, group_csv, sample_csv]:
        if not p.exists():
            raise RuntimeError(f"missing file: {p}")

    text = in_html.read_text(encoding="utf-8")
    overall_rows = _read_csv(overall_csv)
    group_rows = _read_csv(group_csv)
    sample_rows = _read_csv(sample_csv)

    d1 = _find_dataset(overall_rows, "Data1")
    d2 = _find_dataset(overall_rows, "Data2_no_iPhone13")

    d1_group_rows = _make_group_rows(group_rows, "Data1")
    d2_group_rows = _make_group_rows(group_rows, "Data2_no_iPhone13")
    d1_sample_rows = _make_sample_rows(sample_rows, "Data1", top_n=6)
    d2_sample_rows = _make_sample_rows(sample_rows, "Data2_no_iPhone13", top_n=6)

    slide5 = _build_slide_5(d1, d1_group_rows, d1_sample_rows)
    slide6 = _build_slide_6(d2, d2_group_rows, d2_sample_rows)

    # Update page counters 1/4 -> 1/6 etc.
    text = text.replace("1 / 4", "1 / 6").replace("2 / 4", "2 / 6").replace("3 / 4", "3 / 6").replace("4 / 4", "4 / 6")
    text = text.replace("p. 1 / 4", "p. 1 / 6").replace("p. 2 / 4", "p. 2 / 6").replace("p. 3 / 4", "p. 3 / 6").replace("p. 4 / 4", "p. 4 / 6")

    # Title bump
    text = text.replace("rPPG 精度向上開発 (Ver3) 報告書", "rPPG 精度向上開発 (Ver3) 報告書 + Data1/Data2 詳細")

    # inject before deck closing
    marker = "\n</div>\n</body>\n</html>\n"
    if marker not in text:
        raise RuntimeError("unexpected input html structure: closing marker not found")
    text = text.replace(marker, f"\n{slide5}\n{slide6}\n</div>\n</body>\n</html>\n")

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(text, encoding="utf-8")
    print(f"[OK] output: {out_html}")


if __name__ == "__main__":
    main()
