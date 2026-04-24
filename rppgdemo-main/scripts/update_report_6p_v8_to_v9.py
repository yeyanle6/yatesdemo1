#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Row:
    stem: str
    sec: float
    ecg: float
    est: float


CONDS: List[Tuple[str, str, str]] = [
    ("C1", "標準条件", "オフィス通常照明・正面・静止・通常呼吸"),
    ("C2", "頭部の左右移動", "標準条件下で頭を左右に動かす"),
    ("C3", "通常会話", "標準条件下で会話を行う"),
    ("C4", "PC作業 (うつむき)", "標準条件下でうつむき姿勢で PC を使用"),
    ("C5", "PC画面光のみ", "室内照明オフ・PC画面の光のみ・静止"),
    ("C6", "自然光のみ", "室内照明オフ・窓外の少量自然光のみ"),
]


def _read_detail(path: Path) -> List[Row]:
    out: List[Row] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            stem = row.get("stem", "")
            try:
                sec = float(row["sec"])
                ecg = float(row["ecg_hr"])
                est = float(row["est_hr"])
            except Exception:
                continue
            out.append(Row(stem=stem, sec=sec, ecg=ecg, est=est))
    return out


def _cls(mae: float | None, n: int) -> str:
    if n <= 0 or mae is None or not math.isfinite(mae):
        return "cm-na"
    if mae <= 3.0:
        return "cm-good"
    if mae <= 5.0:
        return "cm-warn"
    return "cm-bad"


def _fmt_mae(mae: float | None) -> str:
    if mae is None or not math.isfinite(mae):
        return "取得失敗"
    return f"MAE {mae:.2f}<span class=\"cm-unit\">bpm</span>"


def _nice_step(v: float) -> float:
    if v <= 0:
        return 1.0
    exp = math.floor(math.log10(v))
    base = 10**exp
    x = v / base
    if x <= 1:
        m = 1
    elif x <= 2:
        m = 2
    elif x <= 5:
        m = 5
    else:
        m = 10
    return m * base


def _path(points: List[Tuple[float, float]]) -> str:
    if not points:
        return ""
    return " ".join(
        f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points)
    )


def _circles(points: List[Tuple[float, float]], color: str, r: float = 1.3) -> str:
    if not points:
        return ""
    return "\n".join(
        f"<circle cx=\"{x:.1f}\" cy=\"{y:.1f}\" r=\"{r:.1f}\" fill=\"{color}\" opacity=\"0.85\"/>"
        for x, y in points
    )


def _build_card(cond: str, title: str, desc: str, rows: List[Row]) -> str:
    pc_stem = f"{cond}-lenovo"
    mb_stem = f"{cond}-iphone16e"

    pc_rows = sorted([r for r in rows if r.stem == pc_stem], key=lambda x: x.sec)
    mb_rows = sorted([r for r in rows if r.stem == mb_stem], key=lambda x: x.sec)

    ecg_by_sec: Dict[float, float] = {}
    pc_by_sec: Dict[float, float] = {}
    mb_by_sec: Dict[float, float] = {}
    for r in pc_rows + mb_rows:
        ecg_by_sec[r.sec] = r.ecg
    for r in pc_rows:
        pc_by_sec[r.sec] = r.est
    for r in mb_rows:
        mb_by_sec[r.sec] = r.est

    ecg_secs = sorted(ecg_by_sec.keys())
    x_min = min(ecg_secs) if ecg_secs else 0.0
    x_max = max(ecg_secs) if ecg_secs else 1.0
    if x_max <= x_min:
        x_max = x_min + 1.0

    y_vals: List[float] = list(ecg_by_sec.values()) + list(pc_by_sec.values()) + list(mb_by_sec.values())
    if not y_vals:
        y_vals = [60.0, 90.0]
    y_min_raw, y_max_raw = min(y_vals), max(y_vals)
    pad = max(3.0, (y_max_raw - y_min_raw) * 0.10)
    y_min, y_max = y_min_raw - pad, y_max_raw + pad
    if y_max <= y_min:
        y_max = y_min + 1.0

    # chart geometry
    vw, vh = 380.0, 200.0
    xl, xr = 36.0, 372.0
    yt, yb = 8.0, 176.0

    def sx(sec: float) -> float:
        return xl + (sec - x_min) / (x_max - x_min) * (xr - xl)

    def sy(v: float) -> float:
        return yb - (v - y_min) / (y_max - y_min) * (yb - yt)

    ecg_pts = [(sx(s), sy(ecg_by_sec[s])) for s in ecg_secs]
    pc_pts = [(sx(s), sy(pc_by_sec[s])) for s in sorted(pc_by_sec.keys())]
    mb_pts = [(sx(s), sy(mb_by_sec[s])) for s in sorted(mb_by_sec.keys())]

    # y grid
    y_tick_count = 6
    y_grid = []
    for i in range(y_tick_count):
        vv = y_min + (y_max - y_min) * i / (y_tick_count - 1)
        yy = sy(vv)
        y_grid.append(
            f"<line x1=\"{xl:.0f}\" y1=\"{yy:.1f}\" x2=\"{xr:.0f}\" y2=\"{yy:.1f}\" stroke=\"#9ca3af\" stroke-opacity=\"0.35\" stroke-dasharray=\"2 3\" stroke-width=\"0.5\"/>"
            f"\n<text x=\"32\" y=\"{yy+3:.1f}\" font-family=\"JetBrains Mono,monospace\" font-size=\"8.5\" fill=\"#111827\" text-anchor=\"end\">{vv:.0f}</text>"
        )

    # x ticks
    sec_range = max(1.0, x_max - x_min)
    step = _nice_step(sec_range / 3.0)
    ticks: List[float] = []
    start = math.ceil(x_min / step) * step
    if abs(start - x_min) > 1e-6:
        ticks.append(x_min)
    v = start
    while v <= x_max + 1e-6:
        ticks.append(v)
        v += step
    ticks = sorted(set(round(t, 6) for t in ticks))
    if len(ticks) > 6:
        ticks = ticks[::2]
    if ticks and ticks[-1] < x_max - 0.5 * step:
        ticks.append(x_max)

    x_tick_lines = []
    for t in ticks:
        xx = sx(t)
        label = int(round(t))
        x_tick_lines.append(
            f"<line x1=\"{xx:.1f}\" y1=\"176\" x2=\"{xx:.1f}\" y2=\"179\" stroke=\"#111827\" stroke-width=\"0.5\"/>"
            f"\n<text x=\"{xx:.1f}\" y=\"188\" font-family=\"JetBrains Mono,monospace\" font-size=\"8.5\" fill=\"#111827\" text-anchor=\"middle\">{label}</text>"
        )

    # metrics
    n_pc = len(pc_rows)
    n_mb = len(mb_rows)
    mae_pc = (sum(abs(r.est - r.ecg) for r in pc_rows) / n_pc) if n_pc > 0 else None
    mae_mb = (sum(abs(r.est - r.ecg) for r in mb_rows) / n_mb) if n_mb > 0 else None

    cls_pc = _cls(mae_pc, n_pc)
    cls_mb = _cls(mae_mb, n_mb)

    mb_path_html = (
        f"<path d=\"{_path(mb_pts)}\" stroke=\"#ea580c\" stroke-width=\"1.4\" fill=\"none\" opacity=\"0.95\"/>"
        if mb_pts
        else ""
    )
    mb_circles_html = _circles(mb_pts, "#ea580c", 1.15) if mb_pts else ""

    card = f"""
<div class="c-card">
  <div class="c-card-head">
    <span class="c-tag-big">{cond}</span>
    <span class="c-title-big">{title}</span>
  </div>
  <div class="c-desc-big">{desc}</div>
  <div class="c-chart"><svg viewBox="0 0 380 200" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet" style="width:100%;height:auto">
{chr(10).join(y_grid)}
{chr(10).join(x_tick_lines)}
<line x1="{xl:.0f}" y1="{yt:.0f}" x2="{xl:.0f}" y2="{yb:.0f}" stroke="#111827" stroke-width="1.1"/>
<line x1="{xl:.0f}" y1="{yb:.0f}" x2="{xr:.0f}" y2="{yb:.0f}" stroke="#111827" stroke-width="1.1"/>
<text x="6" y="92.0" font-family="Yu Gothic UI,sans-serif" font-size="8" fill="#111827" transform="rotate(-90 6,92.0)" text-anchor="middle">心拍 (bpm)</text>
<text x="204.0" y="197" font-family="Yu Gothic UI,sans-serif" font-size="8" fill="#111827" text-anchor="middle">秒</text>
<path d="{_path(ecg_pts)}" stroke="#dc2626" stroke-width="1.9" fill="none"/>
{_circles(ecg_pts, "#dc2626", 1.25)}
<path d="{_path(pc_pts)}" stroke="#1d4ed8" stroke-width="1.4" fill="none" opacity="0.95"/>
{_circles(pc_pts, "#1d4ed8", 1.15)}
{mb_path_html}
{mb_circles_html}
</svg></div>
  <div class="c-metrics">
    <div class="cm {cls_pc}">
      <div class="cm-row1"><span class="cm-label">社内 PC</span><span class="cm-n">{f"n={n_pc}秒" if n_pc > 0 else "―"}</span></div>
      <div class="cm-val">{_fmt_mae(mae_pc)}</div>
    </div>
    <div class="cm {cls_mb}">
      <div class="cm-row1"><span class="cm-label">社内手机</span><span class="cm-n">{f"n={n_mb}秒" if n_mb > 0 else "―"}</span></div>
      <div class="cm-val">{_fmt_mae(mae_mb)}</div>
    </div>
  </div>
</div>"""
    return card


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch v8 report into v9 with limited updates.")
    ap.add_argument(
        "--input-html",
        default="/Users/liangwenwang/Downloads/rppg_精度向上開発_報告書_6p_v8.html",
    )
    ap.add_argument(
        "--data2-best-comparison",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/strict_20260424_data2_noiphone13/rppg_ecg_comparison_best_opencv_timestamp.csv",
    )
    ap.add_argument(
        "--output-html",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/visualizations/rppg_精度向上開発_報告書_6p_v9.html",
    )
    args = ap.parse_args()

    input_html = Path(args.input_html)
    csv_path = Path(args.data2_best_comparison)
    output_html = Path(args.output_html)

    text = input_html.read_text(encoding="utf-8")
    rows = _read_detail(csv_path)

    # Slide 4 phrase neutralization
    text = text.replace(
        "        <b>状態別の傾向:</b> C1～C4 (標準照明系) は両デバイス MAE 1～3 bpm と高精度。<b>低光環境 (C5: PC画面光、C6: 自然光)</b> で一部誤差増大あり。詳細は次ページ参照。",
        "        <b>状態別の傾向:</b> C1～C4 (標準照明系) は両デバイス MAE 1～3 bpm と高精度。C5 (PC 画面光のみ)・C6 (自然光のみ) の低光環境では一部誤差が増大する傾向が見られた。詳細は次ページの推移グラフ参照。",
    )

    # Slide 6 annotation for "7名"
    text = text.replace(
        "<tr><td>被験者数</td><td class=\"num\">7 名</td><td class=\"num\"><b>14 ～ 20 名</b></td></tr>",
        "<tr><td>被験者数</td><td class=\"num\">7 名</td><td class=\"num\"><b>14 ～ 20 名</b></td></tr>\n"
        "          <tr><td></td><td colspan=\"2\" class=\"small\">※ Data1 の取得セッションは 9 件あるが、同一被験者の重複を除いた実数は 7 名</td></tr>",
    )

    # Ensure legend ECG color
    text = re.sub(
        r"\.leg-line\.ecg-line\{background:[^;]+;height:2\.5px\}",
        ".leg-line.ecg-line{background:#dc2626;height:2.5px}",
        text,
    )

    # Regenerate slide-5 six cards from comparison-best CSV
    cards = [_build_card(cond, title, desc, rows) for cond, title, desc in CONDS]
    cards_html = "\n\n".join(cards)

    patt = re.compile(
        r"(<!-- ================= SLIDE 5:.*?<div class=\"c-cards-grid\">)(.*?)(</div>\s*\n\n  <div class=\"slide-foot\"><span>出典: rppg_ecg_comparison_best_opencv_timestamp\.csv \(全秒データ\)</span><span>p\. 5 / 6</span></div>\s*</section>)",
        flags=re.S,
    )
    m = patt.search(text)
    if not m:
        raise RuntimeError("failed to locate slide-5 chart block in input html")
    text = text[: m.start(2)] + "\n" + cards_html + "\n  " + text[m.end(2) :]

    # Append change log comment
    comment = """\n<!-- v9 changes:
     - Slide 4: 状態別傾向の文言を中性化
     - Slide 5: ECG 線色を赤に統一 (#dc2626), legend 同期
     - Slide 5: 坐标轴と刻度を黑系 (#111827) に統一
     - Slide 5: 网格線を淡色・点線化 (#9ca3af, dash 2 3)
     - Slide 5: 6 枚の SVG を comparison_best CSV から再生成
     - Slide 5: 各線に小円点 (r=1.15〜1.25) を追加
     - Slide 6: 被験者数 7 名に補足注釈を追加
     - Slide 2/3/4/6 の既存構成は必要箇所以外を維持
-->\n"""
    text = re.sub(r"\s*</html>\s*$", comment + "</html>\n", text, flags=re.S)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(text, encoding="utf-8")
    print(f"[OK] wrote {output_html}")


if __name__ == "__main__":
    main()
