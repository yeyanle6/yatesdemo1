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
class Pt:
    stem: str
    sec: float
    ecg: float
    est: float
    err: float


def _read_subject_points(csv_path: Path, group_id: str) -> List[Pt]:
    out: List[Pt] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("group") != group_id:
                continue
            try:
                sec = float(row["sec"])
                ecg = float(row["ecg_hr"])
                est = float(row["est_hr"])
            except Exception:
                continue
            out.append(Pt(stem=row.get("stem", ""), sec=sec, ecg=ecg, est=est, err=abs(est - ecg)))
    stem_order = {"1-1": 0, "1-2": 1, "1-3": 2}
    out.sort(key=lambda p: (stem_order.get(p.stem, 99), p.sec))
    return out


def _warmup_second(points: List[Pt], window: int = 10, mae_th: float = 3.0) -> float | None:
    if len(points) < window:
        return None
    errs = [p.err for p in points]
    for i in range(0, len(points) - window + 1):
        w = errs[i : i + window]
        if (sum(w) / window) <= mae_th:
            return points[i].sec
    return None


def _build_subject_chart_svg(points: List[Pt], warmup_by_stem: Dict[str, float | None]) -> str:
    if not points:
        return "<div class='box-note warn'><b>可視化データなし:</b> 被験者 1 の有効データを読み込めませんでした。</div>"

    w, h = 940, 250
    ml, mt, mr, mb = 46, 16, 16, 34
    pw = w - ml - mr
    ph = h - mt - mb

    y_vals = [p.ecg for p in points] + [p.est for p in points]
    y_min, y_max = min(y_vals), max(y_vals)
    pad = max(3.0, (y_max - y_min) * 0.08)
    y0, y1 = y_min - pad, y_max + pad

    def x(i: int) -> float:
        return ml + (i / max(1, len(points) - 1)) * pw

    def y(v: float) -> float:
        return mt + (1.0 - (v - y0) / (y1 - y0)) * ph

    ecg_path = " ".join(
        f"{'M' if i == 0 else 'L'}{x(i):.1f},{y(p.ecg):.1f}" for i, p in enumerate(points)
    )
    est_path = " ".join(
        f"{'M' if i == 0 else 'L'}{x(i):.1f},{y(p.est):.1f}" for i, p in enumerate(points)
    )

    stems = ["1-1", "1-2", "1-3"]
    stem_points: Dict[str, List[Tuple[int, Pt]]] = {k: [] for k in stems}
    for i, p in enumerate(points):
        if p.stem in stem_points:
            stem_points[p.stem].append((i, p))

    boundaries: List[float] = []
    labels: List[str] = []
    for s in stems:
        spts = stem_points[s]
        if not spts:
            continue
        i0 = spts[0][0]
        i1 = spts[-1][0]
        if i0 > 0:
            boundaries.append(x(i0))
        labels.append(
            f"<text x='{(x(i0)+x(i1))/2:.1f}' y='14' font-size='10.5' fill='#111827' text-anchor='middle'>{s}</text>"
        )

    boundary_lines = "\n".join(
        f"<line x1='{bx:.1f}' y1='{mt}' x2='{bx:.1f}' y2='{mt+ph}' stroke='#111827' stroke-opacity='0.45' stroke-dasharray='3 3' stroke-width='1'/>"
        for bx in boundaries
    )

    warmup_lines: List[str] = []
    warmup_texts: List[str] = []
    for stem in stems:
        target_sec = warmup_by_stem.get(stem)
        if target_sec is None:
            continue
        spts = stem_points.get(stem, [])
        idx = None
        for i, p in spts:
            if p.sec >= target_sec:
                idx = i
                break
        if idx is None:
            continue
        xx = x(idx)
        warmup_lines.append(
            f"<line x1='{xx:.1f}' y1='{mt}' x2='{xx:.1f}' y2='{mt+ph}' stroke='#dc2626' stroke-dasharray='2 2' stroke-width='1.1'/>"
        )
        warmup_texts.append(
            f"<text x='{xx:.1f}' y='{mt+ph+16}' font-size='9.5' fill='#111827' text-anchor='middle'>{stem}: ~{int(target_sec)}s</text>"
        )

    y_ticks = []
    for i in range(5):
        vv = y0 + (y1 - y0) * i / 4
        yy = y(vv)
        y_ticks.append(
            f"<line x1='{ml}' y1='{yy:.1f}' x2='{ml+pw}' y2='{yy:.1f}' stroke='#111827' stroke-opacity='0.15' stroke-width='1'/>"
            f"<text x='{ml-6}' y='{yy+3:.1f}' font-size='9.5' fill='#111827' text-anchor='end'>{vv:.0f}</text>"
        )

    return f"""
<div class="subject-chart-wrap">
  <svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
    {''.join(y_ticks)}
    <line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#111827" stroke-width="1.2"/>
    <line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#111827" stroke-width="1.2"/>
    {boundary_lines}
    {' '.join(labels)}
    <path d="{ecg_path}" stroke="#dc2626" stroke-width="1.5" fill="none" opacity="0.95"/>
    <path d="{est_path}" stroke="#1d4ed8" stroke-width="1.3" fill="none" opacity="0.93"/>
    {''.join(warmup_lines)}
    {''.join(warmup_texts)}
    <text x="{w/2:.1f}" y="{h-4}" font-size="10" fill="#111827" text-anchor="middle">サンプル順 (被験者1: 1-1 → 1-2 → 1-3)</text>
  </svg>
</div>
"""


def _replace_once(src: str, old: str, new: str) -> str:
    if old not in src:
        raise RuntimeError("target block not found for replacement")
    return src.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Update 6p_v7 report to v8 with user-requested edits.")
    ap.add_argument(
        "--input-html",
        default="/Users/liangwenwang/Downloads/rppg_精度向上開発_報告書_6p_v7.html",
    )
    ap.add_argument(
        "--data1-published-csv",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/strict_20260424_data1/rppg_ecg_comparison_published_opencv_timestamp.csv",
    )
    ap.add_argument(
        "--output-html",
        default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/visualizations/rppg_精度向上開発_報告書_6p_v8.html",
    )
    args = ap.parse_args()

    in_html = Path(args.input_html)
    out_html = Path(args.output_html)
    data_csv = Path(args.data1_published_csv)

    text = in_html.read_text(encoding="utf-8")

    # 1) CSS additions
    css_anchor = ".leg-line.mb-line{background:#ea580c}\n</style>"
    css_add = """
  .subject-chart-wrap{margin-top:8px;border:1px solid var(--line-2);background:#fff;padding:7px 8px}
  .subject-legend{display:flex;gap:14px;align-items:center;font-size:10.5px;color:var(--ink-2);margin-top:4px}
  .subject-legend .leg{display:inline-flex;align-items:center;gap:5px}
  .subject-legend .line{display:inline-block;width:20px;height:2px}
  .subject-legend .line.ecg{background:#dc2626;height:2.4px}
  .subject-legend .line.rppg{background:#1d4ed8}
"""
    text = _replace_once(text, css_anchor, css_add + css_anchor)

    # 2) Slide-2 improvements: 4 points
    pattern_imp = re.compile(
        r"<h3 class=\"sub\">Ver3 の主な改善点</h3>\s*<ul>.*?</ul>\s*</div>",
        flags=re.S,
    )
    new_imp = """<h3 class="sub">Ver3 の主な改善点</h3>
      <ul>
        <li><b>顔検出処理の統一</b><br>
          <span class="note">※ どの端末でも同じ顔検出方式を使うように揃え、端末依存の挙動差を抑制。</span>
        </li>
        <li><b>高品質区間の自動抽出</b><br>
          <span class="note">※ 撮影中に品質が落ちる秒を除外し、品質の高い区間を自動選択して評価。</span>
        </li>
        <li><b>信号安定性に基づくデータ選別</b><br>
          <span class="note">※ 複数の安定性指標を組み合わせ、厳選モード (published) で採用秒を決定。</span>
        </li>
        <li><b>パラメータ調整による精度向上</b><br>
          <span class="note">※ ROI / 融合 / gate 関連パラメータを再調整し、MAE を段階的に改善。</span>
        </li>
      </ul>
    </div>"""
    text, n_imp = pattern_imp.subn(new_imp, text, count=1)
    if n_imp != 1:
        raise RuntimeError("failed to patch improvement section")

    # 3) Data2 wording update
    text = _replace_once(
        text,
        '<tr><th>Data2<br><span class="small">(状態耐性の検証)</span></th><td>C1 ～ C6 の 6 状態 &times; 2 デバイス = <b>12 動画</b><span class="small"> (有効 11 本)</span></td></tr>',
        '<tr><th>Data2<br><span class="small">(環境・動作条件の検証)</span></th><td>C1 ～ C6 の 6 状態 (環境 + 動作) &times; 2 デバイス = <b>12 動画</b><span class="small"> (有効 11 本)</span></td></tr>',
    )
    text = text.replace(
        "<li><b>Data2:</b> 撮影状態 (C1～C6) を変えた動画により、状態変化への耐性を検証 (詳細 p.4 & p.5)</li>",
        "<li><b>Data2:</b> 環境・動作条件 (C1～C6) を変えた動画により、条件変化への耐性を検証 (詳細 p.4 & p.5)</li>",
    )
    text = text.replace(
        "検証結果 ② &mdash; Data2 全体 (撮影状態 C1 ～ C6)",
        "検証結果 ② &mdash; Data2 全体 (環境・動作条件 C1 ～ C6)",
    )

    # 4) Build subject-1 chart + session explanation in slide-3
    pts = _read_subject_points(data_csv, "1")
    warmups: Dict[str, float | None] = {}
    for stem in ("1-1", "1-2", "1-3"):
        spts = [p for p in pts if p.stem == stem]
        warmups[stem] = _warmup_second(spts, window=10, mae_th=3.0)
    chart_svg = _build_subject_chart_svg(pts, warmups)
    subject_block = f"""
  <h3 class="sub" style="margin-top:8px">被験者 1 の全点可視化 (厳選モード)</h3>
  <div class="subject-legend">
    <span class="leg"><span class="line ecg"></span>ECG</span>
    <span class="leg"><span class="line rppg"></span>rPPG (published)</span>
    <span class="small">赤の縦点線: 収束目安 (10秒移動窓 MAE≦3 bpm)</span>
  </div>
  {chart_svg}
  <div class="box-note info" style="margin-top:7px">
    <b>主要セッション / 補助セッションの区分理由:</b> 主要セッション (被験者 1～6) は同一撮影設計で取得した主評価データ。補助セッション (001～003) は過去取得分を含み、品質ばらつき確認目的の補助評価として分離表示。<br>
    <b>被験者1の収束目安:</b> 1-1 は約 {int(warmups['1-1'] or 0)} 秒、1-2 は約 {int(warmups['1-2'] or 0)} 秒、1-3 は約 {int(warmups['1-3'] or 0)} 秒から ECG との乖離が安定的に縮小。
  </div>
"""
    old_box = """  <div class="box-note" style="margin-top:8px">
    <b>主な結果:</b> Data1 全体で <b>精度 95.6% (目標 90% を +5.6 pt 上回る)</b> を確認。9 セッション中 8 セッションで精度 92% 以上。被験者 003 のみ厳選後サンプル数が 26 秒と少なく要確認だが、全体結果には影響軽微。
  </div>
"""
    new_box = subject_block + """
  <div class="box-note" style="margin-top:8px">
    <b>主な結果:</b> Data1 全体で <b>精度 95.6% (目標 90% を +5.6 pt 上回る)</b> を確認。主要 6 セッションは概ね安定し、補助側では被験者 003 のみサンプル数不足で要確認。
  </div>
"""
    text = _replace_once(text, old_box, new_box)

    # 5) Slide-5 visual style updates (ECG red, axis black)
    text = text.replace(".leg-line.ecg-line{background:#374151;height:2.5px}", ".leg-line.ecg-line{background:#dc2626;height:2.5px}")
    text = text.replace('stroke="#374151"', 'stroke="#dc2626"')
    text = text.replace('stroke="#e5e7eb" stroke-width="0.5"', 'stroke="#111827" stroke-opacity="0.22" stroke-width="0.5"')
    text = text.replace('stroke="#9ca3af"', 'stroke="#111827"')
    text = text.replace('fill="#9ca3af"', 'fill="#111827"')
    text = text.replace('fill="#6b7280"', 'fill="#111827"')

    # 6) Slide-6 updates per user notes
    old_p6_intro = """      <p style="font-size:12.5px">Data1 (主要 6 名) と Data2 (撮影状態 11 本) では、<b>年齢・性別・肌色</b>といった被験者属性の違いによる影響を確認できていない。
      Ver3 が達成した精度 (96.7%) を<b>多様な属性の被験者でも維持できるか</b>を確認するため、次フェーズで以下の拡充を提案したい。</p>"""
    new_p6_intro = """      <p style="font-size:12.5px">現時点の主評価対象は <b>被験者 7 名</b>。一方で、<b>性別比・年齢層は未管理</b>であり、属性分布に対する頑健性は未検証。
      Ver3 の目標 (精度 90%) を多様な条件でも再現できるか確認するため、次フェーズでは取得条件と運用定義を拡張したい。</p>"""
    text = _replace_once(text, old_p6_intro, new_p6_intro)

    old_tbl = """          <tr><td>被験者数</td><td class="num">6 名</td><td class="num"><b>14 ～ 20 名</b></td></tr>
          <tr><td>性別バランス</td><td>未管理</td><td>男女 4 : 1</td></tr>
          <tr><td>年齢層</td><td>偏り有</td><td>20 / 30 / 40 / 50+ 代<br><span class="small">各 3 ～ 5 名</span></td></tr>
          <tr><td>明るさ条件</td><td>1 種相当</td><td>明 / 中 / 暗 の 3 段階</td></tr>
          <tr><td>動きの有無</td><td>静止主体</td><td>静止 &middot; 微動 &middot; 会話 の 3 条件</td></tr>
          <tr><td>デバイス</td><td class="num">2 機種</td><td class="num">3 ～ 5 機種</td></tr>"""
    new_tbl = """          <tr><td>被験者数</td><td class="num">7 名</td><td class="num"><b>14 ～ 20 名</b></td></tr>
          <tr><td>性別バランス</td><td>未管理</td><td>次フェーズで管理開始</td></tr>
          <tr><td>年齢層</td><td>未管理</td><td>年代分布を収集して評価軸に追加</td></tr>
          <tr><td>明るさ条件</td><td>2 種 (照明ON/OFF系)<br><span class="small">※ lux 実測値なし</span></td><td>条件定義 + lux 計測導入</td></tr>
          <tr><td>動きの有無</td><td>静止 &middot; 微動 &middot; 会話 の 3 条件</td><td>同条件でサンプル増</td></tr>
          <tr><td>デバイス</td><td class="num">2 機種</td><td class="num">多デバイス対応 / SDK 化を見据え拡張</td></tr>"""
    text = _replace_once(text, old_tbl, new_tbl)

    text = _replace_once(
        text,
        """      <div class="box-note info" style="margin-top:10px">
        <b>ご相談事項:</b> 上記規模の取得にあたり、被験者リクルーティングの体制・予算・スケジュールについてご確認いただきたい。
      </div>""",
        """      <div class="box-note info" style="margin-top:10px">
        <b>ご相談事項:</b> 上記規模の取得にあたり、被験者リクルーティングの体制・予算・スケジュールについてご確認いただきたい。
      </div>
      <div class="small" style="margin-top:6px;color:#374151">※ 明るさ条件の定量管理のため、照度計など環境計測デバイスの購買が必要。</div>""",
    )

    # Remove low-light item and update roadmap wording
    text = _replace_once(
        text,
        """      <div class="todo-item">
        <div class="chk"></div>
        <div class="txt"><b>低光環境 (C5/C6) の精度改善検討</b>C5 (PC画面光のみ) と C6 (自然光のみ) で一部誤差増大が確認された。低光下の信号品質改善を Ver3 仕様確定前に検討する。</div>
        <div class="pri mid">中</div>
      </div>
""",
        """      <div class="todo-item">
        <div class="chk"></div>
        <div class="txt"><b>多デバイス対応に向けた設計整理</b>現在の 2 機種検証を土台に、端末追加時の評価テンプレートと SDK 提供を見据えた I/F 方針を定義する。</div>
        <div class="pri mid">中</div>
      </div>
""",
    )
    text = text.replace(
        "左記提案規模 (14～20 名) で C1～C6 の撮影状態のデータを追加取得し、",
        "左記提案規模 (14～20 名) で C1～C6 の環境・動作条件データを追加取得し、",
    )
    text = _replace_once(
        text,
        """      <div class="todo-item">
        <div class="chk"></div>
        <div class="txt"><b>対応デバイス範囲の明確化</b>現在の検証は 2 機種 (社内手机・社内 PC) のみ。公式サポート対象デバイスの基準と、新規デバイス追加時の検証フローを定義する。</div>
        <div class="pri low">低</div>
      </div>
""",
        """      <div class="todo-item">
        <div class="chk"></div>
        <div class="txt"><b>SDK 提供に向けた実装計画</b>SDK 化を想定し、入力仕様・品質判定 API・検証レポート出力の標準インターフェースを定義する。</div>
        <div class="pri low">低</div>
      </div>
""",
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(text, encoding="utf-8")
    print(f"[OK] wrote: {out_html}")


if __name__ == "__main__":
    main()
