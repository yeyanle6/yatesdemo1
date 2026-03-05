#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from evaluate_dataset import (
    Sample,
    compare_to_ecg,
    discover_samples,
    load_ecg_hr,
    process_video,
    summarize,
    write_csv,
)
from main import Config


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def cfg_to_dict(cfg: Config) -> Dict[str, float | bool]:
    return asdict(cfg)


def apply_overrides(base: Config, overrides: Dict[str, float | bool]) -> Config:
    cfg = Config(**cfg_to_dict(base))
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            continue
        setattr(cfg, k, v)
    return cfg


def pick_samples(samples: List[Sample], groups: List[str], max_per_group: int) -> List[Sample]:
    if groups:
        allowed = {g.lstrip("0") for g in groups}
        samples = [s for s in samples if s.group.lstrip("0") in allowed]
    if max_per_group <= 0:
        return samples

    out: List[Sample] = []
    count: Dict[str, int] = {}
    for s in samples:
        g = s.group.lstrip("0")
        c = count.get(g, 0)
        if c >= max_per_group:
            continue
        out.append(s)
        count[g] = c + 1
    return out


def group_metrics(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not rows:
        return out
    bucket: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        g = str(r["group"]).lstrip("0")
        bucket.setdefault(g, []).append(r)

    for g, rws in bucket.items():
        s = summarize(rws)
        err = np.array([float(r["error"]) for r in rws], dtype=np.float64)
        ecg = np.array([float(r["ecg_hr"]) for r in rws], dtype=np.float64)
        est = np.array([float(r["rppg_hr"]) for r in rws], dtype=np.float64)
        ratio = est / np.maximum(ecg, 1e-6)
        out[g] = {
            "n": float(s["n"]),
            "mae": float(s["mae"]),
            "rmse": float(s["rmse"]),
            "mape": float(s["mape"]),
            "corr": float(s["corr"]),
            "bias": float(np.mean(err)),
            "ratio_mean": float(np.mean(ratio)),
        }
    return out


def high_hr_metrics(rows: List[Dict[str, object]], threshold: float = 100.0) -> Dict[str, float]:
    sub = [r for r in rows if float(r["ecg_hr"]) >= threshold]
    if not sub:
        return {"n": 0.0, "mae": math.nan, "bias": math.nan}
    err = np.array([float(r["error"]) for r in sub], dtype=np.float64)
    ae = np.abs(err)
    return {"n": float(len(sub)), "mae": float(np.mean(ae)), "bias": float(np.mean(err))}


def objective(overall: Dict[str, float], g: Dict[str, Dict[str, float]], high_hr: Dict[str, float]) -> float:
    g3 = g.get("3", {})
    g3_mae = float(g3.get("mae", 0.0))
    g3_bias = abs(float(g3.get("bias", 0.0)))
    high_mae = float(high_hr.get("mae", 0.0))
    base_mae = float(overall["mae"])
    return base_mae + 0.55 * g3_mae + 0.15 * g3_bias + 0.20 * high_mae


def propose_next(
    current_overrides: Dict[str, float | bool],
    analysis: Dict[str, object],
    iteration: int,
) -> Tuple[Dict[str, float | bool], List[str]]:
    nxt = dict(current_overrides)
    reason: List[str] = []

    g = analysis.get("groups", {})
    g3 = g.get("3", {})
    g3_mae = float(g3.get("mae", 0.0))
    g3_bias = float(g3.get("bias", 0.0))
    g3_ratio = float(g3.get("ratio_mean", 1.0))

    high = analysis.get("high_hr", {})
    high_bias = float(high.get("bias", 0.0))
    high_mae = float(high.get("mae", 0.0))

    # Rule 1: high-HR underestimation -> widen response and stabilize PSD estimate.
    if g3_mae > 18.0 and (g3_bias < -15.0 or g3_ratio < 0.85):
        nxt["high_hz"] = clamp(float(nxt.get("high_hz", 4.0)) + 0.25, 3.5, 5.5)
        nxt["welch_seg_sec"] = clamp(float(nxt.get("welch_seg_sec", 3.0)) + 0.5, 2.5, 6.0)
        nxt["buffer_sec"] = clamp(float(nxt.get("buffer_sec", 8.0)) + 1.0, 6.0, 14.0)
        nxt["temporal_sigma_bpm"] = clamp(float(nxt.get("temporal_sigma_bpm", 18.0)) + 2.0, 10.0, 32.0)
        nxt["high_hr_cluster_boost"] = clamp(float(nxt.get("high_hr_cluster_boost", 0.10)) + 0.03, 0.0, 0.40)
        reason.append("group3/high-HR appears under-estimated; increased high_hz, welch_seg_sec, buffer_sec")

    # Rule 2: severe high-HR bias -> enable PPI assist.
    if high_mae > 18.0 and high_bias < -15.0:
        nxt["enable_ppi_assist"] = True
        reason.append("high-HR bias is strongly negative; enabled PPI assist")

    # Rule 3: if low-HR overestimation appears, reduce high-pass floor slightly.
    low = analysis.get("low_hr", {})
    low_bias = float(low.get("bias", 0.0))
    if low_bias > 6.0:
        nxt["low_hz"] = clamp(float(nxt.get("low_hz", 0.65)) - 0.05, 0.45, 0.9)
        reason.append("low-HR bias is positive; decreased low_hz slightly")

    # Rule 4: ROI exploration (anatomical region strategy).
    roi_preset = str(nxt.get("roi_preset", "classic4"))
    if iteration == 1 and roi_preset == "classic4":
        nxt["roi_preset"] = "hybrid7"
        reason.append("ROI exploration: switched preset classic4 -> hybrid7")
    elif iteration == 2 and roi_preset == "hybrid7":
        nxt["roi_preset"] = "cheek_forehead6"
        reason.append("ROI exploration: switched preset hybrid7 -> cheek_forehead6")
    elif iteration == 3 and roi_preset == "cheek_forehead6":
        nxt["roi_preset"] = "classic4"
        reason.append("ROI exploration: reverted to classic4")

    # Rule 5: periodic exploration to avoid local minima.
    if not reason:
        if iteration % 2 == 0:
            nxt["welch_overlap"] = clamp(float(nxt.get("welch_overlap", 0.75)) + 0.05, 0.60, 0.90)
            reason.append("exploration step: increased welch_overlap")
        else:
            nxt["pos_window_sec"] = clamp(float(nxt.get("pos_window_sec", 1.6)) + 0.2, 1.2, 2.4)
            reason.append("exploration step: increased pos_window_sec")

    return nxt, reason


def bin_metrics(rows: List[Dict[str, object]], lo: float, hi: float) -> Dict[str, float]:
    sub = [r for r in rows if lo <= float(r["ecg_hr"]) < hi]
    if not sub:
        return {"n": 0.0, "mae": math.nan, "bias": math.nan}
    err = np.array([float(r["error"]) for r in sub], dtype=np.float64)
    return {"n": float(len(sub)), "mae": float(np.mean(np.abs(err))), "bias": float(np.mean(err))}


def evaluate_once(
    samples: List[Sample],
    cfg: Config,
    roi_mode: str,
    mp_face_detector_model: str,
    strict_roi: bool,
    use_published: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    all_rows: List[Dict[str, object]] = []
    for s in samples:
        pred_rows = process_video(
            s,
            cfg,
            roi_mode=roi_mode,
            mp_face_detector_model=mp_face_detector_model,
            strict_roi=strict_roi,
        )
        ecg = load_ecg_hr(s.ecg_csv_path)
        all_rows.extend(compare_to_ecg(pred_rows, ecg, use_published=use_published))

    overall = summarize(all_rows)
    g = group_metrics(all_rows)
    high = high_hr_metrics(all_rows, threshold=100.0)
    low = bin_metrics(all_rows, lo=0.0, hi=70.0)
    obj = objective(overall, g, high)
    analysis: Dict[str, object] = {
        "overall": overall,
        "groups": g,
        "high_hr": high,
        "low_hr": low,
        "objective": obj,
    }
    return all_rows, analysis


def write_iteration_artifacts(
    run_dir: Path,
    i: int,
    rows: List[Dict[str, object]],
    analysis: Dict[str, object],
    cfg: Config,
    decision: List[str],
) -> None:
    tag = f"iter_{i:02d}"
    detail_path = run_dir / f"{tag}_detail.csv"
    summary_path = run_dir / f"{tag}_summary.json"

    columns = [
        "group",
        "stem",
        "sec",
        "ecg_hr",
        "rppg_hr",
        "error",
        "abs_error",
        "ape_percent",
        "ppi_hr",
        "pnn50",
        "pnn50_reliable",
        "sqi",
        "frequency_confidence",
        "snr_db",
        "state",
        "roi_mode",
    ]
    write_csv(detail_path, rows, columns)

    payload = {
        "iteration": i,
        "config": cfg_to_dict(cfg),
        "decision": decision,
        "analysis": analysis,
        "detail_csv": str(detail_path),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    history_line = run_dir / "history.jsonl"
    with history_line.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def write_markdown_report(run_dir: Path) -> None:
    history = run_dir / "history.jsonl"
    if not history.exists():
        return
    lines = history.read_text(encoding="utf-8").splitlines()
    items = [json.loads(x) for x in lines if x.strip()]
    if not items:
        return

    best = min(items, key=lambda x: float(x["analysis"]["objective"]))
    out: List[str] = []
    out.append("# Auto Optimization Report")
    out.append("")
    out.append(f"- run_dir: `{run_dir}`")
    out.append(f"- total_iterations: {len(items)}")
    out.append(f"- best_iteration: {best['iteration']}")
    out.append(f"- best_objective: {best['analysis']['objective']:.4f}")
    out.append("")
    out.append("## Iterations")
    out.append("")
    for it in items:
        ov = it["analysis"]["overall"]
        g3 = it["analysis"]["groups"].get("3", {})
        out.append(
            f"- iter {it['iteration']:02d}: obj={it['analysis']['objective']:.4f}, "
            f"MAE={ov['mae']:.3f}, RMSE={ov['rmse']:.3f}, "
            f"G3_MAE={g3.get('mae', float('nan')):.3f}, "
            f"G3_bias={g3.get('bias', float('nan')):.3f}"
        )
        for d in it.get("decision", []):
            out.append(f"  - decision: {d}")
    out.append("")
    out.append("## Best Config")
    out.append("")
    out.append("```json")
    out.append(json.dumps(best["config"], ensure_ascii=True, indent=2))
    out.append("```")
    (run_dir / "REPORT.md").write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop auto optimizer for rPPG parameters")
    parser.add_argument("--data-dir", default="/Users/liangwenwang/Downloads/Code/demo1/data")
    parser.add_argument("--out-dir", default="/Users/liangwenwang/Downloads/Code/demo1/rpi_rppg/results/auto_opt")
    parser.add_argument("--roi-mode", choices=["auto", "mediapipe", "opencv"], default="opencv")
    parser.add_argument("--roi-preset", choices=["classic4", "hybrid7", "cheek_forehead6", "whole_face"], default="hybrid7")
    parser.add_argument("--roi-scale-x", type=float, default=1.1)
    parser.add_argument("--roi-scale-y", type=float, default=1.1)
    parser.add_argument("--roi-shift-y", type=float, default=0.0)
    parser.add_argument("--mp-face-detector-model", default="")
    parser.add_argument("--strict-roi", action="store_true")
    parser.add_argument("--use-published", action="store_true")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--groups", default="001,002,003", help="comma-separated, e.g. 003 or 001,003")
    parser.add_argument("--max-samples-per-group", type=int, default=0, help="0 means all")
    args = parser.parse_args()

    groups = [x.strip() for x in args.groups.split(",") if x.strip()]
    samples_all = discover_samples(Path(args.data_dir))
    samples = pick_samples(samples_all, groups, args.max_samples_per_group)
    if not samples:
        raise RuntimeError("no valid samples selected")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = Config()
    base_cfg.roi_preset = args.roi_preset
    base_cfg.roi_scale_x = args.roi_scale_x
    base_cfg.roi_scale_y = args.roi_scale_y
    base_cfg.roi_shift_y = args.roi_shift_y
    overrides: Dict[str, float | bool] = {}
    best_obj = float("inf")
    best_overrides: Dict[str, float | bool] = {}

    print(f"[AUTO] run_dir={run_dir}")
    print(f"[AUTO] samples={len(samples)} roi_mode={args.roi_mode} iterations={args.iterations}")

    for i in range(1, args.iterations + 1):
        cfg = apply_overrides(base_cfg, overrides)
        print(f"[AUTO] iter={i:02d} cfg_overrides={json.dumps(overrides, ensure_ascii=True)}")

        rows, analysis = evaluate_once(
            samples=samples,
            cfg=cfg,
            roi_mode=args.roi_mode,
            mp_face_detector_model=args.mp_face_detector_model,
            strict_roi=args.strict_roi,
            use_published=args.use_published,
        )
        obj = float(analysis["objective"])
        ov = analysis["overall"]
        g3 = analysis["groups"].get("3", {})
        print(
            f"[AUTO] iter={i:02d} obj={obj:.4f} MAE={ov['mae']:.3f} RMSE={ov['rmse']:.3f} "
            f"G3_MAE={g3.get('mae', math.nan):.3f} G3_bias={g3.get('bias', math.nan):.3f}"
        )

        decision: List[str] = []
        if obj < best_obj:
            best_obj = obj
            best_overrides = dict(overrides)
            decision.append("kept as current best")
        else:
            decision.append("not better than best")

        write_iteration_artifacts(run_dir, i, rows, analysis, cfg, decision)
        nxt, reasons = propose_next(overrides, analysis, i)
        overrides = nxt

        if reasons:
            print(f"[AUTO] iter={i:02d} next_decision={' | '.join(reasons)}")
            # append proposal to the just-written iteration record for traceability
            line = {
                "iteration": i,
                "next_overrides": overrides,
                "decision_reasons": reasons,
            }
            with (run_dir / "decisions.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=True) + "\n")

    best_cfg = apply_overrides(base_cfg, best_overrides)
    (run_dir / "best_config.json").write_text(
        json.dumps(cfg_to_dict(best_cfg), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(run_dir)
    print(f"[AUTO] done. best_objective={best_obj:.4f}")
    print(f"[AUTO] best_config={run_dir / 'best_config.json'}")
    print(f"[AUTO] report={run_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
