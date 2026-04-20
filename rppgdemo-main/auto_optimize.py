#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from evaluate_dataset import (
    Sample,
    build_split_map,
    compare_to_ecg,
    discover_samples,
    load_ecg_series,
    parse_metrics_arg,
    process_video,
    summarize_all_metrics,
    write_csv,
)
from main import Config


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def cfg_to_dict(cfg: Config) -> Dict[str, float | bool | str]:
    return asdict(cfg)


def apply_overrides(base: Config, overrides: Dict[str, float | bool | str]) -> Config:
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


def component_loss(stats: Dict[str, float], mae_w: float, mape_w: float, corr_w: float) -> float:
    n = float(stats.get("n", 0.0))
    if n <= 0:
        return 1e6

    mae = float(stats.get("mae", math.nan))
    mape = float(stats.get("mape", math.nan))
    corr = float(stats.get("corr", math.nan))

    mae_term = mae if math.isfinite(mae) else 1e5
    mape_term = mape if math.isfinite(mape) else 1e5
    if math.isfinite(corr):
        corr_pen = max(0.0, 1.0 - corr)
    else:
        corr_pen = 2.0
    return mae_w * mae_term + mape_w * mape_term + corr_w * corr_pen


def objective_hr_lf_balanced(summary: Dict[str, Dict[str, float]]) -> Tuple[float, Dict[str, float]]:
    hr = summary["hr"]
    hf = summary["hf"]
    lfhf = summary["lfhf"]
    lfratio = summary["lfratio"]

    hr_loss = component_loss(hr, mae_w=0.35, mape_w=0.65, corr_w=8.0)

    lf_losses: List[float] = []
    for s in (hf, lfhf, lfratio):
        if float(s.get("n", 0.0)) > 0:
            lf_losses.append(component_loss(s, mae_w=0.10, mape_w=0.90, corr_w=6.0))
    lf_loss = float(np.mean(lf_losses)) if lf_losses else 1e6

    total = 0.5 * hr_loss + 0.5 * lf_loss
    parts = {
        "hr_loss": hr_loss,
        "lf_loss": lf_loss,
        "hf_loss": component_loss(hf, mae_w=0.10, mape_w=0.90, corr_w=6.0),
        "lfhf_loss": component_loss(lfhf, mae_w=0.10, mape_w=0.90, corr_w=6.0),
        "lfratio_loss": component_loss(lfratio, mae_w=0.10, mape_w=0.90, corr_w=6.0),
    }
    return total, parts


def objective_hr_only(summary: Dict[str, Dict[str, float]]) -> Tuple[float, Dict[str, float]]:
    hr = summary["hr"]
    hr_loss = component_loss(hr, mae_w=0.50, mape_w=0.50, corr_w=10.0)
    return hr_loss, {"hr_loss": hr_loss}


def evaluate_subset(
    samples: List[Sample],
    cfg: Config,
    roi_mode: str,
    mp_face_detector_model: str,
    strict_roi: bool,
    use_published: bool,
    metrics: Set[str],
    align_mode: str,
    include_cbcr: bool,
    lf_window_sec: float,
    lf_resample_fs: float,
    ecg_cache: Dict[str, Dict[int, object]],
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, float]]]:
    all_rows: List[Dict[str, object]] = []

    for s in samples:
        pred_rows = process_video(
            s,
            cfg,
            roi_mode=roi_mode,
            mp_face_detector_model=mp_face_detector_model,
            strict_roi=strict_roi,
            include_cbcr=include_cbcr,
            lf_window_sec=lf_window_sec,
            lf_resample_fs=lf_resample_fs,
        )
        cache_key = str(s.ecg_csv_path)
        if cache_key not in ecg_cache:
            ecg_cache[cache_key] = load_ecg_series(s.ecg_csv_path, align_mode=align_mode)
        ecg = ecg_cache[cache_key]
        compared = compare_to_ecg(pred_rows, ecg, use_published=use_published, metrics=metrics)
        for row in compared:
            row["split"] = row.get("split", "unknown")
        all_rows.extend(compared)

    summary = summarize_all_metrics(all_rows)
    return all_rows, summary


def objective_for_profile(profile: str, summary: Dict[str, Dict[str, float]]) -> Tuple[float, Dict[str, float]]:
    if profile == "hr_lf_balanced":
        return objective_hr_lf_balanced(summary)
    if profile == "hr_only":
        return objective_hr_only(summary)
    raise RuntimeError(f"unsupported objective profile: {profile}")


def stable_float(v: float) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if math.isfinite(v):
        return f"{v:.6f}"
    return str(v)


def signature_for_variant(overrides: Dict[str, float | bool | str], phase: str, iteration: int) -> str:
    items = []
    for k in sorted(overrides.keys()):
        val = overrides[k]
        if isinstance(val, bool):
            txt = "true" if val else "false"
        elif isinstance(val, (int, float)):
            txt = stable_float(float(val))
        else:
            txt = str(val)
        items.append(f"{k}={txt}")
    raw = f"phase={phase};iter={iteration};" + "|".join(items)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "low_hz": (0.55, 0.95),
    "high_hz": (3.6, 5.5),
    "welch_seg_sec": (3.0, 6.0),
    "welch_overlap": (0.65, 0.90),
    "pos_window_sec": (1.2, 2.6),
    "buffer_sec": (8.0, 15.0),
    "temporal_sigma_bpm": (10.0, 34.0),
    "high_hr_cluster_boost": (0.0, 0.45),
    "freq_conf_gate": (0.25, 0.70),
    "roi_scale_x": (0.9, 1.25),
    "roi_scale_y": (0.9, 1.25),
    "roi_shift_y": (-0.06, 0.06),
    "high_hr_bias_threshold": (85.0, 120.0),
    "high_hr_bias_gain": (0.90, 1.22),
    "high_hr_bias_offset": (-8.0, 16.0),
}

CODE_RANGES: Dict[str, Tuple[float, float]] = {
    "fusion_support_weight": (0.25, 0.75),
    "fusion_temporal_weight": (0.15, 0.65),
    "fusion_stability_weight": (0.05, 0.40),
    "quality_snr_weight": (0.10, 0.55),
    "quality_stability_weight": (0.10, 0.45),
    "quality_corr_weight": (0.10, 0.45),
    "quality_periodicity_weight": (0.10, 0.45),
    "quality_snr_norm_div": (10.0, 25.0),
    "quality_signal_strength_min": (0.001, 0.008),
    "ppi_gate_sqi_min": (0.40, 0.75),
    "ppi_gate_std_max": (8.0, 20.0),
    "ppi_gate_diff_min": (8.0, 28.0),
    "ppi_gate_ratio_min": (1.05, 1.40),
    "ppi_gate_ratio_max": (1.8, 2.6),
    "ppi_blend_tree_self": (0.0, 0.35),
    "ppi_blend_tree_ppi": (0.65, 1.0),
    "ppi_blend_weak_self": (0.05, 0.45),
    "ppi_blend_weak_ppi": (0.55, 0.95),
    "ppi_blend_strong_self": (0.10, 0.60),
    "ppi_blend_strong_ppi": (0.40, 0.90),
    "peak_threshold_k": (0.25, 0.85),
    "peak_window_divisor": (10.0, 24.0),
    "peak_missing_threshold_scale": (0.5, 0.9),
    "peak_min_distance_hz": (2.2, 4.0),
    "peak_max_distance_hz": (0.55, 0.95),
    "publish_quality_sqi_weight": (0.4, 0.9),
    "publish_quality_freq_weight": (0.1, 0.6),
    "publish_quality_low_freq_penalty": (0.55, 0.9),
    "publish_conf_sqi_weight": (0.25, 0.7),
    "publish_conf_freq_weight": (0.15, 0.6),
    "publish_conf_bias": (0.05, 0.35),
    "output_drop_guard_ref_min_bpm": (90.0, 120.0),
    "output_drop_guard_out_max_bpm": (70.0, 100.0),
    "output_drop_guard_ratio_max": (0.65, 0.90),
    "output_drop_guard_freq_conf_min": (0.45, 0.80),
    "output_drop_guard_sqi_min": (0.45, 0.75),
    "output_drop_guard_floor_ratio": (0.80, 0.96),
    "output_drop_guard_floor_delta": (8.0, 24.0),
}

BOOL_KEYS = [
    "use_cbcr_candidate",
    "enable_ppi_assist",
    "output_drop_guard_enabled",
]

CATEGORICAL_KEYS: Dict[str, List[str]] = {
    "roi_preset": ["classic4", "hybrid7", "cheek_forehead6", "whole_face"],
}



def _mutate_scalar(
    key: str,
    current: float,
    lo: float,
    hi: float,
    phase: str,
    rng: np.random.Generator,
) -> float:
    span = hi - lo
    if phase == "A":
        val = rng.uniform(lo, hi)
    elif phase == "B":
        if rng.random() < 0.35:
            val = rng.uniform(lo, hi)
        else:
            val = current + rng.normal(0.0, 0.12 * span)
    else:
        val = current + rng.normal(0.0, 0.06 * span)
    return clamp(float(val), lo, hi)


def mutate_overrides(
    parent: Dict[str, float | bool | str],
    base_cfg: Config,
    phase: str,
    rng: np.random.Generator,
    allow_code_mutation: bool,
) -> Tuple[Dict[str, float | bool | str], List[str]]:
    nxt = dict(parent)
    reasons: List[str] = []

    key_pool = list(PARAM_RANGES.keys())
    if allow_code_mutation and phase in {"B", "C"}:
        key_pool.extend(CODE_RANGES.keys())
        key_pool.extend(BOOL_KEYS)
        key_pool.extend(CATEGORICAL_KEYS.keys())

    if phase == "A":
        n_changes = int(rng.integers(3, 7))
    elif phase == "B":
        n_changes = int(rng.integers(4, 9))
    else:
        n_changes = int(rng.integers(2, 6))

    chosen = list(rng.choice(key_pool, size=min(n_changes, len(key_pool)), replace=False))
    for key in chosen:
        if key in BOOL_KEYS:
            nxt[key] = bool(rng.integers(0, 2))
            reasons.append(f"toggle {key} -> {nxt[key]}")
            continue
        if key in CATEGORICAL_KEYS:
            options = CATEGORICAL_KEYS[key]
            cur = str(nxt.get(key, getattr(base_cfg, key)))
            cand = [x for x in options if x != cur]
            if cand:
                nxt[key] = str(rng.choice(cand))
                reasons.append(f"switch {key} -> {nxt[key]}")
            continue

        if key in PARAM_RANGES:
            lo, hi = PARAM_RANGES[key]
        else:
            lo, hi = CODE_RANGES[key]
        current = float(nxt.get(key, getattr(base_cfg, key)))
        val = _mutate_scalar(key, current, lo, hi, phase=phase, rng=rng)
        nxt[key] = val
        reasons.append(f"mutate {key} -> {val:.4f}")

    # Keep paired blend parameters sane.
    for p_self, p_ppi in (
        ("ppi_blend_tree_self", "ppi_blend_tree_ppi"),
        ("ppi_blend_weak_self", "ppi_blend_weak_ppi"),
        ("ppi_blend_strong_self", "ppi_blend_strong_ppi"),
    ):
        if p_self in nxt or p_ppi in nxt:
            s = float(nxt.get(p_self, getattr(base_cfg, p_self)))
            p = float(nxt.get(p_ppi, getattr(base_cfg, p_ppi)))
            total = max(1e-6, s + p)
            nxt[p_self] = max(0.0, min(1.0, s / total))
            nxt[p_ppi] = max(0.0, min(1.0, p / total))

    if phase == "C" and not reasons:
        reasons.append("local refinement no-op")
    return nxt, reasons


def rows_columns() -> List[str]:
    return [
        "split",
        "group",
        "stem",
        "sec",
        "ecg_hr",
        "rppg_hr",
        "error",
        "abs_error",
        "ape_percent",
        "ecg_hf",
        "est_hf",
        "error_hf",
        "abs_error_hf",
        "ape_percent_hf",
        "ecg_lfhf",
        "est_lfhf",
        "error_lfhf",
        "abs_error_lfhf",
        "ape_percent_lfhf",
        "ecg_lfratio",
        "est_lfratio",
        "error_lfratio",
        "abs_error_lfratio",
        "ape_percent_lfratio",
        "ppi_hr",
        "pnn50",
        "pnn50_reliable",
        "sqi",
        "frequency_confidence",
        "snr_db",
        "state",
        "roi_mode",
    ]


def write_iteration_artifacts(
    run_dir: Path,
    iteration: int,
    phase: str,
    train_rows: List[Dict[str, object]],
    test_rows: List[Dict[str, object]],
    payload: Dict[str, object],
) -> None:
    tag = f"iter_{iteration:03d}"
    train_detail = run_dir / f"{tag}_train_detail.csv"
    test_detail = run_dir / f"{tag}_test_detail.csv"
    summary_path = run_dir / f"{tag}_summary.json"

    cols = rows_columns()
    write_csv(train_detail, train_rows, cols)
    write_csv(test_detail, test_rows, cols)

    dump = dict(payload)
    dump["phase"] = phase
    dump["train_detail_csv"] = str(train_detail)
    dump["test_detail_csv"] = str(test_detail)
    summary_path.write_text(json.dumps(dump, ensure_ascii=True, indent=2), encoding="utf-8")

    with (run_dir / "history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(dump, ensure_ascii=True) + "\n")


def write_markdown_report(run_dir: Path) -> None:
    history = run_dir / "history.jsonl"
    if not history.exists():
        return

    items = [json.loads(x) for x in history.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not items:
        return

    accepted = [x for x in items if bool(x.get("accepted", False))]
    best = min(accepted, key=lambda x: float(x["train_objective"])) if accepted else min(items, key=lambda x: float(x["train_objective"]))

    out: List[str] = []
    out.append("# Auto Optimization Report")
    out.append("")
    out.append(f"- run_dir: `{run_dir}`")
    out.append(f"- total_iterations: {len(items)}")
    out.append(f"- accepted_iterations: {len(accepted)}")
    out.append(f"- best_iteration: {best['iteration']}")
    out.append(f"- best_signature: {best['variant_signature']}")
    out.append(f"- best_train_objective: {best['train_objective']:.4f}")
    out.append(f"- best_test_objective: {best['test_objective']:.4f}")
    out.append("")
    out.append("## Iterations")
    out.append("")
    for it in items:
        status = "ACCEPTED" if it.get("accepted", False) else "REJECTED"
        out.append(
            f"- iter {it['iteration']:03d} [{it.get('phase', '?')}] {status}: "
            f"train_obj={it['train_objective']:.4f}, test_obj={it['test_objective']:.4f}, "
            f"sig={it['variant_signature']}"
        )
        for d in it.get("decision_reasons", []):
            out.append(f"  - {d}")
    out.append("")
    out.append("## Best Config")
    out.append("")
    out.append("```json")
    out.append(json.dumps(best["config"], ensure_ascii=True, indent=2))
    out.append("```")
    (run_dir / "REPORT.md").write_text("\n".join(out), encoding="utf-8")


def phase_for_iteration(iteration: int, budget: int) -> str:
    a_end = max(1, int(0.30 * budget))
    b_end = max(a_end + 1, int(0.75 * budget))
    if iteration <= a_end:
        return "A"
    if iteration <= b_end:
        return "B"
    return "C"


def pick_parent(
    leaderboard: List[Dict[str, object]],
    top_k: int,
    rng: np.random.Generator,
) -> Dict[str, float | bool | str]:
    if not leaderboard:
        return {}
    sorted_board = sorted(leaderboard, key=lambda x: float(x["train_objective"]))
    choices = sorted_board[: max(1, min(top_k, len(sorted_board)))]
    idx = int(rng.integers(0, len(choices)))
    return dict(choices[idx]["overrides"])


def main() -> None:
    parser = argparse.ArgumentParser(description="High-intensity auto optimizer for Demo2 Python rPPG")
    parser.add_argument("--data-dir", default="/Users/liangwenwang/Downloads/Code/Demo2")
    parser.add_argument("--out-dir", default="/Users/liangwenwang/Downloads/Code/Demo2/rppgdemo-main/results/auto_opt")
    parser.add_argument("--roi-mode", choices=["auto", "mediapipe", "opencv"], default="mediapipe")
    parser.add_argument("--roi-preset", choices=["classic4", "hybrid7", "cheek_forehead6", "whole_face"], default="hybrid7")
    parser.add_argument("--roi-scale-x", type=float, default=1.1)
    parser.add_argument("--roi-scale-y", type=float, default=1.1)
    parser.add_argument("--roi-shift-y", type=float, default=0.0)
    parser.add_argument("--mp-face-detector-model", default="")
    parser.add_argument("--strict-roi", action="store_true")
    parser.add_argument("--use-published", action="store_true")
    parser.add_argument("--groups", default="001,002,003", help="comma-separated, e.g. 003 or 001,003")
    parser.add_argument("--max-samples-per-group", type=int, default=0, help="0 means all")
    parser.add_argument("--split-file", default="", help="optional train/test split file")
    parser.add_argument("--holdout-list", default="", help="comma-separated test samples: 001/3-3,002/3-4,...")
    parser.add_argument("--align-mode", choices=["timestamp", "index"], default="timestamp")
    parser.add_argument("--metrics", default="hr,lf")
    parser.add_argument("--lf-window-sec", type=float, default=30.0)
    parser.add_argument("--lf-resample-fs", type=float, default=4.0)
    parser.add_argument("--disable-cbcr", action="store_true")
    parser.add_argument("--allow-code-mutation", action="store_true")
    parser.add_argument("--search-budget", type=int, default=60)
    parser.add_argument("--objective-profile", choices=["hr_lf_balanced", "hr_only"], default="hr_lf_balanced")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260317)
    parser.add_argument("--max-test-regression-ratio", type=float, default=0.03)
    parser.add_argument("--iterations", type=int, default=0, help="deprecated; if >0 overrides --search-budget")
    args = parser.parse_args()

    budget = int(args.iterations if args.iterations > 0 else args.search_budget)
    if budget <= 0:
        raise RuntimeError("search budget must be > 0")

    metrics = parse_metrics_arg(args.metrics)
    groups = [x.strip() for x in args.groups.split(",") if x.strip()]

    samples_all = discover_samples(Path(args.data_dir))
    samples = pick_samples(samples_all, groups, args.max_samples_per_group)
    if not samples:
        raise RuntimeError("no valid samples selected")

    split_map = build_split_map(samples, split_file=args.split_file, holdout_list=args.holdout_list)
    train_samples = [s for s in samples if split_map.get(f"{s.group}/{s.stem}", "train") == "train"]
    test_samples = [s for s in samples if split_map.get(f"{s.group}/{s.stem}", "train") == "test"]

    if not train_samples:
        raise RuntimeError("train split is empty")
    if not test_samples:
        print("[AUTO] warning: test split is empty, using train split for test monitoring")
        test_samples = list(train_samples)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = Config()
    base_cfg.roi_preset = args.roi_preset
    base_cfg.roi_scale_x = args.roi_scale_x
    base_cfg.roi_scale_y = args.roi_scale_y
    base_cfg.roi_shift_y = args.roi_shift_y

    rng = np.random.default_rng(args.seed)
    ecg_cache: Dict[str, Dict[int, object]] = {}

    print(f"[AUTO] run_dir={run_dir}")
    print(
        f"[AUTO] train={len(train_samples)} test={len(test_samples)} roi_mode={args.roi_mode} "
        f"budget={budget} objective={args.objective_profile}"
    )

    # Baseline (iteration 0)
    base_overrides: Dict[str, float | bool | str] = {}
    base_cfg_effective = apply_overrides(base_cfg, base_overrides)

    train_rows0, train_summary0 = evaluate_subset(
        train_samples,
        base_cfg_effective,
        roi_mode=args.roi_mode,
        mp_face_detector_model=args.mp_face_detector_model,
        strict_roi=args.strict_roi,
        use_published=args.use_published,
        metrics=metrics,
        align_mode=args.align_mode,
        include_cbcr=not args.disable_cbcr,
        lf_window_sec=args.lf_window_sec,
        lf_resample_fs=args.lf_resample_fs,
        ecg_cache=ecg_cache,
    )
    test_rows0, test_summary0 = evaluate_subset(
        test_samples,
        base_cfg_effective,
        roi_mode=args.roi_mode,
        mp_face_detector_model=args.mp_face_detector_model,
        strict_roi=args.strict_roi,
        use_published=args.use_published,
        metrics=metrics,
        align_mode=args.align_mode,
        include_cbcr=not args.disable_cbcr,
        lf_window_sec=args.lf_window_sec,
        lf_resample_fs=args.lf_resample_fs,
        ecg_cache=ecg_cache,
    )

    base_train_obj, base_train_parts = objective_for_profile(args.objective_profile, train_summary0)
    base_test_obj, base_test_parts = objective_for_profile(args.objective_profile, test_summary0)

    baseline_payload = {
        "iteration": 0,
        "phase": "BASE",
        "variant_signature": signature_for_variant(base_overrides, "BASE", 0),
        "overrides": base_overrides,
        "config": cfg_to_dict(base_cfg_effective),
        "train_summary": train_summary0,
        "test_summary": test_summary0,
        "train_objective": base_train_obj,
        "test_objective": base_test_obj,
        "train_objective_parts": base_train_parts,
        "test_objective_parts": base_test_parts,
        "accepted": True,
        "rollback": False,
        "decision_reasons": ["baseline"],
    }
    write_iteration_artifacts(run_dir, 0, "BASE", train_rows0, test_rows0, baseline_payload)

    best_overrides = dict(base_overrides)
    best_train_obj = float(base_train_obj)
    best_test_obj = float(base_test_obj)
    leaderboard: List[Dict[str, object]] = [
        {
            "iteration": 0,
            "train_objective": best_train_obj,
            "test_objective": best_test_obj,
            "overrides": dict(base_overrides),
            "variant_signature": baseline_payload["variant_signature"],
        }
    ]

    for i in range(1, budget + 1):
        phase = phase_for_iteration(i, budget)
        parent = pick_parent(leaderboard, top_k=args.top_k, rng=rng)
        candidate_overrides, reasons = mutate_overrides(
            parent,
            base_cfg,
            phase=phase,
            rng=rng,
            allow_code_mutation=args.allow_code_mutation,
        )
        cfg_i = apply_overrides(base_cfg, candidate_overrides)
        signature = signature_for_variant(candidate_overrides, phase=phase, iteration=i)

        print(
            f"[AUTO] iter={i:03d}/{budget} phase={phase} sig={signature} "
            f"mut={len(reasons)}"
        )

        train_rows, train_summary = evaluate_subset(
            train_samples,
            cfg_i,
            roi_mode=args.roi_mode,
            mp_face_detector_model=args.mp_face_detector_model,
            strict_roi=args.strict_roi,
            use_published=args.use_published,
            metrics=metrics,
            align_mode=args.align_mode,
            include_cbcr=not args.disable_cbcr,
            lf_window_sec=args.lf_window_sec,
            lf_resample_fs=args.lf_resample_fs,
            ecg_cache=ecg_cache,
        )
        test_rows, test_summary = evaluate_subset(
            test_samples,
            cfg_i,
            roi_mode=args.roi_mode,
            mp_face_detector_model=args.mp_face_detector_model,
            strict_roi=args.strict_roi,
            use_published=args.use_published,
            metrics=metrics,
            align_mode=args.align_mode,
            include_cbcr=not args.disable_cbcr,
            lf_window_sec=args.lf_window_sec,
            lf_resample_fs=args.lf_resample_fs,
            ecg_cache=ecg_cache,
        )

        train_obj, train_parts = objective_for_profile(args.objective_profile, train_summary)
        test_obj, test_parts = objective_for_profile(args.objective_profile, test_summary)

        accepted = False
        rollback = False
        decision: List[str] = []

        train_hr_n = float(train_summary["hr"].get("n", 0.0))
        test_hr_n = float(test_summary["hr"].get("n", 0.0))
        has_nan = not (math.isfinite(train_obj) and math.isfinite(test_obj))
        if has_nan:
            rollback = True
            decision.append("invalid objective (NaN/inf) -> rollback")
        elif train_hr_n < 10 or test_hr_n < 10:
            rollback = True
            decision.append("insufficient HR aligned samples -> rollback")
        else:
            test_gate = test_obj <= best_test_obj * (1.0 + args.max_test_regression_ratio)
            if train_obj < best_train_obj and test_gate:
                accepted = True
                decision.append("accepted: train improved and test gate passed")
            else:
                rollback = True
                if train_obj >= best_train_obj:
                    decision.append("rejected: train objective not improved")
                if not test_gate:
                    decision.append("rejected: test objective regression beyond gate")

        if accepted:
            best_overrides = dict(candidate_overrides)
            best_train_obj = float(train_obj)
            best_test_obj = float(test_obj)
            leaderboard.append(
                {
                    "iteration": i,
                    "train_objective": best_train_obj,
                    "test_objective": best_test_obj,
                    "overrides": dict(candidate_overrides),
                    "variant_signature": signature,
                }
            )
        else:
            # Track rejected variants for audit, but keep active config unchanged.
            leaderboard.append(
                {
                    "iteration": i,
                    "train_objective": float(train_obj),
                    "test_objective": float(test_obj),
                    "overrides": dict(candidate_overrides),
                    "variant_signature": signature,
                }
            )

        payload = {
            "iteration": i,
            "phase": phase,
            "variant_signature": signature,
            "overrides": candidate_overrides,
            "config": cfg_to_dict(cfg_i),
            "train_summary": train_summary,
            "test_summary": test_summary,
            "train_objective": float(train_obj),
            "test_objective": float(test_obj),
            "train_objective_parts": train_parts,
            "test_objective_parts": test_parts,
            "accepted": accepted,
            "rollback": rollback,
            "decision_reasons": reasons + decision,
            "best_train_objective_after_iter": best_train_obj,
            "best_test_objective_after_iter": best_test_obj,
        }
        write_iteration_artifacts(run_dir, i, phase, train_rows, test_rows, payload)

        print(
            f"[AUTO] iter={i:03d} train_obj={train_obj:.4f} test_obj={test_obj:.4f} "
            f"accepted={accepted} rollback={rollback}"
        )

    best_cfg = apply_overrides(base_cfg, best_overrides)
    (run_dir / "best_config.json").write_text(
        json.dumps(cfg_to_dict(best_cfg), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (run_dir / "best_overrides.json").write_text(
        json.dumps(best_overrides, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    decisions: List[Dict[str, object]] = []
    history_path = run_dir / "history.jsonl"
    if history_path.exists():
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            decisions.append(
                {
                    "iteration": item.get("iteration"),
                    "phase": item.get("phase"),
                    "variant_signature": item.get("variant_signature"),
                    "accepted": item.get("accepted"),
                    "rollback": item.get("rollback"),
                    "decision_reasons": item.get("decision_reasons", []),
                }
            )
    (run_dir / "decisions.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=True) for x in decisions),
        encoding="utf-8",
    )

    write_markdown_report(run_dir)
    print(f"[AUTO] done. best_train_objective={best_train_obj:.4f} best_test_objective={best_test_obj:.4f}")
    print(f"[AUTO] best_config={run_dir / 'best_config.json'}")
    print(f"[AUTO] report={run_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
