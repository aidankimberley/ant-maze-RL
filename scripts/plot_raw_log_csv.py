#!/usr/bin/env python3
"""
Plot CSV logs produced by this repo.

Supports two formats:

1) Online training raw logs (from test_online_transfer.py)
   Columns include: type, env_step, episode_return, eval_success, ...

   Usage:
     python scripts/plot_raw_log_csv.py --csv experiments/.../raw_log.csv

   Outputs (saved next to the CSV by default):
     - eval_success.png
     - episode_return_vs_env_steps.png
     - episode_return_vs_episode_idx.png

2) Evaluation summaries (from scripts/eval_hiql_nominal_vs_shifted.py)
   Columns: env_name, checkpoint_pkl, seed, variant, shift_family, shift_level, task_id, success

   Usage (compare multiple checkpoints):
     python scripts/plot_raw_log_csv.py \
       --csv experiments/eval_nominal_vs_shifted_step300000_seed0.csv \
             experiments/eval_nominal_vs_shifted_step400000_seed0.csv

   Outputs:
     - eval_success_comparison.png
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


def _to_int(x: str) -> int | None:
    x = (x or "").strip()
    if x == "":
        return None
    return int(float(x))


def _to_float(x: str) -> float | None:
    x = (x or "").strip()
    if x == "":
        return None
    return float(x)


def _read_header(csv_path: Path) -> list[str]:
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    return [h.strip() for h in header]


def _is_raw_log_csv(header: Iterable[str]) -> bool:
    return "type" in set(header) and "env_step" in set(header)


def _is_eval_summary_csv(header: Iterable[str]) -> bool:
    required = {"variant", "shift_level", "task_id", "success", "checkpoint_pkl"}
    return required.issubset(set(header))


def read_raw_log(csv_path: Path) -> dict[str, Any]:
    eval_env_steps: list[int] = []
    eval_success: list[float] = []

    ep_env_steps: list[int] = []
    ep_idx: list[int] = []
    ep_return: list[float] = []
    ep_len: list[int] = []

    meta: dict[str, str] = {}

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_type = (row.get("type") or "").strip()

            # Capture metadata from the first row that has it.
            if not meta:
                for k in ("method", "env_name", "task_id", "offline_step"):
                    v = (row.get(k) or "").strip()
                    if v != "":
                        meta[k] = v

            if row_type == "eval":
                s = _to_float(row.get("eval_success", ""))
                t = _to_int(row.get("env_step", ""))
                if s is not None and t is not None:
                    eval_env_steps.append(t)
                    eval_success.append(s)
            elif row_type == "episode":
                t = _to_int(row.get("env_step", ""))
                i = _to_int(row.get("episode_idx", ""))
                r = _to_float(row.get("episode_return", ""))
                L = _to_int(row.get("episode_len", ""))
                if t is not None and i is not None and r is not None:
                    ep_env_steps.append(t)
                    ep_idx.append(i)
                    ep_return.append(r)
                    ep_len.append(L or 0)

    return dict(
        meta=meta,
        eval_env_steps=eval_env_steps,
        eval_success=eval_success,
        ep_env_steps=ep_env_steps,
        ep_idx=ep_idx,
        ep_return=ep_return,
        ep_len=ep_len,
    )


def read_eval_summary(csv_path: Path) -> dict[str, Any]:
    """
    Read scripts/eval_hiql_nominal_vs_shifted.py output CSV.
    Returns a nested mapping:
      results[(variant, shift_level)][task_id] = success
    and metadata.
    """
    results: dict[tuple[str, str], dict[int, float]] = {}
    meta: dict[str, str] = {}

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not meta:
                for k in ("env_name", "checkpoint_pkl", "seed", "shift_family"):
                    v = (row.get(k) or "").strip()
                    if v != "":
                        meta[k] = v

            variant = (row.get("variant") or "").strip()
            shift_level = (row.get("shift_level") or "").strip()
            task_id = _to_int(row.get("task_id", ""))
            success = _to_float(row.get("success", ""))
            if variant == "" or shift_level == "" or task_id is None or success is None:
                continue

            key = (variant, shift_level)
            results.setdefault(key, {})[int(task_id)] = float(success)

    return {"meta": meta, "results": results}


def _label_for_checkpoint(path_str: str) -> str:
    p = Path(path_str)
    name = p.name
    # Prefer params_XXXXX.pkl => XXXXX, else fallback to filename stem.
    m = re.match(r"params_(\d+)\.pkl$", name)
    if m:
        return m.group(1)
    return p.stem


def plot_eval_comparison(csv_paths: list[Path], out_dir: Path) -> None:
    series = []
    for p in csv_paths:
        d = read_eval_summary(p)
        meta = d["meta"]
        ckpt = meta.get("checkpoint_pkl", str(p))
        series.append((p, ckpt, d["results"], meta))

    # Determine which conditions exist (in a stable order).
    # We map (variant, shift_level) -> display name.
    preferred_order = [
        ("nominal", "base"),
        ("shifted", "moderate_low"),
        ("shifted", "severe_high"),
        ("shifted", "moderate_high"),
        ("shifted", "severe_low"),
    ]
    present_keys = set()
    for _, _, results, _ in series:
        present_keys |= set(results.keys())
    cond_keys = [k for k in preferred_order if k in present_keys] + sorted(present_keys - set(preferred_order))

    # Bar plot: mean success per condition, with per-task dots.
    plt.figure(figsize=(9, 4.5))
    width = 0.8 / max(1, len(series))
    x = np.arange(len(cond_keys))

    for j, (_, ckpt_path, results, meta) in enumerate(series):
        means = []
        for key in cond_keys:
            per_task = results.get(key, {})
            vals = [per_task.get(t, float("nan")) for t in range(1, 6)]
            vals = [v for v in vals if not np.isnan(v)]
            means.append(float(np.mean(vals)) if len(vals) else float("nan"))

        offset = (j - (len(series) - 1) / 2) * width
        plt.bar(x + offset, means, width=width, alpha=0.75, label=f"ckpt { _label_for_checkpoint(ckpt_path) }")

        # Scatter per-task values for each condition
        for i, key in enumerate(cond_keys):
            per_task = results.get(key, {})
            ys = [per_task.get(t, None) for t in range(1, 6)]
            ys = [y for y in ys if y is not None]
            xs = [x[i] + offset] * len(ys)
            plt.scatter(xs, ys, s=18, color="black", alpha=0.6)

    cond_labels = []
    for variant, shift_level in cond_keys:
        if variant == "nominal":
            cond_labels.append("nominal")
        else:
            cond_labels.append(shift_level)

    env_name = series[0][3].get("env_name", "")
    shift_family = series[0][3].get("shift_family", "")
    seed = series[0][3].get("seed", "")
    title_bits = [b for b in [env_name, f"shift={shift_family}" if shift_family else "", f"seed={seed}" if seed else ""] if b]

    plt.xticks(x, cond_labels, rotation=0)
    plt.ylabel("Success (mean over tasks 1–5)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, axis="y", alpha=0.25)
    plt.title("  ".join(title_bits))
    plt.legend(frameon=False)
    plt.tight_layout()

    out_path = out_dir / "eval_success_comparison.png"
    plt.savefig(out_path, dpi=170)
    print(f"wrote {out_path}")
    plt.close()

def _paper_style():
    # Simple Matplotlib styling for paper-ready plots without extra deps.
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _condition_display(variant: str, shift_level: str) -> str:
    if variant == "nominal":
        return "Nominal (regular physics)"
    # Use the user’s naming convention.
    if shift_level == "moderate_low":
        return "Moderate physics shift"
    if shift_level == "severe_high":
        return "Severe physics shift"
    return f"{variant}:{shift_level}"


def _ckpt_display_from_meta(checkpoint_pkl: str) -> str:
    step = _label_for_checkpoint(checkpoint_pkl)
    if step == "300000":
        return "Frozen baseline"
    if step == "400000":
        return "Online finetuned"
    return f"Checkpoint {step}"


def _eval_seed_task_means(results: dict[tuple[str, str], dict[int, float]]) -> dict[tuple[str, str], float]:
    """
    For each (variant, shift_level), average over tasks 1..5.
    """
    out: dict[tuple[str, str], float] = {}
    for key, per_task in results.items():
        vals = [per_task.get(t, float("nan")) for t in range(1, 6)]
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals):
            out[key] = float(np.mean(vals))
    return out


def plot_eval_across_seeds(csv_paths: list[Path], out_dir: Path) -> None:
    """
    Make paper-ready plots from multiple eval summary CSVs:
      1) Moderate-only comparison: Frozen baseline vs Online finetuned (mean±std across seeds).
      2) Full comparison across conditions (Nominal/Moderate/Severe) with per-seed points + mean±std.
    """
    _paper_style()

    # Read all CSVs into: model_label -> seed -> condition -> mean_success_over_tasks
    models: dict[str, dict[int, dict[tuple[str, str], float]]] = {}
    meta_any: dict[str, str] = {}

    for p in csv_paths:
        d = read_eval_summary(p)
        meta = d["meta"]
        results = d["results"]
        if not meta_any:
            meta_any = meta

        model_label = _ckpt_display_from_meta(meta.get("checkpoint_pkl", str(p)))
        seed = int(float(meta.get("seed", "0")))
        cond_means = _eval_seed_task_means(results)
        models.setdefault(model_label, {})[seed] = cond_means

    # Conditions to show (in required order)
    cond_keys = [
        ("nominal", "base"),
        ("shifted", "moderate_low"),
        ("shifted", "severe_high"),
    ]

    # --- Plot 1: moderate-only ---
    moderate_key = ("shifted", "moderate_low")
    plot_models = ["Frozen baseline", "Online finetuned"]
    colors = {
        "Frozen baseline": "#4C78A8",
        "Online finetuned": "#F58518",
    }

    means = []
    errs = []
    for m in plot_models:
        seed_map = models.get(m, {})
        vals = []
        for _, conds in sorted(seed_map.items()):
            if moderate_key in conds:
                vals.append(conds[moderate_key])
        means.append(float(np.mean(vals)) if len(vals) else float("nan"))
        errs.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

    plt.figure(figsize=(5.5, 3.6))
    x = np.arange(len(plot_models))
    bars = plt.bar(
        x,
        means,
        yerr=errs,
        capsize=5,
        color=[colors.get(m, "#999999") for m in plot_models],
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
    )

    # Per-seed scatter (jittered)
    for i, m in enumerate(plot_models):
        seed_map = models.get(m, {})
        ys = [conds[moderate_key] for _, conds in sorted(seed_map.items()) if moderate_key in conds]
        xs = i + (np.random.RandomState(0).rand(len(ys)) - 0.5) * 0.12
        plt.scatter(xs, ys, s=28, color="black", alpha=0.75, zorder=3)

    plt.xticks(x, plot_models)
    plt.ylabel("Success (mean over tasks 1–5)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, axis="y", alpha=0.25)
    plt.title("Moderate physics shift")
    plt.tight_layout()

    out_png = out_dir / "eval_moderate_shift_baseline_vs_finetuned.png"
    out_pdf = out_dir / "eval_moderate_shift_baseline_vs_finetuned.pdf"
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    plt.close()

    # --- Plot 2: all conditions with error bars ---
    plt.figure(figsize=(7.6, 3.9))
    group_x = np.arange(len(cond_keys))
    width = 0.36
    offsets = {
        "Frozen baseline": -width / 2,
        "Online finetuned": +width / 2,
    }

    for m in plot_models:
        seed_map = models.get(m, {})
        m_means = []
        m_errs = []
        for key in cond_keys:
            vals = [conds[key] for _, conds in sorted(seed_map.items()) if key in conds]
            m_means.append(float(np.mean(vals)) if len(vals) else float("nan"))
            m_errs.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

        xs = group_x + offsets[m]
        plt.bar(
            xs,
            m_means,
            width=width,
            yerr=m_errs,
            capsize=4,
            color=colors.get(m, "#999999"),
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
            label=m,
        )

        # Per-seed points (jitter within bar)
        rng = np.random.RandomState(1)
        for i, key in enumerate(cond_keys):
            ys = [conds[key] for _, conds in sorted(seed_map.items()) if key in conds]
            jx = xs[i] + (rng.rand(len(ys)) - 0.5) * (width * 0.45)
            plt.scatter(jx, ys, s=18, color="black", alpha=0.6, zorder=3)

    plt.xticks(group_x, [_condition_display(v, s) for (v, s) in cond_keys], rotation=0)
    plt.ylabel("Success (mean over tasks 1–5)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, axis="y", alpha=0.25)

    env_name = meta_any.get("env_name", "")
    shift_family = meta_any.get("shift_family", "")
    title_bits = [b for b in [env_name, f"shift={shift_family}" if shift_family else ""] if b]
    plt.title("  ".join(title_bits))
    plt.legend(frameon=False, ncols=2, loc="lower right")
    plt.tight_layout()

    out_png = out_dir / "eval_success_all_conditions_baseline_vs_finetuned.png"
    out_pdf = out_dir / "eval_success_all_conditions_baseline_vs_finetuned.pdf"
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, nargs="+", required=True, help="One raw_log.csv or one/more eval summary CSVs.")
    p.add_argument("--out_dir", type=str, default=None, help="Defaults to the CSV's directory.")
    args = p.parse_args()

    csv_paths = [Path(c).expanduser().resolve() for c in args.csv]
    for csv_path in csv_paths:
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else csv_paths[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    header0 = _read_header(csv_paths[0])
    if _is_raw_log_csv(header0):
        if len(csv_paths) != 1:
            raise ValueError("Raw log plotting expects exactly one CSV path.")
        csv_path = csv_paths[0]
        data = read_raw_log(csv_path)
        meta = data["meta"]
        title = f"{meta.get('method','')}  {meta.get('env_name','')}  task={meta.get('task_id','')}  ckpt={meta.get('offline_step','')}"

        # --- Eval success vs env steps ---
        plt.figure(figsize=(7, 4))
        plt.plot(data["eval_env_steps"], data["eval_success"], marker="o", linewidth=2)
        plt.xlabel("Environment steps")
        plt.ylabel("Eval success (mean)")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        plt.title(title.strip())
        out_path = out_dir / "eval_success.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        print(f"wrote {out_path}")
        plt.close()

        # --- Episode return vs env steps ---
        plt.figure(figsize=(7, 4))
        plt.plot(data["ep_env_steps"], data["ep_return"], marker="o", linewidth=1.5)
        plt.xlabel("Environment steps (episode end)")
        plt.ylabel("Episode return")
        plt.grid(True, alpha=0.3)
        plt.title(title.strip())
        out_path = out_dir / "episode_return_vs_env_steps.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        print(f"wrote {out_path}")
        plt.close()

        # --- Episode return vs episode idx ---
        plt.figure(figsize=(7, 4))
        plt.plot(data["ep_idx"], data["ep_return"], marker="o", linewidth=1.5)
        plt.xlabel("Episode index")
        plt.ylabel("Episode return")
        plt.grid(True, alpha=0.3)
        plt.title(title.strip())
        out_path = out_dir / "episode_return_vs_episode_idx.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        print(f"wrote {out_path}")
        plt.close()

    elif _is_eval_summary_csv(header0):
        # Validate all provided CSVs match this schema.
        for pth in csv_paths[1:]:
            h = _read_header(pth)
            if not _is_eval_summary_csv(h):
                raise ValueError(f"Mixed CSV formats; {pth} does not look like an eval summary CSV.")
        if len(csv_paths) <= 2:
            plot_eval_comparison(csv_paths, out_dir)
        else:
            plot_eval_across_seeds(csv_paths, out_dir)
    else:
        raise ValueError(
            "Unrecognized CSV format. Expected raw_log.csv (type/env_step) or eval summary CSV "
            "(variant/shift_level/task_id/success)."
        )


if __name__ == "__main__":
    main()

