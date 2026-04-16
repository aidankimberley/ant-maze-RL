from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SHIFT_ORDER = [
    "base",
    "mild_low",
    "moderate_low",
    "severe_low",
    "mild_high",
    "moderate_high",
    "severe_high",
    "mild",
    "moderate",
    "severe",
]

SHIFT_LABELS = {
    "base": "base",
    "mild_low": "mild low",
    "moderate_low": "mod low",
    "severe_low": "sev low",
    "mild_high": "mild high",
    "moderate_high": "mod high",
    "severe_high": "sev high",
    "mild": "mild",
    "moderate": "moderate",
    "severe": "severe",
}


def load_all_csvs(results_dir: Path, pattern: str) -> pd.DataFrame:
    paths = sorted(results_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No CSVs found matching {pattern} in {results_dir}")

    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--pattern", type=str, default="medium_composite_shift_seed*.csv")
    parser.add_argument("--output_png", type=str, default="results/medium_composite_shift_plot.png")
    parser.add_argument("--output_summary_csv", type=str, default="results/medium_composite_shift_summary.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    df = load_all_csvs(results_dir, args.pattern)

    # Normalize shift level names
    df["shift_level_plot"] = df["shift_level"].replace({"base": "base"})
    df.loc[df["variant"] == "nominal", "shift_level_plot"] = "base"

    # Aggregate over tasks within each seed first
    seed_level = (
        df.groupby(["seed", "shift_level_plot"], as_index=False)["success"]
        .mean()
    )

    # Then aggregate across seeds
    summary = (
        seed_level.groupby("shift_level_plot", as_index=False)["success"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_success", "std": "std_success"})
    )

    summary["order"] = summary["shift_level_plot"].map({k: i for i, k in enumerate(SHIFT_ORDER)})
    summary = summary.sort_values("order").reset_index(drop=True)
    summary["label"] = summary["shift_level_plot"].map(SHIFT_LABELS)

    # Save summary table
    summary[["shift_level_plot", "label", "mean_success", "std_success"]].to_csv(
        args.output_summary_csv, index=False
    )

    # Plot
    x = range(len(summary))
    y = summary["mean_success"]
    yerr = summary["std_success"].fillna(0.0)

    plt.figure(figsize=(9, 5))
    plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4)
    plt.xticks(list(x), summary["label"], rotation=25)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Composite physics shift")
    plt.ylabel("Mean success")
    plt.title("HIQL zero-shot performance under physics shift")
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=200)

    print("Saved plot to:", args.output_png)
    print("Saved summary CSV to:", args.output_summary_csv)
    print("\nSummary:")
    print(summary[["label", "mean_success", "std_success"]].to_string(index=False))


if __name__ == "__main__":
    main()