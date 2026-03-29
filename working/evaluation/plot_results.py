#!/usr/bin/env python3
"""
Plot PSCAN experiment results.

Experiment 1 — accuracy.png:
    ARI and NMI vs epsilon, one line per dataset size.

Experiment 2 — runtime.png:
    Wall-clock time vs number of machines, one line per BA graph size.
    (Matches the speedup/scaleup plots in the paper.)

Usage:
    python evaluation/plot_results.py \
        --accuracy results_accuracy.csv \
        --runtime  results_runtime.csv \
        --output-dir figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


# ---------------------------------------------------------------------------
# Experiment 1 — Accuracy plot
# ---------------------------------------------------------------------------

def plot_accuracy(csv_path: Path, output_dir: Path) -> None:
    """ARI and NMI vs epsilon, one line per dataset."""
    df = pd.read_csv(csv_path)
    if df.empty:
        print("[plot] Accuracy CSV is empty, skipping.")
        return

    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle("Experiment 1 — Accuracy vs Epsilon", fontweight="bold")

    for ax, metric in zip(axes, ["ari", "nmi"]):
        for i, name in enumerate(datasets):
            sub = df[df["dataset"] == name].sort_values("epsilon")
            ax.plot(
                sub["epsilon"], sub[metric],
                marker=MARKERS[i % len(MARKERS)],
                linewidth=2, markersize=5,
                color=COLORS[i % len(COLORS)],
                label=name,
            )
        ax.set_xlabel("Epsilon (ε)")
        ax.set_ylabel(metric.upper())
        ax.set_title(metric.upper())
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9, framealpha=0.5)

    plt.tight_layout()
    out = output_dir / "accuracy.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[plot] Accuracy plot → {out}")


# ---------------------------------------------------------------------------
# Experiment 2 — Runtime plot
# ---------------------------------------------------------------------------

def plot_runtime(csv_path: Path, output_dir: Path) -> None:
    """Wall-clock time vs number of machines, one line per BA graph size."""
    df = pd.read_csv(csv_path)
    if df.empty:
        print("[plot] Runtime CSV is empty, skipping.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Experiment 2 — Runtime vs Number of Machines", fontweight="bold")

    sizes = sorted(df["n_nodes"].unique())

    for i, n_nodes in enumerate(sizes):
        sub = df[df["n_nodes"] == n_nodes].sort_values("n_machines")
        label = f"n={n_nodes:,}"
        ax.plot(
            sub["n_machines"], sub["time_s"],
            marker=MARKERS[i % len(MARKERS)],
            linewidth=2, markersize=5,
            color=COLORS[i % len(COLORS)],
            label=label,
        )

    ax.set_xlabel("Number of machines (workers)")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_xticks(sorted(df["n_machines"].unique()))
    ax.legend(fontsize=9, framealpha=0.5, title="Graph size")

    plt.tight_layout()
    out = output_dir / "runtime.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[plot] Runtime plot → {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PSCAN experiment results.")
    parser.add_argument("--accuracy",   type=Path, default=None)
    parser.add_argument("--runtime",    type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.accuracy is None and args.runtime is None:
        print("[plot] Pass --accuracy and/or --runtime.")
        return

    if args.accuracy:
        if args.accuracy.exists():
            plot_accuracy(args.accuracy, args.output_dir)
        else:
            print(f"[plot] Not found: {args.accuracy}")

    if args.runtime:
        if args.runtime.exists():
            plot_runtime(args.runtime, args.output_dir)
        else:
            print(f"[plot] Not found: {args.runtime}")


if __name__ == "__main__":
    main()