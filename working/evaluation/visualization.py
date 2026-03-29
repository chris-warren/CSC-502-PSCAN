#!/usr/bin/env python3
"""
Visualization for PSCAN clustering results.

- Loads graph (.adjlist)
- Loads ground truth labels
- Loads predicted clusters
- Plots graph with predicted cluster colors

Cluster files are now epsilon-tagged (e.g. lfr_500.sim_eps0.4_clusters.csv).
The best_epsilon folder is used by default to find the correct cluster file.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import ast
import glob


def find_cluster_file(output_dir, dataset_name, epsilon=None):
    """
    Find the cluster file for a given dataset.

    Priority:
    1. If epsilon is given → look for exact epsilon-tagged file in clusters/
    2. If epsilon is None  → look in best_epsilon/ folder for best_eps file
    3. Fallback            → pick the first matching file in clusters/

    Parameters
    ----------
    output_dir  : Path — root output directory (e.g. /scratch/nidita/pscan_output)
    dataset_name: str  — e.g. "lfr_500"
    epsilon     : float or None

    Returns
    -------
    Path to cluster file, or None if not found.
    """
    output_dir = Path(output_dir)

    # Priority 1: exact epsilon-tagged file
    if epsilon is not None:
        eps_tag = f"eps{epsilon}"
        exact = output_dir / "clusters" / f"{dataset_name}.sim_{eps_tag}_clusters.csv"
        if exact.exists():
            print(f"[viz] Using cluster file: {exact}")
            return exact

    # Priority 2: best_epsilon folder
    best_dir = output_dir / "best_epsilon"
    if best_dir.exists():
        matches = sorted(best_dir.glob(f"{dataset_name}_best_eps*_clusters.csv"))
        if matches:
            print(f"[viz] Using best-epsilon cluster file: {matches[0]}")
            return matches[0]

    # Priority 3: fallback — any matching file in clusters/
    fallback = sorted((output_dir / "clusters").glob(f"{dataset_name}*_clusters.csv"))
    if fallback:
        print(f"[viz] Fallback cluster file: {fallback[0]}")
        return fallback[0]

    print(f"[viz] ERROR: No cluster file found for {dataset_name}")
    return None


def load_predicted_labels(cluster_file):
    """
    Convert cluster CSV → DataFrame with columns [node, cluster]
    """
    df = pd.read_csv(cluster_file, header=None, names=["cluster", "nodes"])

    # nodes stored as string list → convert
    df["nodes"] = df["nodes"].apply(ast.literal_eval)
    df = df.explode("nodes").reset_index(drop=True)
    df["node"] = df["nodes"].astype(int)

    return df[["node", "cluster"]]


def visualize(adjlist_path, gt_path, output_dir, dataset_name, epsilon=None):
    """
    Plot graph with predicted cluster coloring.

    Parameters
    ----------
    adjlist_path  : path to .adjlist file
    gt_path       : path to ground truth labels .tsv
    output_dir    : root output directory to find cluster files
    dataset_name  : e.g. "lfr_500"
    epsilon       : specific epsilon to use (None → use best epsilon)
    """
    adjlist_path = Path(adjlist_path)
    gt_path      = Path(gt_path)
    output_dir   = Path(output_dir)

    # ----------------------------
    # Find cluster file
    # ----------------------------
    cluster_file = find_cluster_file(output_dir, dataset_name, epsilon)
    if cluster_file is None:
        return

    # ----------------------------
    # Load data
    # ----------------------------
    print("[viz] Loading graph...")
    G = nx.read_adjlist(adjlist_path, nodetype=int)

    print("[viz] Loading ground truth...")
    gt_df = pd.read_csv(gt_path, sep="\t")

    print(f"[viz] Loading predicted clusters from {cluster_file.name}...")
    pred_df = load_predicted_labels(cluster_file)

    # ----------------------------
    # Merge labels
    # ----------------------------
    df = pd.merge(gt_df, pred_df, on="node", how="inner")

    # ----------------------------
    # Create color map
    # ----------------------------
    classes = sorted(df["cluster"].unique())
    cmap = cm.get_cmap("tab20", len(classes))
    class_color = {cls: cmap(i) for i, cls in enumerate(classes)}

    node_colors = []
    for node in G.nodes():
        row = df[df["node"] == node]
        if not row.empty:
            label = row["cluster"].values[0]
            node_colors.append(class_color[label])
        else:
            node_colors.append((0.5, 0.5, 0.5))  # gray for unmatched nodes

    # ----------------------------
    # Layout
    # ----------------------------
    print("[viz] Computing layout...")
    pos = nx.spring_layout(G, seed=42)

    # ----------------------------
    # Plot
    # ----------------------------
    print("[viz] Drawing graph...")
    plt.figure(figsize=(12, 8))

    nx.draw(
        G,
        pos,
        node_color=node_colors,
        node_size=30,
        edge_color="gray",
        with_labels=False,
    )

    eps_label = f"ε={epsilon}" if epsilon is not None else "best ε"
    plt.title(f"PSCAN Clustering Visualization — {dataset_name} ({eps_label})")
    plt.tight_layout()
    plt.show()


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    SCRATCH_DIR  = "/scratch/nidita/pscan_output"
    DATASET_NAME = "lfr_500"

    visualize(
        adjlist_path  = f"{SCRATCH_DIR}/adjlists/{DATASET_NAME}.adjlist",
        gt_path       = f"{SCRATCH_DIR}/labels/{DATASET_NAME}.labels.tsv",
        output_dir    = SCRATCH_DIR,
        dataset_name  = DATASET_NAME,
        epsilon       = None,   # None → automatically uses best epsilon
    )