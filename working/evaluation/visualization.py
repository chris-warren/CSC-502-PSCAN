#!/usr/bin/env python3

"""
Visualization for PSCAN clustering results.

- Loads graph (.adjlist)
- Loads ground truth labels
- Loads predicted clusters
- Plots graph with predicted cluster colors
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import ast


def load_predicted_labels(cluster_file):
    """
    Convert cluster CSV → node → label
    """
    df = pd.read_csv(cluster_file, header=None, names=["cluster", "nodes"])

    # nodes stored as string list → convert
    df["nodes"] = df["nodes"].apply(ast.literal_eval)

    df = df.explode("nodes").reset_index(drop=True)
    df["node"] = df["nodes"].astype(int)

    return df[["node", "cluster"]]


def visualize(adjlist_path, gt_path, cluster_file):
    """
    Plot graph with predicted cluster coloring
    """

    adjlist_path = Path(adjlist_path)
    gt_path = Path(gt_path)
    cluster_file = Path(cluster_file)

    # ----------------------------
    # Load data
    # ----------------------------
    print("[viz] Loading graph...")
    G = nx.read_adjlist(adjlist_path, nodetype=int)

    print("[viz] Loading ground truth...")
    gt_df = pd.read_csv(gt_path, sep="\t")

    print("[viz] Loading predicted clusters...")
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
            node_colors.append((0.5, 0.5, 0.5))  # gray for missing

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

    plt.title("PSCAN Clustering Visualization")
    plt.tight_layout()
    plt.show()


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    visualize(
        adjlist_path="data/output/adjlists/lfr_500.adjlist",
        gt_path="data/output/labels/lfr_500.labels.tsv",
        cluster_file="data/output/clusters/lfr_500_clusters.csv",
    )