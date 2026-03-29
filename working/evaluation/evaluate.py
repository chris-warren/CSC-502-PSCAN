#!/usr/bin/env python3

"""
Evaluation module for PSCAN.

Computes:
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)

Inputs:
- Ground truth labels file (from datasets.py)
- Predicted labels (from clustering, passed as dict)

Output:
- Prints ARI and NMI
"""

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def evaluate(gt_path, pred_labels_dict):
    """
    Parameters
    ----------
    gt_path : str or Path
        Path to ground truth labels (.labels.tsv)

    pred_labels_dict : dict
        {node: predicted_cluster_label}

    Returns
    -------
    (ari, nmi)
    """

    # ----------------------------
    # Load ground truth
    # ----------------------------
    gt_df = pd.read_csv(gt_path, sep="\t")

    if "node" not in gt_df.columns or "label" not in gt_df.columns:
        raise ValueError("Ground truth file must contain columns: node, label")

    # ----------------------------
    # Convert predictions to DataFrame
    # ----------------------------
    pred_df = pd.DataFrame(
        list(pred_labels_dict.items()), columns=["node", "predicted_label"]
    )

    # ----------------------------
    # Merge on node
    # ----------------------------
    df = pd.merge(gt_df, pred_df, on="node")

    if df.empty:
        raise ValueError("No matching nodes between ground truth and predictions")

    # ----------------------------
    # Compute metrics
    # ----------------------------
    ari = adjusted_rand_score(df["label"], df["predicted_label"])
    nmi = normalized_mutual_info_score(df["label"], df["predicted_label"])

    print("\n[Evaluation Results]")
    print(f"ARI: {ari:.4f}")
    print(f"NMI: {nmi:.4f}")

    return ari, nmi