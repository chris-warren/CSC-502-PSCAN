#!/usr/bin/env python3
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

    gt_df = pd.read_csv(gt_path, sep="\t")

    if "node" not in gt_df.columns or "label" not in gt_df.columns:
        raise ValueError("Ground truth file must contain columns: node, label")

    pred_df = pd.DataFrame(
        list(pred_labels_dict.items()), columns=["node", "predicted_label"]
    )

    # Merge on node — only evaluate nodes that appear in both
    df = pd.merge(gt_df, pred_df, on="node")

    total_gt   = len(gt_df)
    total_pred = len(pred_df)
    matched    = len(df)

    print(f"[evaluate] Ground truth nodes : {total_gt}")
    print(f"[evaluate] Predicted nodes    : {total_pred}")
    print(f"[evaluate] Matched nodes      : {matched}")

    if matched == 0:
        print("[evaluate] WARNING: No matching nodes — returning ARI=0, NMI=0")
        return 0.0, 0.0

    if matched < total_gt:
        print(f"[evaluate] NOTE: {total_gt - matched} nodes had no prediction "
              f"(pruned by epsilon) — evaluating on matched nodes only")

    ari = adjusted_rand_score(df["label"], df["predicted_label"])
    nmi = normalized_mutual_info_score(df["label"], df["predicted_label"])

    print(f"\n[Evaluation Results]")
    print(f"ARI: {ari:.4f}")
    print(f"NMI: {nmi:.4f}")

    return ari, nmi
