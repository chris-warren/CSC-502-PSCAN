#!/usr/bin/env python3
from pathlib import Path
import csv
import sys

from LPSS import create_filtered_adjlist_and_LPCC_emitter
from LPSS_pyspark import run_lpcc
from hub_outlier import detect, write_classification, summary

SCRATCH_DIR = Path("/scratch/nidita/pscan_output")


def cluster_results(sim_path, epsilon, PROJECT_ROOT, mu=1, sc=None):
    """
    Full PSCAN pipeline:
        Similarity → Prune → LPCC → Node Classification (core/hub/outlier)

    Parameters
    ----------
    sim_path     : path to similarity TSV file
    epsilon      : similarity threshold for pruning
    PROJECT_ROOT : root path of the project
    mu           : minimum neighbors in pruned graph to be a core node

    Returns
    -------
    labels         : {node: cluster_label}
    classification : {node: "core" | "hub" | "outlier"}
    output_paths   : {
                        "clusters"        : Path to cluster CSV,
                        "filtered_adjlist": Path to filtered adjacency list,
                        "parsed_input"    : Path to parsed LPCC input,
                        "classification"  : Path to classification TSV,
                     }
    """
    sim_path = Path(sim_path)
    eps_tag  = f"eps{epsilon}"

    # Step 1 & 2: Prune edges by epsilon and prepare LPCC input
    # LPSS.py now saves files with epsilon in the name
    parsed_input_path = create_filtered_adjlist_and_LPCC_emitter(
        project_root=PROJECT_ROOT,
        tsv_file=sim_path,
        EPSILON=epsilon,
    )
    parsed_input_path = Path(parsed_input_path)

    # Path produced by LPSS.py — includes epsilon tag
    filtered_adj_path = (
        SCRATCH_DIR / "filtered_adjlists"
        / f"filtered_edge_{sim_path.stem}_{eps_tag}.tsv"
    )

    # Step 3: Label propagation (LPCC) → cluster labels
    print(f"[clustering] Running LPCC on {parsed_input_path} ...")
    labels = run_lpcc(str(parsed_input_path))

    # Build cluster groups
    clusters = {}
    for node, label in labels.items():
        clusters.setdefault(label, []).append(node)
    for k in clusters:
        clusters[k] = sorted(clusters[k])

    # Save cluster CSV — includes epsilon tag
    output_dir = SCRATCH_DIR / "clusters"
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_file = output_dir / f"{sim_path.stem}_{eps_tag}_clusters.csv"

    with cluster_file.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for cluster_id, nodes in sorted(clusters.items()):
            writer.writerow([cluster_id, nodes])

    print(f"[clustering] Clusters saved -> {cluster_file}")
    print(f"[clustering] Total clusters: {len(clusters)}")

    # Step 4: Node classification (core / hub / outlier)
    pruned_adj = {}
    if filtered_adj_path.exists():
        with filtered_adj_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                node = int(parts[0])
                neighbors = (
                    [int(v) for v in parts[1].split()]
                    if len(parts) > 1 and parts[1].strip()
                    else []
                )
                pruned_adj[node] = neighbors
    else:
        print(f"[clustering] WARNING: Pruned adjlist not found at {filtered_adj_path}. "
              f"Using empty adjacency for classification.")
        pruned_adj = {node: [] for node in labels}

    int_labels = {int(k): int(v) for k, v in labels.items()}
    classification = detect(int_labels, pruned_adj, mu=mu)
    counts = summary(classification)

    print(f"[clustering] Core: {counts['core']:,}  "
          f"Hub: {counts['hub']:,}  "
          f"Outlier: {counts['outlier']:,}")

    # Save classification TSV — includes epsilon tag
    class_dir = SCRATCH_DIR / "classifications"
    class_dir.mkdir(parents=True, exist_ok=True)
    class_file = class_dir / f"{sim_path.stem}_{eps_tag}_classification.tsv"
    write_classification(classification, class_file)

    output_paths = {
        "clusters":         cluster_file,
        "filtered_adjlist": filtered_adj_path,
        "parsed_input":     parsed_input_path,
        "classification":   class_file,
    }

    return labels, classification, output_paths