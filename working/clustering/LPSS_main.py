#!/usr/bin/env python3

from LPSS import create_filtered_adjlist_and_LPCC_emitter
from LPSS_pyspark import run_lpcc
import csv
from pathlib import Path

SCRATCH_DIR = Path("/scratch/nidita/pscan_output")


def cluster_results(sim_path, epsilon, PROJECT_ROOT, sc=None):
    """
    Runs full clustering pipeline:
    1. Edge pruning (epsilon-threshold)
    2. LPCC (connected components)
    3. Save clusters to file

    Returns:
        dict: node -> cluster_label
    """

    sim_path = Path(sim_path)

    # -------------------------------------------------
    # Step 1: Filter graph + prepare LPCC input
    # -------------------------------------------------
    parsed_input_path = create_filtered_adjlist_and_LPCC_emitter(
        project_root=PROJECT_ROOT,
        tsv_file=sim_path,
        EPSILON=epsilon,
    )

    # -------------------------------------------------
    # Step 2: Run LPCC (pure Python)
    # -------------------------------------------------
    print(f"[clustering] Running LPCC on {parsed_input_path} ...")
    labels = run_lpcc(parsed_input_path)

    # labels: {node: cluster_label}

    # -------------------------------------------------
    # Step 3: Convert to cluster -> nodes
    # -------------------------------------------------
    clusters = {}
    for node, label in labels.items():
        clusters.setdefault(label, []).append(node)

    for k in clusters:
        clusters[k] = sorted(clusters[k])

    # -------------------------------------------------
    # Step 4: Save output to scratch
    # -------------------------------------------------
    output_dir = SCRATCH_DIR / "clusters"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{sim_path.stem}_clusters.csv"

    with output_file.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for cluster_id, nodes in sorted(clusters.items()):
            writer.writerow([cluster_id, nodes])

    print(f"[clustering] Clusters saved -> {output_file}")
    print(f"[clustering] Total clusters: {len(clusters)}")

    # -------------------------------------------------
    # Step 5: Return node -> label mapping
    # -------------------------------------------------
    return labels