#!/usr/bin/env python3
"""
Edge pruning + adjacency preparation for PSCAN (epsilon-threshold step).

Writes all output to /scratch to avoid permission errors on /project.
"""

from pathlib import Path
from collections import defaultdict

SCRATCH_DIR = Path("/scratch/nidita/pscan_output")


def create_filtered_adjlist_and_LPCC_emitter(
    project_root, tsv_file, EPSILON: float = 0.5
) -> str:
    """
    Parameters
    ----------
    project_root : Path or str
    tsv_file : Path or str
        similarity file (.sim.tsv)
    EPSILON : float
        threshold for pruning edges

    Returns
    -------
    str : path to LPCC input file
    """

    tsv_file = Path(tsv_file)

    # Output directories — always write to scratch
    filtered_dir = SCRATCH_DIR / "filtered_adjlists"
    parsed_dir   = SCRATCH_DIR / "parsed_input"

    filtered_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    # Build adjacency list after filtering
    adj_dict = defaultdict(list)

    with tsv_file.open("r") as file:
        next(file)  # skip header

        for line in file:
            u, v, sim = line.strip().split("\t")

            if float(sim) >= EPSILON:
                adj_dict[u].append(v)
                adj_dict[v].append(u)

    # ----------------------------
    # 1. Write filtered adjacency list
    # ----------------------------
    filtered_path = filtered_dir / f"filtered_edge_{tsv_file.stem}.tsv"

    with filtered_path.open("w") as f:
        for node, neighbors in adj_dict.items():
            f.write(f"{node}\t{' '.join(neighbors)}\n")

    # ----------------------------
    # 2. Write LPCC input
    # ----------------------------
    parsed_path = parsed_dir / f"parse_{tsv_file.stem}.tsv"

    with parsed_path.open("w") as f:
        for node, neighbors in adj_dict.items():
            adj_str = " ".join(neighbors)
            # format: node, active, label, adjacency
            f.write(f"{node},True,{node},{adj_str}\n")

    print(f"[LPSS] Filtered graph saved -> {filtered_path}")
    print(f"[LPSS] LPCC input saved -> {parsed_path}")

    return str(parsed_path)