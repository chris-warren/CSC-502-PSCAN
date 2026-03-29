#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict

SCRATCH_DIR = Path("/scratch/nidita/pscan_output")

def create_filtered_adjlist_and_LPCC_emitter(project_root, tsv_file, EPSILON=0.5):
    tsv_file = Path(tsv_file)
    filtered_dir = SCRATCH_DIR / "filtered_adjlists"
    parsed_dir   = SCRATCH_DIR / "parsed_input"
    filtered_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    adj_dict = defaultdict(list)
    with tsv_file.open("r") as file:
        next(file)
        for line in file:
            u, v, sim = line.strip().split("\t")
            if float(sim) >= EPSILON:
                adj_dict[u].append(v)
                adj_dict[v].append(u)

    # Include epsilon in filename so each run saves a separate file
    eps_tag = f"eps{EPSILON}"

    filtered_path = filtered_dir / f"filtered_edge_{tsv_file.stem}_{eps_tag}.tsv"
    with filtered_path.open("w") as f:
        for node, neighbors in adj_dict.items():
            f.write(f"{node}\t{' '.join(neighbors)}\n")

    parsed_path = parsed_dir / f"parse_{tsv_file.stem}_{eps_tag}.tsv"
    with parsed_path.open("w") as f:
        for node, neighbors in adj_dict.items():
            adj_str = " ".join(neighbors)
            f.write(f"{node},True,{node},{adj_str}\n")

    print(f"[LPSS] Filtered graph saved -> {filtered_path}")
    print(f"[LPSS] LPCC input saved     -> {parsed_path}")
    return str(parsed_path)