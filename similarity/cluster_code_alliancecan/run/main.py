#!/usr/bin/env python3
"""PSCAN full pipeline — integration entry point.

Folder layout
-------------
project/
    data/           datasets.py         (Division 1 — Shikha)
    similarity/     similarity_mapper.py
                    similarity_reducer.py
                    similarity_main.py  (Division 2 — Nidita)
    run/            main.py             ← this file
                    job.sh

This script ties all divisions together:
    1. Generate datasets          (data/datasets.py)
    2. Compute similarities       (similarity/similarity_main.py)
    3. Cluster + hub/outlier      (clustering/clustering.py)   — Van, Division 3
    4. Evaluate ARI/NMI + runtime (evaluation/evaluation.py)   — Chris, Division 4

For now Divisions 3 and 4 are stubbed — replace the stubs with real imports
once Van and Chris deliver their modules.

Usage
-----
    # Run full pipeline on all LFR paper sizes
    python main.py --output-dir ../data/output --paper-scales --seed 42 --verbose

    # Run on small graph for testing
    python main.py --output-dir ../data/output --lfr-sizes 500 --seed 42 --verbose

    # Skip dataset generation (datasets already exist)
    python main.py --output-dir ../data/output --paper-scales --skip-datasets --verbose
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make data/ and similarity/ importable from run/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # project/
DATA_DIR      = PROJECT_ROOT / "data"
SIMILARITY_DIR = PROJECT_ROOT / "similarity"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DATA_DIR))
sys.path.insert(0, str(SIMILARITY_DIR))

# ---------------------------------------------------------------------------
# Division 1 — dataset generation (Shikha)
# ---------------------------------------------------------------------------
try:
    import datasets as datasets_module
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("[main] WARNING: data/datasets.py not found. Use --skip-datasets if already generated.")

# ---------------------------------------------------------------------------
# Division 2 — similarity (Nidita — this file's owner)
# ---------------------------------------------------------------------------
from similarity_main import run_pipeline as run_similarity

# ---------------------------------------------------------------------------
# Division 3 — clustering (Van) — STUB
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, str(PROJECT_ROOT / "clustering"))
    from clustering import run_clustering
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    def run_clustering(sim_path, output_path, epsilon, verbose=False):
        print(f"[main] STUB: clustering not yet integrated. sim={sim_path}", file=sys.stderr)
        return {}

# ---------------------------------------------------------------------------
# Division 4 — evaluation (Chris) — STUB
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
    from evaluation import run_evaluation
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    def run_evaluation(cluster_labels, ground_truth_path, verbose=False):
        print(f"[main] STUB: evaluation not yet integrated.", file=sys.stderr)
        return {}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PSCAN full pipeline: dataset → similarity → clustering → evaluation."
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "output",
                        help="Root output directory for all generated files.")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--epsilon",    type=float, default=0.5,
                        help="PSCAN epsilon threshold for edge pruning (Division 3).")

    # Dataset sizes
    parser.add_argument("--paper-scales", action="store_true",
                        help="Use the graph sizes from the PSCAN paper.")
    parser.add_argument("--lfr-sizes",    type=int, nargs="*", default=None)
    parser.add_argument("--ba-sizes",     type=int, nargs="*", default=None)

    # Skip flags
    parser.add_argument("--skip-datasets",   action="store_true",
                        help="Skip dataset generation (use existing .adjlist files).")
    parser.add_argument("--skip-similarity", action="store_true",
                        help="Skip similarity computation (use existing .sim.tsv files).")
    parser.add_argument("--skip-clustering", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")

    # LFR generation parameters
    parser.add_argument("--tau1",           type=float, default=3.0)
    parser.add_argument("--tau2",           type=float, default=1.5)
    parser.add_argument("--mu",             type=float, default=0.1)
    parser.add_argument("--average-degree", type=int,   default=15)
    parser.add_argument("--max-degree",     type=int,   default=75)
    parser.add_argument("--min-community",  type=int,   default=20)
    parser.add_argument("--max-community",  type=int,   default=100)
    parser.add_argument("--ba-m",           type=int,   default=7)

    parser.add_argument("--verbose", "-v", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step 1 — Dataset generation
# ---------------------------------------------------------------------------

def step_datasets(args: argparse.Namespace) -> None:
    """Call data/datasets.py to generate LFR and/or BA graphs."""
    if not DATASETS_AVAILABLE:
        raise RuntimeError("data/datasets.py is not importable. Check your folder structure.")

    print("\n[main] ── Step 1: Dataset Generation ──────────────────────────")

    # Build argv for datasets.py's own argparser
    argv = [
        "--output-dir", str(args.output_dir),
        "--seed",        str(args.seed),
        "--tau1",        str(args.tau1),
        "--tau2",        str(args.tau2),
        "--mu",          str(args.mu),
        "--average-degree", str(args.average_degree),
        "--max-degree",     str(args.max_degree),
        "--min-community",  str(args.min_community),
        "--max-community",  str(args.max_community),
        "--ba-m",           str(args.ba_m),
    ]

    if args.paper_scales:
        argv.append("--paper-scales")
    if args.lfr_sizes:
        argv += ["--lfr-sizes"] + [str(n) for n in args.lfr_sizes]
    if args.ba_sizes:
        argv += ["--ba-sizes"] + [str(n) for n in args.ba_sizes]

    old_argv = sys.argv
    sys.argv = ["datasets.py"] + argv
    try:
        datasets_module.main()
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# Step 2 — Similarity computation
# ---------------------------------------------------------------------------

def step_similarity(args: argparse.Namespace) -> dict:
    """Run similarity pipeline on every .adjlist in the output directory."""
    print("\n[main] ── Step 2: Structural Similarity (PCSS) ─────────────────")

    adjlist_dir = args.output_dir / "adjlists"
    if not adjlist_dir.exists():
        raise FileNotFoundError(f"Adjacency list directory not found: {adjlist_dir}")

    adjlist_files = sorted(adjlist_dir.glob("*.adjlist"))
    if not adjlist_files:
        raise FileNotFoundError(f"No .adjlist files found in {adjlist_dir}")

    all_sim = {}
    for adjlist_path in adjlist_files:
        sim_path = adjlist_dir / (adjlist_path.stem + ".sim.tsv")

        if sim_path.exists() and not args.skip_similarity:
            if args.verbose:
                print(f"[main]   Skipping {adjlist_path.name} (sim file already exists)")
            continue

        print(f"[main]   Processing {adjlist_path.name} ...")
        t0 = time.perf_counter()

        sim = run_similarity(
            input_path=adjlist_path,
            output_path=sim_path,
            verbose=args.verbose,
        )

        elapsed = time.perf_counter() - t0
        print(f"[main]   ✓ {adjlist_path.stem}: {len(sim):,} similarities in {elapsed:.2f}s → {sim_path.name}")
        all_sim[adjlist_path.stem] = sim_path

    return all_sim


# ---------------------------------------------------------------------------
# Step 3 — Clustering (Van — Division 3)
# ---------------------------------------------------------------------------

def step_clustering(args: argparse.Namespace, sim_files: dict) -> dict:
    """Run clustering on each similarity file."""
    print("\n[main] ── Step 3: Clustering (ε-pruning + LPCC) ────────────────")

    if not CLUSTERING_AVAILABLE:
        print("[main]   Division 3 (clustering) not yet integrated — skipping.")
        return {}

    cluster_dir = args.output_dir / "clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    all_clusters = {}
    for name, sim_path in sim_files.items():
        out_path = cluster_dir / f"{name}.clusters.tsv"
        print(f"[main]   Clustering {name} with ε={args.epsilon} ...")
        labels = run_clustering(sim_path, out_path, epsilon=args.epsilon, verbose=args.verbose)
        all_clusters[name] = (labels, out_path)

    return all_clusters


# ---------------------------------------------------------------------------
# Step 4 — Evaluation (Chris — Division 4)
# ---------------------------------------------------------------------------

def step_evaluation(args: argparse.Namespace, cluster_results: dict) -> None:
    """Run ARI/NMI evaluation on each clustered result."""
    print("\n[main] ── Step 4: Evaluation (ARI / NMI) ───────────────────────")

    if not EVALUATION_AVAILABLE:
        print("[main]   Division 4 (evaluation) not yet integrated — skipping.")
        return

    labels_dir = args.output_dir / "labels"

    for name, (labels, _) in cluster_results.items():
        gt_path = labels_dir / f"{name}.labels.tsv"
        if not gt_path.exists():
            print(f"[main]   No ground truth for {name} — skipping evaluation.")
            continue
        print(f"[main]   Evaluating {name} ...")
        results = run_evaluation(labels, gt_path, verbose=args.verbose)
        print(f"[main]   {name}: {results}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    print(f"[main] PSCAN Pipeline starting  (output → {args.output_dir})")

    # Step 1
    if not args.skip_datasets:
        step_datasets(args)
    else:
        print("\n[main] ── Step 1: Dataset Generation ── SKIPPED")

    # Step 2
    if not args.skip_similarity:
        sim_files = step_similarity(args)
    else:
        print("\n[main] ── Step 2: Similarity ── SKIPPED")
        adjlist_dir = args.output_dir / "adjlists"
        sim_files = {p.stem: p for p in adjlist_dir.glob("*.sim.tsv")} if adjlist_dir.exists() else {}

    # Step 3
    if not args.skip_clustering:
        cluster_results = step_clustering(args, sim_files)
    else:
        print("\n[main] ── Step 3: Clustering ── SKIPPED")
        cluster_results = {}

    # Step 4
    if not args.skip_evaluation:
        step_evaluation(args, cluster_results)
    else:
        print("\n[main] ── Step 4: Evaluation ── SKIPPED")

    elapsed = time.perf_counter() - t_start
    print(f"\n[main] Pipeline complete in {elapsed:.2f}s")


if __name__ == "__main__":
    main()