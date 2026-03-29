#!/usr/bin/env python3
"""
Runtime experiment on LFR graphs.
Measures wall-clock time for each LFR size x machine count.
Uses best epsilon from Experiment 1 (accuracy) if available,
otherwise falls back to --epsilon flag.
Speedup is calculated relative to 4-CPU baseline.
Results saved to results_runtime_lfr.csv
"""
import argparse
import csv
import sys
import time
import multiprocessing
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
SIM_DIR      = PROJECT_ROOT / "similarity"
CLUST_DIR    = PROJECT_ROOT / "clustering"
EVAL_DIR     = PROJECT_ROOT / "evaluation"
SCRATCH_DIR  = Path("/scratch/nidita/pscan_output")

for p in [PROJECT_ROOT, DATA_DIR, SIM_DIR, CLUST_DIR, EVAL_DIR]:
    sys.path.insert(0, str(p))

from similarity_main import run_pipeline as run_similarity
from LPSS_main import cluster_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=SCRATCH_DIR)
    parser.add_argument("--lfr-sizes", nargs="*", type=int,
                        default=[500, 1000, 2000, 5000, 10000, 20000, 40000, 80000, 160000])
    parser.add_argument("--machines", nargs="*", type=int, default=[4, 8, 15])
    parser.add_argument("--epsilon", type=float, default=0.4,
                        help="Fallback epsilon if accuracy results not found.")
    parser.add_argument("--mu", type=int, default=1,
                        help="Minimum neighbors in pruned graph to be a core node.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def best_epsilon_from_results(results_file):
    """
    Read results_accuracy.csv and return the epsilon with
    the highest mean ARI across all LFR dataset sizes.
    Returns None if file does not exist or is empty.
    """
    if not results_file.exists():
        return None
    try:
        rows = defaultdict(list)
        with results_file.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows[float(row["epsilon"])].append(float(row["ari"]))
        if not rows:
            return None
        best = max(rows, key=lambda e: sum(rows[e]) / len(rows[e]))
        print(f"[runtime_lfr] Best epsilon from accuracy results: {best}")
        return best
    except Exception as exc:
        print(f"[runtime_lfr] Could not determine best epsilon: {exc}")
        return None


def run_pipeline_with_workers(adj_path, sim_path, eps, mu, n_workers, verbose):
    """
    Run similarity + clustering using a multiprocessing Pool
    with n_workers processes to simulate parallel CPU execution.

    Returns elapsed wall-clock time in seconds.
    """
    t_start = time.perf_counter()

    # Similarity stage — run with worker pool
    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.apply(run_similarity, kwds=dict(
            input_path=adj_path,
            output_path=sim_path,
            verbose=verbose,
        ))

    # Clustering stage — runs after similarity
    cluster_results(
        sim_path=sim_path,
        epsilon=eps,
        PROJECT_ROOT=PROJECT_ROOT,
        mu=mu,
    )

    return time.perf_counter() - t_start


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Use best epsilon from Experiment 1 if available
    eps = best_epsilon_from_results(SCRATCH_DIR / "results_accuracy.csv")
    if eps is None:
        eps = args.epsilon
        print(f"[runtime_lfr] Accuracy results not found. "
              f"Using fallback epsilon={eps} (from --epsilon flag)")

    results_file = SCRATCH_DIR / "results_runtime_lfr.csv"
    with results_file.open("w") as f:
        f.write("dataset,n_nodes,n_machines,epsilon,time_s,speedup\n")

    print(f"\n[runtime_lfr] LFR sizes : {args.lfr_sizes}")
    print(f"[runtime_lfr] Machines  : {args.machines}")
    print(f"[runtime_lfr] Epsilon   : {eps}")
    print(f"[runtime_lfr] Mu        : {args.mu}")

    for n in args.lfr_sizes:
        name     = f"lfr_{n}"
        adj_path = args.output_dir / "adjlists" / f"{name}.adjlist"

        if not adj_path.exists():
            print(f"[runtime_lfr] {adj_path} not found, skipping.")
            continue

        sim_path = args.output_dir / "adjlists" / f"{name}.sim.tsv"

        # baseline_time set at smallest machine count (4 CPUs → 1×)
        baseline_time = None

        for n_machines in sorted(args.machines):
            actual = min(n_machines, multiprocessing.cpu_count())
            print(f"\n--- {name}  machines={n_machines} (workers={actual}) ---")

            elapsed = run_pipeline_with_workers(
                adj_path=adj_path,
                sim_path=sim_path,
                eps=eps,
                mu=args.mu,
                n_workers=actual,
                verbose=args.verbose,
            )

            # First (smallest) machine count = baseline (4 CPUs → 1×)
            if baseline_time is None:
                baseline_time = elapsed

            speedup = baseline_time / elapsed if elapsed > 0 else 1.0

            print(f"  -> {elapsed:.2f}s  (speedup vs 4-CPU baseline: {speedup:.2f}x)")

            with results_file.open("a") as f:
                f.write(f"{name},{n},{n_machines},{eps},{elapsed:.4f},{speedup:.4f}\n")

    print(f"\n[runtime_lfr] Results -> {results_file}")


if __name__ == "__main__":
    main()