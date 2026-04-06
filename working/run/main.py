#!/usr/bin/env python3
import argparse
import sys
import time
import multiprocessing
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
SIM_DIR      = PROJECT_ROOT / "similarity"
CLUST_DIR    = PROJECT_ROOT / "clustering"
EVAL_DIR     = PROJECT_ROOT / "evaluation"
SCRATCH_DIR  = Path("/scratch/nidita/pscan_output")

for p in [PROJECT_ROOT, DATA_DIR, SIM_DIR, CLUST_DIR, EVAL_DIR]:
    sys.path.insert(0, str(p))

import datasets as datasets_module
from similarity_main import run_pipeline as run_similarity
from LPSS_main import cluster_results
from evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="PSCAN experiments")
    parser.add_argument("--output-dir", type=Path, default=SCRATCH_DIR)
    parser.add_argument("--lfr-sizes", nargs="*", type=int, default=[500, 1000, 2000])
    parser.add_argument("--ba-sizes", nargs="*", type=int,
                        default=[1000000, 2000000, 3000000, 4000000])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--eps-list", nargs="*", type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--mu", type=int, default=1,
                        help="Minimum neighbors in pruned graph to be a core node.")
    parser.add_argument("--machines", nargs="*", type=int, default=[4, 8, 15])
    parser.add_argument("--experiment", choices=["accuracy", "runtime"], default=None)
    parser.add_argument("--skip-datasets", action="store_true")
    parser.add_argument("--skip-similarity", action="store_true")
    parser.add_argument("--skip-clustering", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--paper-scales", action="store_true")
    parser.add_argument("--tau1", type=float, default=3.0)
    parser.add_argument("--tau2", type=float, default=1.5)
    parser.add_argument("--mu-lfr", type=float, default=0.1)
    parser.add_argument("--average-degree", type=int, default=15)
    parser.add_argument("--max-degree", type=int, default=75)
    parser.add_argument("--min-community", type=int, default=20)
    parser.add_argument("--max-community", type=int, default=100)
    parser.add_argument("--ba-m", type=int, default=7)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def step_datasets(args, lfr_sizes=None, ba_sizes=None):
    print("\n[main] Step 1: Dataset generation")
    lfr = lfr_sizes if lfr_sizes is not None else args.lfr_sizes
    ba  = ba_sizes  if ba_sizes  is not None else []
    argv = (
        ["--output-dir", str(args.output_dir),
         "--seed", str(args.seed),
         "--lfr-sizes"] + [str(x) for x in lfr]
    )
    if ba:
        argv += ["--ba-sizes"] + [str(x) for x in ba]
    else:
        argv += ["--skip-ba"]
    old_argv = sys.argv
    sys.argv = ["datasets.py"] + argv
    try:
        datasets_module.main()
    finally:
        sys.argv = old_argv


def step_similarity(args, pattern="*.adjlist", n_workers=1):
    adj_dir = args.output_dir / "adjlists"
    sim_files = {}
    if not adj_dir.exists():
        print(f"[main] No adjlists directory at {adj_dir}")
        return sim_files
    for adj_path in sorted(adj_dir.glob(pattern)):
        sim_path = adj_dir / f"{adj_path.stem}.sim.tsv"
        print(f"[main] Similarity: {adj_path.name} (workers={n_workers})")
        run_similarity(
            input_path=adj_path,
            output_path=sim_path,
            n_workers=n_workers,
            verbose=args.verbose,
        )
        sim_files[adj_path.stem] = sim_path
    return sim_files


def step_clustering(args, sim_files, epsilon):
    """Run clustering for all sim_files at given epsilon.

    Returns
    -------
    all_labels : {name: {node: label}}
    all_paths  : {name: output_paths dict from cluster_results}
    """
    all_labels = {}
    all_paths  = {}
    for name, sim_path in sim_files.items():
        print(f"[main] Clustering {name} (eps={epsilon}, mu={args.mu})")
        raw_labels, _, output_paths = cluster_results(
            sim_path=sim_path,
            epsilon=epsilon,
            PROJECT_ROOT=PROJECT_ROOT,
            mu=args.mu,
        )
        labels = {int(k): int(v) for k, v in raw_labels.items()}
        all_labels[name] = labels
        all_paths[name]  = output_paths
    return all_labels, all_paths


def save_best_epsilon_outputs(name, best_eps, output_paths, output_dir):
    """
    Copy cluster, filtered_adjlist, parsed_input, and classification files
    for the best epsilon into a dedicated 'best_epsilon' subfolder,
    renaming them to include 'best' in the filename.

    The per-epsilon filtered adjlists written by LPSS.py during the accuracy
    experiment loop are intentionally left untouched — this function only
    adds an additional copy tagged with 'best' for easy identification.

    Folder layout:
        <output_dir>/best_epsilon/
            lfr_1000_best_eps0.4_clusters.csv
            lfr_1000_best_eps0.4_filtered_adjlist.tsv
            lfr_1000_best_eps0.4_parsed_input.tsv
            lfr_1000_best_eps0.4_classification.tsv

    All per-epsilon filtered adjlists remain at:
        <output_dir>/filtered_adjlists/
            filtered_edge_<stem>_eps0.2.tsv
            filtered_edge_<stem>_eps0.4.tsv   ← same file as best, just not renamed
            ...
    """
    best_dir = output_dir / "best_epsilon"
    best_dir.mkdir(parents=True, exist_ok=True)

    eps_tag = f"best_eps{best_eps}"

    file_map = {
        "clusters":         f"{name}_{eps_tag}_clusters.csv",
        "filtered_adjlist": f"{name}_{eps_tag}_filtered_adjlist.tsv",
        "parsed_input":     f"{name}_{eps_tag}_parsed_input.tsv",
        "classification":   f"{name}_{eps_tag}_classification.tsv",
    }

    for key, dest_name in file_map.items():
        src = output_paths.get(key)
        if src and Path(src).exists():
            dest = best_dir / dest_name
            shutil.copy(src, dest)
            print(f"[main] Best-epsilon file saved -> {dest}")
        else:
            print(f"[main] WARNING: {key} file not found for {name} at eps={best_eps}, skipping.")

    # Also copy the ground truth labels (no epsilon suffix needed)
    gt_src = output_dir / "labels" / f"{name}.labels.tsv"
    if gt_src.exists():
        gt_dest = best_dir / f"{name}_labels.tsv"
        shutil.copy(gt_src, gt_dest)
        print(f"[main] Ground truth labels saved -> {gt_dest}")


def best_epsilon_from_results(results_file):
    if not results_file.exists():
        return None
    try:
        import csv
        from collections import defaultdict
        rows = defaultdict(list)
        with results_file.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows[float(row["epsilon"])].append(float(row["ari"]))
        if not rows:
            return None
        best = max(rows, key=lambda e: sum(rows[e]) / len(rows[e]))
        print(f"[main] Best epsilon from experiment 1: {best}")
        return best
    except Exception as exc:
        print(f"[main] Could not determine best epsilon: {exc}")
        return None


def run_accuracy_experiment(args):
    print("\n[Experiment 1] Accuracy vs Epsilon")
    print(f"  LFR sizes : {args.lfr_sizes}")
    print(f"  Epsilons  : {args.eps_list}")
    print(f"  Mu        : {args.mu}")

    results_file = SCRATCH_DIR / "results_accuracy.csv"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with results_file.open("w") as f:
        f.write("dataset,epsilon,ari,nmi\n")

    # Compute similarity once — epsilon only affects pruning step
    # Use all available CPUs for parallel similarity computation
    n_workers = multiprocessing.cpu_count()
    sim_files = step_similarity(args, pattern="*.adjlist", n_workers=n_workers)

    # Track best epsilon and its ARI + output paths per dataset
    # Structure: {name: {"best_eps": float, "best_ari": float, "best_paths": dict}}
    best_per_dataset = {}

    for eps in sorted(args.eps_list):
        print(f"\n--- eps={eps} ---")
        all_labels, all_paths = step_clustering(args, sim_files, eps)

        # all_paths[name]["filtered_adjlist"] is already written by LPSS.py
        # with eps in the filename (e.g. filtered_edge_lfr_1000.sim_eps0.4.tsv).
        # We do NOT delete or overwrite it here — all epsilon variants persist.

        for name, labels in all_labels.items():
            gt_path = args.output_dir / "labels" / f"{name}.labels.tsv"
            if not gt_path.exists():
                print(f"[main] No ground truth for {name}, skipping.")
                continue

            ari, nmi = evaluate(gt_path, labels)

            with results_file.open("a") as f:
                f.write(f"{name},{eps},{ari:.6f},{nmi:.6f}\n")

            # Track best epsilon per dataset based on ARI
            if name not in best_per_dataset or ari > best_per_dataset[name]["best_ari"]:
                best_per_dataset[name] = {
                    "best_eps":   eps,
                    "best_ari":   ari,
                    "best_paths": all_paths.get(name, {}),
                }

    # Save best-epsilon output files for each dataset.
    # This adds a 'best'-tagged COPY alongside the already-saved per-epsilon files.
    print("\n[main] Saving best-epsilon output files...")
    for name, info in best_per_dataset.items():
        best_eps   = info["best_eps"]
        best_ari   = info["best_ari"]
        best_paths = info["best_paths"]
        print(f"[main] {name}: best epsilon={best_eps} (ARI={best_ari:.4f})")
        save_best_epsilon_outputs(name, best_eps, best_paths, args.output_dir)

    print(f"\n[main] Accuracy results -> {results_file}")

    try:
        shutil.copy(results_file, PROJECT_ROOT / "results_accuracy.csv")
        print(f"[main] Copied to {PROJECT_ROOT / 'results_accuracy.csv'}")
    except Exception as e:
        print(f"[main] Could not copy results back: {e}")


def run_pipeline_with_workers(adj_path, sim_path, eps, mu, n_workers, verbose):
    """
    Run similarity (parallel) + clustering.

    Each worker reads its own chunk of the adjlist file directly —
    no pickling of the large adjacency list across processes.

    Returns elapsed wall-clock time in seconds.
    """
    t_start = time.perf_counter()

    # Similarity stage — parallel file-based chunking across n_workers
    run_similarity(
        input_path=adj_path,
        output_path=sim_path,
        n_workers=n_workers,
        verbose=verbose,
    )

    # Clustering stage — runs after similarity
    cluster_results(
        sim_path=sim_path,
        epsilon=eps,
        PROJECT_ROOT=PROJECT_ROOT,
        mu=mu,
    )

    return time.perf_counter() - t_start


def run_runtime_experiment(args):
    print("\n[Experiment 2] Runtime vs Machines")
    print(f"  BA sizes  : {args.ba_sizes}")
    print(f"  Machines  : {args.machines}")
    print(f"  Mu        : {args.mu}")

    # Use best epsilon from experiment 1 if available
    eps = best_epsilon_from_results(SCRATCH_DIR / "results_accuracy.csv")
    if eps is None:
        eps = args.epsilon
        print(f"[main] Using epsilon={eps} (from --epsilon flag)")

    results_file = SCRATCH_DIR / "results_runtime.csv"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with results_file.open("w") as f:
        f.write("dataset,n_nodes,n_machines,epsilon,time_s,speedup\n")

    for n in sorted(args.ba_sizes):
        name = f"ba_{n}"
        print(f"\n--- BA n={n:,} ---")

        if not args.skip_datasets:
            step_datasets(args, lfr_sizes=[], ba_sizes=[n])

        adj_path = args.output_dir / "adjlists" / f"{name}.adjlist"
        if not adj_path.exists():
            print(f"[main] {adj_path} not found, skipping.")
            continue

        sim_path = args.output_dir / "adjlists" / f"{name}.sim.tsv"

        # baseline_time set at smallest machine count (4 CPUs → 1×)
        baseline_time = None

        for n_machines in sorted(args.machines):
            actual_workers = min(n_machines, multiprocessing.cpu_count())
            print(f"  Machines={n_machines} (actual CPU workers={actual_workers})")

            elapsed = run_pipeline_with_workers(
                adj_path=adj_path,
                sim_path=sim_path,
                eps=eps,
                mu=args.mu,
                n_workers=actual_workers,
                verbose=args.verbose,
            )

            # First (smallest) machine count = baseline (4 CPUs → 1×)
            if baseline_time is None:
                baseline_time = elapsed

            speedup = baseline_time / elapsed if elapsed > 0 else 1.0

            print(f"  -> {elapsed:.2f}s  (speedup vs 4-CPU baseline: {speedup:.2f}x)")

            with results_file.open("a") as f:
                f.write(f"{name},{n},{n_machines},{eps},{elapsed:.4f},{speedup:.4f}\n")

    print(f"\n[main] Runtime results -> {results_file}")

    try:
        shutil.copy(results_file, PROJECT_ROOT / "results_runtime.csv")
        print(f"[main] Copied to {PROJECT_ROOT / 'results_runtime.csv'}")
    except Exception as e:
        print(f"[main] Could not copy results back: {e}")


def run_default_pipeline(args):
    print("\n[main] Running full pipeline")
    sim_files = step_similarity(args, pattern="lfr_*.adjlist")
    all_labels, _ = step_clustering(args, sim_files, args.epsilon)
    for name, labels in all_labels.items():
        gt_path = args.output_dir / "labels" / f"{name}.labels.tsv"
        if gt_path.exists():
            evaluate(gt_path, labels)
        else:
            print(f"[main] No ground truth for {name}.")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[main] PSCAN started")
    print(f"[main] Project root : {PROJECT_ROOT}")
    print(f"[main] Output dir   : {args.output_dir}")
    print(f"[main] Mu           : {args.mu}")

    t0 = time.perf_counter()

    if not args.skip_datasets:
        if args.experiment != "runtime":
            step_datasets(args)

    if args.experiment == "accuracy":
        run_accuracy_experiment(args)
    elif args.experiment == "runtime":
        run_runtime_experiment(args)
    else:
        run_default_pipeline(args)

    print(f"\n[main] Finished in {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()