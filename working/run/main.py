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
    parser.add_argument("--ba-sizes", nargs="*", type=int, default=[100000, 200000, 300000, 400000])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--eps-list", nargs="*", type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--machines", nargs="*", type=int, default=[4, 8, 15])
    parser.add_argument("--experiment", choices=["accuracy", "runtime"], default=None)
    parser.add_argument("--skip-datasets", action="store_true")
    parser.add_argument("--skip-similarity", action="store_true")
    parser.add_argument("--skip-clustering", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--paper-scales", action="store_true")
    parser.add_argument("--tau1", type=float, default=3.0)
    parser.add_argument("--tau2", type=float, default=1.5)
    parser.add_argument("--mu", type=float, default=0.1)
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


def step_similarity(args, pattern="*.adjlist"):
    adj_dir = args.output_dir / "adjlists"
    sim_files = {}
    if not adj_dir.exists():
        print(f"[main] No adjlists directory at {adj_dir}")
        return sim_files
    for adj_path in sorted(adj_dir.glob(pattern)):
        sim_path = adj_dir / f"{adj_path.stem}.sim.tsv"
        print(f"[main] Similarity: {adj_path.name}")
        run_similarity(input_path=adj_path, output_path=sim_path, verbose=args.verbose)
        sim_files[adj_path.stem] = sim_path
    return sim_files


def step_clustering(args, sim_files, epsilon):
    all_labels = {}
    for name, sim_path in sim_files.items():
        print(f"[main] Clustering {name} (eps={epsilon})")
        raw_labels = cluster_results(
            sim_path=sim_path,
            epsilon=epsilon,
            PROJECT_ROOT=PROJECT_ROOT,
        )
        labels = {int(k): int(v) for k, v in raw_labels.items()}
        all_labels[name] = labels
    return all_labels


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

    results_file = SCRATCH_DIR / "results_accuracy.csv"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with results_file.open("w") as f:
        f.write("dataset,epsilon,ari,nmi\n")

    # Compute similarity once — epsilon only affects pruning step
    sim_files = step_similarity(args, pattern="lfr_*.adjlist")

    for eps in sorted(args.eps_list):
        print(f"\n--- eps={eps} ---")
        all_labels = step_clustering(args, sim_files, eps)
        for name, labels in all_labels.items():
            gt_path = args.output_dir / "labels" / f"{name}.labels.tsv"
            if not gt_path.exists():
                print(f"[main] No ground truth for {name}, skipping.")
                continue
            ari, nmi = evaluate(gt_path, labels)
            with results_file.open("a") as f:
                f.write(f"{name},{eps},{ari:.6f},{nmi:.6f}\n")

    print(f"\n[main] Accuracy results -> {results_file}")

    # Copy back to working folder (read-only /project is fine for small CSVs)
    try:
        shutil.copy(results_file, PROJECT_ROOT / "results_accuracy.csv")
        print(f"[main] Copied to {PROJECT_ROOT / 'results_accuracy.csv'}")
    except Exception as e:
        print(f"[main] Could not copy results back: {e}")


def run_runtime_experiment(args):
    print("\n[Experiment 2] Runtime vs Machines")
    print(f"  BA sizes  : {args.ba_sizes}")
    print(f"  Machines  : {args.machines}")

    # Use best epsilon from experiment 1 if available
    eps = best_epsilon_from_results(SCRATCH_DIR / "results_accuracy.csv")
    if eps is None:
        eps = args.epsilon
        print(f"[main] Using epsilon={eps} (from --epsilon flag)")

    results_file = SCRATCH_DIR / "results_runtime.csv"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with results_file.open("w") as f:
        f.write("dataset,n_nodes,n_machines,epsilon,time_s\n")

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

        for n_machines in sorted(args.machines):
            actual_workers = min(n_machines, multiprocessing.cpu_count())
            print(f"  Machines={n_machines} (actual CPU workers={actual_workers})")

            t_start = time.perf_counter()
            run_similarity(input_path=adj_path, output_path=sim_path, verbose=args.verbose)
            step_clustering(args, {name: sim_path}, eps)
            elapsed = time.perf_counter() - t_start

            print(f"  -> {elapsed:.2f}s")
            with results_file.open("a") as f:
                f.write(f"{name},{n},{n_machines},{eps},{elapsed:.4f}\n")

    print(f"\n[main] Runtime results -> {results_file}")

    # Copy back to working folder
    try:
        shutil.copy(results_file, PROJECT_ROOT / "results_runtime.csv")
        print(f"[main] Copied to {PROJECT_ROOT / 'results_runtime.csv'}")
    except Exception as e:
        print(f"[main] Could not copy results back: {e}")


def run_default_pipeline(args):
    print("\n[main] Running full pipeline")
    sim_files = step_similarity(args, pattern="lfr_*.adjlist")
    all_labels = step_clustering(args, sim_files, args.epsilon)
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