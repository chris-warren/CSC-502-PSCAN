# PSCAN: A Parallel Structural Clustering Algorithm for Big Networks in MapReduce
# PSCAN — Parallel Structural Clustering Algorithm for Networks

A MapReduce-based implementation of the PSCAN graph clustering algorithm, evaluated on LFR benchmark graphs and real-world datasets (Orkut).

---

## Project Structure

```
CSC-502-PSCAN/working/
├── data/
│   └── datasets.py              # LFR and BA graph generation
├── similarity/
│   ├── similarity_mapper.py     # PCSS mapper (parallel file-based chunking)
│   ├── similarity_reducer.py    # PCSS reducer
│   └── similarity_main.py       # Similarity pipeline orchestrator
├── clustering/
│   ├── LPSS.py                  # Edge pruning by epsilon
│   ├── LPSS_main.py             # Clustering pipeline orchestrator
│   ├── LPSS_pyspark.py          # Label propagation (LPCC) via PySpark
│   └── hub_outlier.py           # Core / Hub / Outlier classification
├── evaluation/
│   ├── evaluate.py              # ARI and NMI evaluation
│   ├── plot_results.py          # Accuracy and runtime plots
│   └── visualization.py        # Graph visualization with cluster colors
└── run/
    ├── main.py                  # Experiment entry point
    ├── run_accuracy.sh          # SLURM job: Experiment 1 (accuracy vs epsilon)
    └── run_runtime.sh           # SLURM job: Experiment 2 (runtime vs machines)
```

---

## Setup

### 1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_1.txt
```

### 2. On Compute Canada (Cedar / Graham / Narval)

```bash
module purge
module load python/3.11
module load scipy-stack
source /project/def-a2nyi4/$USER/pscan/venv/bin/activate
```

---

## How to Run

### Experiment 1 — Accuracy vs Epsilon (LFR graphs)

Runs PSCAN across multiple epsilon values and evaluates ARI and NMI against ground truth labels.

```bash
sbatch run/run_accuracy.sh
```

Or manually:

```bash
python run/main.py \
    --experiment accuracy \
    --output-dir /scratch/$USER/pscan_output \
    --lfr-sizes 5000 10000 20000 40000 80000 160000 \
    --eps-list 0.2 0.4 0.6 0.8 1.0 \
    --mu 5 \
    --skip-datasets \
    --verbose
```

**Output:**
- `/scratch/$USER/pscan_output/results_accuracy.csv` — ARI and NMI per (dataset, epsilon)
- `/scratch/$USER/pscan_output/filtered_adjlists/` — one filtered adjlist per (dataset × epsilon)
- `/scratch/$USER/pscan_output/best_epsilon/` — best epsilon outputs per dataset

---

### Experiment 2 — Runtime vs Number of Workers (BA graphs)

Runs PSCAN with 4, 8, and 15 parallel workers on Barabasi-Albert graphs to measure speedup.

```bash
sbatch run/run_runtime.sh
```

Or manually:

```bash
python run/main.py \
    --experiment runtime \
    --output-dir /scratch/$USER/pscan_output \
    --ba-sizes 1000000 2000000 3000000 4000000 \
    --machines 4 8 15 \
    --mu 5 \
    --epsilon 0.6 \
    --skip-datasets \
    --verbose
```

**Output:**
- `/scratch/$USER/pscan_output/results_runtime.csv` — wall-clock time and speedup per (dataset, n_workers)

---

### Dataset Generation (if needed)

```bash
python data/datasets.py \
    --output-dir /scratch/$USER/pscan_output \
    --lfr-sizes 5000 10000 20000 \
    --ba-sizes 1000000 2000000 \
    --seed 42
```


## Output Directory Structure

```
/scratch/$USER/pscan_output/
├── adjlists/               # Input graphs (.adjlist) and similarity scores (.sim.tsv)
├── labels/                 # Ground truth labels for LFR graphs
├── filtered_adjlists/      # Pruned graphs per (dataset × epsilon)
├── parsed_input/           # LPCC input files per (dataset × epsilon)
├── clusters/               # Cluster assignments per (dataset × epsilon)
├── classifications/        # Core / Hub / Outlier per (dataset × epsilon)
├── best_epsilon/           # Best epsilon outputs per dataset
├── metadata/               # Graph metadata JSON files
├── manifest.json           # Dataset manifest
├── results_accuracy.csv    # Experiment 1 results
└── results_runtime.csv     # Experiment 2 results
```

---

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--epsilon` | Similarity threshold for edge pruning | `0.5` |
| `--eps-list` | List of epsilons for accuracy experiment | `0.2 0.4 0.6 0.8 1.0` |
| `--mu` | Min neighbors in pruned graph to be a core node | `1` |
| `--machines` | Worker counts for runtime experiment | `4 8 15` |
| `--seed` | Random seed for graph generation | `42` |
| `--skip-datasets` | Skip dataset generation (use existing files) | `False` |
| `--verbose` | Print progress and timing | `False` |
