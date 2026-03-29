#!/bin/bash
#SBATCH --job-name=pscan_runtime
#SBATCH --account=def-a2nyi4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/nidita/logs/%x_%j.out
#SBATCH --error=/scratch/nidita/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=niditaroy@uvic.ca

echo "=========================================="
echo "Job started:  $(date)"
echo "Node:         $SLURMD_NODENAME"
echo "Job ID:       $SLURM_JOB_ID"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "=========================================="

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load python/3.11
module load scipy-stack

source /project/def-a2nyi4/nidita/pscan/venv/bin/activate

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR="/project/def-a2nyi4/nidita/pscan/working"
OUTPUT_DIR="/scratch/nidita/pscan_output"

# Create all output folders on scratch
mkdir -p /scratch/nidita/logs
mkdir -p "$OUTPUT_DIR/adjlists"
mkdir -p "$OUTPUT_DIR/labels"
mkdir -p "$OUTPUT_DIR/clusters"
mkdir -p "$OUTPUT_DIR/classifications"
mkdir -p "$OUTPUT_DIR/filtered_adjlists"
mkdir -p "$OUTPUT_DIR/parsed_input"
mkdir -p "$OUTPUT_DIR/metadata"

cd "$PROJECT_DIR"

# ── Run Experiment 1 first if results not available ───────────────────────────
if [ ! -f "$OUTPUT_DIR/results_accuracy.csv" ]; then
    echo "results_accuracy.csv not found — running Experiment 1 first..."
    python run/main.py \
        --experiment accuracy \
        --output-dir "$OUTPUT_DIR" \
        --lfr-sizes 500 1000 2000 \
        --eps-list 0.2 0.4 0.6 0.8 1.0 \
        --verbose
    echo "Experiment 1 done."
    echo ""
fi

# ── Run Experiment 2 ──────────────────────────────────────────────────────────
echo "Starting Experiment 2: Runtime vs Machines..."

python run/main.py \
    --experiment runtime \
    --output-dir "$OUTPUT_DIR" \
    --ba-sizes 100000 200000 300000 400000 \
    --machines 4 8 15 \
    --skip-datasets \
    --verbose

EXIT_CODE=$?

echo ""
echo "Exit code: $EXIT_CODE"
echo "Experiment 2 finished: $(date)"

# ── Copy results back to working folder ───────────────────────────────────────
cp "$OUTPUT_DIR/results_accuracy.csv" "$PROJECT_DIR/results_accuracy.csv" 2>/dev/null && \
    echo "results_accuracy.csv copied" || \
    echo "No results_accuracy.csv to copy"

cp "$OUTPUT_DIR/results_runtime.csv" "$PROJECT_DIR/results_runtime.csv" 2>/dev/null && \
    echo "results_runtime.csv copied" || \
    echo "No results_runtime.csv to copy"

echo "=========================================="