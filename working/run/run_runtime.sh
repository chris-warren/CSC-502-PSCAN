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
echo "=========================================="

module purge
module load python/3.11
module load scipy-stack
source /project/def-a2nyi4/nidita/pscan/venv/bin/activate

PROJECT_DIR="/project/def-a2nyi4/nidita/pscan/working"
OUTPUT_DIR="/scratch/nidita/pscan_output"
MU=5

mkdir -p /scratch/nidita/logs
mkdir -p "$OUTPUT_DIR/adjlists"
mkdir -p "$OUTPUT_DIR/labels"
mkdir -p "$OUTPUT_DIR/clusters"
mkdir -p "$OUTPUT_DIR/filtered_adjlists"
mkdir -p "$OUTPUT_DIR/parsed_input"
mkdir -p "$OUTPUT_DIR/metadata"

cd "$PROJECT_DIR"

echo "Starting Experiment 2: Runtime vs Machines..."
echo "BA sizes: 1M 2M 3M 4M"
echo "Machines: 4 8 15 (speedup relative to 4-CPU baseline)"
echo "Mu: $MU"

python run/main.py \
    --experiment runtime \
    --output-dir "$OUTPUT_DIR" \
    --ba-sizes 1000000 2000000 3000000 4000000 \
    --machines 4 8 15 \
    --mu "$MU" \
    --verbose

EXIT_CODE=$?
echo "Exit code: $EXIT_CODE"
echo "Experiment 2 finished: $(date)"

cp "$OUTPUT_DIR/results_runtime.csv" "$PROJECT_DIR/results_runtime.csv" 2>/dev/null && \
    echo "results_runtime.csv copied" || \
    echo "No results_runtime.csv to copy"

echo "=========================================="