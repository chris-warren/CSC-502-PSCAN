#!/bin/bash
#SBATCH --job-name=pscan_accuracy
#SBATCH --account=def-a2nyi4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
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

echo "Starting Experiment 1: Accuracy vs Epsilon..."
echo "Mu: $MU"

python run/main.py \
    --experiment accuracy \
    --output-dir "$OUTPUT_DIR" \
    --lfr-sizes 500 1000 2000 \
    --eps-list 0.2 0.4 0.6 0.8 1.0 \
    --mu "$MU" \
    --verbose

EXIT_CODE=$?
echo "Exit code: $EXIT_CODE"
echo "Experiment 1 finished: $(date)"

cp "$OUTPUT_DIR/results_accuracy.csv" "$PROJECT_DIR/results_accuracy.csv" 2>/dev/null && \
    echo "results_accuracy.csv copied" || \
    echo "No results_accuracy.csv to copy"

echo "=========================================="