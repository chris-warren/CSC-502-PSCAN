#!/bin/bash
#SBATCH --job-name=pscan_pipeline
#SBATCH --account=def-a2nyi4
#SBATCH --partition=compute_full_node
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/nidita/logs/%x_%j.out
#SBATCH --error=/scratch/nidita/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=niditaroy@uvic.ca

# ─────────────────────────────────────────────────────────────────────────────
# PSCAN Pipeline Job Script — Trillium Cluster
#
# Folder layout:
#   /project/def-a2nyi4/nidita/pscan/
#       data/           datasets.py
#       similarity/     similarity_mapper.py
#                       similarity_reducer.py
#                       similarity_main.py
#       run/            main.py
#                       job.sh
#       venv/
#
# Logs go to /scratch/nidita/logs/ (writable on compute nodes)
# Output goes to /scratch/nidita/pscan_output/ (fast large storage)
#
# Submit:
#   cd /project/def-a2nyi4/nidita/pscan
#   sbatch run/job.sh
# ─────────────────────────────────────────────────────────────────────────────

echo "=========================================="
echo "Job started:  $(date)"
echo "Node:         $SLURMD_NODENAME"
echo "Job ID:       $SLURM_JOB_ID"
echo "=========================================="

# ── Environment setup ─────────────────────────────────────────────────────────
module purge
module load python/3.11
module load scipy-stack

source /project/def-a2nyi4/nidita/pscan/venv/bin/activate

echo "Python:  $(which python)"
echo "Version: $(python --version)"

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR="/project/def-a2nyi4/nidita/pscan"
RUN_DIR="$PROJECT_DIR/run"
OUTPUT_DIR="/scratch/nidita/pscan_output"
LOG_DIR="/scratch/nidita/logs"

echo ""
echo "Project root: $PROJECT_DIR"
echo "Output dir:   $OUTPUT_DIR"
echo ""

# Create required directories (scratch is writable on compute nodes)
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR/adjlists"
mkdir -p "$OUTPUT_DIR/labels"
mkdir -p "$OUTPUT_DIR/metadata"

# ── Run the pipeline ──────────────────────────────────────────────────────────
cd "$RUN_DIR"

echo "── Step 1+2: Generating datasets + computing similarities ──"
python main.py \
    --output-dir "$OUTPUT_DIR" \
    --paper-scales \
    --seed 42 \
    --verbose \
    --skip-clustering \
    --skip-evaluation

PIPELINE_EXIT=$?

echo ""
echo "Exit code:    $PIPELINE_EXIT"
echo "Job finished: $(date)"
echo "=========================================="

# ── Summary of outputs ────────────────────────────────────────────────────────
echo ""
echo "Generated similarity files:"
ls -lh "$OUTPUT_DIR/adjlists/"*.sim.tsv 2>/dev/null || echo "  No .sim.tsv files found."

echo ""
echo "Generated adjacency lists:"
ls -lh "$OUTPUT_DIR/adjlists/"*.adjlist 2>/dev/null || echo "  No .adjlist files found."

echo ""
echo "Disk usage on scratch:"
du -sh "$OUTPUT_DIR"

# ── Copy key results back to $PROJECT for safekeeping ─────────────────────────
echo ""
echo "Copying results to project for safekeeping ..."
mkdir -p "$PROJECT_DIR/data/output/adjlists"
mkdir -p "$PROJECT_DIR/data/output/labels"

cp "$OUTPUT_DIR/adjlists/"*.sim.tsv "$PROJECT_DIR/data/output/adjlists/" 2>/dev/null && \
    echo "  ✓ .sim.tsv copied to project" || \
    echo "  ✗ No .sim.tsv files to copy"

cp "$OUTPUT_DIR/labels/"*.tsv "$PROJECT_DIR/data/output/labels/" 2>/dev/null && \
    echo "  ✓ label files copied to project" || \
    echo "  ✗ No label files to copy"

cp "$OUTPUT_DIR/manifest.json" "$PROJECT_DIR/data/output/" 2>/dev/null && \
    echo "  ✓ manifest.json copied to project"

echo ""
echo "Done. Results are in:"
echo "  Large files : $OUTPUT_DIR"
echo "  Sim + labels: $PROJECT_DIR/data/output"