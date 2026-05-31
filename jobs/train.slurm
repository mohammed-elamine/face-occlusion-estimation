#!/bin/bash
#SBATCH --job-name=face_occ_train
#SBATCH --output=outputs/slurm_logs/%x_%j.out
#SBATCH --error=outputs/slurm_logs/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -e

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

module purge
module load python/3.11
module load cuda/12.4
module load uv
hash -r

# Required because the cluster Python module needs libbz2 at runtime.
export LD_LIBRARY_PATH="/projects/share/apps/miniconda3/25.5.1/lib:${LD_LIBRARY_PATH:-}"

# Return to the directory where sbatch was launched, usually the repository root.
cd "$SLURM_SUBMIT_DIR"

echo "Working directory: $(pwd)"

# Default to the starter config, but allow CONFIG_PATH=... sbatch jobs/train.slurm.
CONFIG_PATH="${CONFIG_PATH:-configs/baseline.yaml}"
echo "Config path: $CONFIG_PATH"

mkdir -p outputs/slurm_logs

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: .venv/bin/activate not found."
    echo "Create the environment once from the repository root with:"
    echo "  bash scripts/setup_cluster_env.sh"
    exit 1
fi

source .venv/bin/activate

echo "Python: $(which python)"
python --version

echo "Checking CUDA availability..."
python - <<'PY'
import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA is not available. Training may run on CPU.")
PY

echo "Checking Python native libraries..."
python - <<'PY'
import bz2
import torchvision

print("bz2 OK")
print("torchvision OK")
PY

echo "Starting training..."
python scripts/train.py --config "$CONFIG_PATH"

echo "Job finished at $(date)"
