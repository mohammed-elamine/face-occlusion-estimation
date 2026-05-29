#!/bin/bash
#SBATCH --job-name=face_occ_baseline
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
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

# Make libbz2 visible to Python's bz2 module.
export LD_LIBRARY_PATH="/projects/share/apps/miniconda3/25.5.1/lib:${LD_LIBRARY_PATH:-}"

# Return to the directory where sbatch was launched, usually the repository root.
cd "$SLURM_SUBMIT_DIR"

echo "Working directory: $(pwd)"

# Check that the virtual environment was created with `uv sync`.
if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: .venv/bin/activate not found."
    echo "Create the environment once from the repository root with:"
    echo "  module purge"
    echo "  module load python/3.11 cuda/12.4 uv"
    echo "  uv sync --frozen --extra wandb"
    exit 1
fi

source .venv/bin/activate

mkdir -p outputs/logs outputs/checkpoints outputs/predictions outputs/reports

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
python scripts/train.py --config configs/baseline.yaml

echo "Job finished at $(date)"
