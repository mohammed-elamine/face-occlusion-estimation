#!/bin/bash
# Prepare the cluster environment before launching Slurm jobs.
#
# Run this script from the repository root after pulling a new version of the code.
# It loads the required cluster modules, creates or updates the local `.venv`
# environment using `uv`, installs dependencies from `pyproject.toml` and `uv.lock`,
# and performs a few quick sanity checks.

set -e

echo "Setting up cluster environment..."

# Make sure the script is executed from the repository root.
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found."
    echo "Please run this script from the repository root."
    exit 1
fi

module purge
module load python/3.11
module load cuda/12.4
module load uv
hash -r

# Make libbz2 visible to Python's bz2 module.
export LD_LIBRARY_PATH="/projects/share/apps/miniconda3/25.5.1/lib:${LD_LIBRARY_PATH:-}"

echo "Syncing Python environment with uv..."
# Use the lock file and Python 3.11 to match the cluster module stack.
uv sync --frozen --extra wandb --python python3.11

source .venv/bin/activate

echo "Running sanity checks..."

python -c "import bz2; print('bz2 OK')"
python -c "import torchvision; print('torchvision OK')"

python - <<'PY'
import sys
import torch
import face_occlusion

print("Python:", sys.version.split()[0])
print("Project package: OK")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Note: CUDA may be unavailable on the login node. It should be checked again inside the Slurm job.")
PY

mkdir -p outputs/experiments outputs/slurm_logs outputs/splits outputs/reports

echo "Environment setup completed successfully."
echo "You can now submit the training job with:"
echo "  sbatch jobs/train.slurm"
