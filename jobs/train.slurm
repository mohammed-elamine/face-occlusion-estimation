#!/bin/bash
#SBATCH --job-name=face_occ_train
# `--output` / `--error` are evaluated by SLURM *before* the script runs and
# cannot read $CONFIG_PATH. We therefore keep a stable top-level location
# here and tee stdout/stderr into a run-specific subdirectory below.
#SBATCH --output=outputs/slurm_logs/_raw/%x_%j.out
#SBATCH --error=outputs/slurm_logs/_raw/%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -e

# Return to the directory where sbatch was launched, usually the repository root.
cd "$SLURM_SUBMIT_DIR"

# Default config and run-name resolution. RUN_NAME can be passed explicitly;
# otherwise we derive it from the config basename so `CONFIG_PATH=... sbatch`
# already produces a meaningful, per-experiment log directory.
CONFIG_PATH="${CONFIG_PATH:-configs/baseline.yaml}"
RUN_NAME="${RUN_NAME:-$(basename "$CONFIG_PATH" .yaml)}"
RUN_LOG_DIR="outputs/slurm_logs/${RUN_NAME}"

# The raw SBATCH paths must exist before SLURM writes to them.
mkdir -p outputs/slurm_logs/_raw "$RUN_LOG_DIR"

# Mirror everything from here on into the run-specific log files. The raw
# `_raw/*.out|err` files still get written by SLURM directly, so nothing is
# lost; the run-named copy is the convenient one for humans.
exec > >(tee -a "${RUN_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out") \
     2> >(tee -a "${RUN_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err" >&2)

# ── Run metadata ────────────────────────────────────────────────────────────
echo "Job name:          ${SLURM_JOB_NAME}"
echo "Job ID:            ${SLURM_JOB_ID}"
echo "Hostname:          $(hostname)"
echo "Date:              $(date)"
echo "Working directory: $(pwd)"
echo "CONFIG_PATH:       ${CONFIG_PATH}"
echo "RUN_NAME:          ${RUN_NAME}"
echo "RUN_LOG_DIR:       ${RUN_LOG_DIR}"

if command -v git >/dev/null 2>&1 && [ -d .git ]; then
    GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    echo "Git branch:        ${GIT_BRANCH}"
    echo "Git commit:        ${GIT_COMMIT}"
else
    echo "WARNING: git information unavailable."
fi

module purge
module load python/3.11
module load cuda/12.4
module load uv
hash -r

# Required because the cluster Python module needs libbz2 at runtime.
export LD_LIBRARY_PATH="/projects/share/apps/miniconda3/25.5.1/lib:${LD_LIBRARY_PATH:-}"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: .venv/bin/activate not found."
    echo "Create the environment once from the repository root with:"
    echo "  bash scripts/setup/setup_cluster_env.sh"
    exit 1
fi

source .venv/bin/activate

echo "Python path:       $(which python)"
echo "Python version:    $(python --version 2>&1)"

echo "Checking CUDA availability..."
python - <<'PY'
import torch

print("Torch version:  ", torch.__version__)
print("CUDA available: ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:   ", torch.version.cuda)
    print("GPU name:       ", torch.cuda.get_device_name(0))
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

TRAIN_CMD="python -m scripts.training.train --config ${CONFIG_PATH}"
echo "Starting training: ${TRAIN_CMD}"
${TRAIN_CMD}

echo "Job finished at $(date)"
