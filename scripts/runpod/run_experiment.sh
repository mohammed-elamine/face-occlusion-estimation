#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Run one face occlusion experiment on RunPod
# ============================================================
#
# Usage:
#   bash /workspace/scripts/run_experiment.sh CONFIG
#
# Example:
#   bash /workspace/scripts/run_experiment.sh configs/00_baseline.yaml
#
# ============================================================

REPO_DIR="/workspace/repos/face-occlusion-estimation"
CONFIG="${1:-configs/baseline.yaml}"

# ============================================================
# Persistent uv cache/environment
# ============================================================

export UV_LINK_MODE="copy"
export UV_CACHE_DIR="/workspace/cache/uv"
export UV_PROJECT_ENVIRONMENT="/workspace/venvs/face-occlusion-estimation"

mkdir -p "${UV_CACHE_DIR}"
mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"

cd "${REPO_DIR}"

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config file not found:"
    echo "  ${CONFIG}"
    exit 1
fi

RUN_NAME="$(basename "${CONFIG}" .yaml)"
RUN_LOG_DIR="outputs/runpod_logs/${RUN_NAME}"
mkdir -p "${RUN_LOG_DIR}"

LOG_PATH="${RUN_LOG_DIR}/train_$(date -u +"%Y%m%d_%H%M%S").log"

# Save everything printed from this point into both terminal and log file.
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "============================================================"
echo "Running RunPod experiment"
echo "============================================================"
echo "Repo dir:   ${REPO_DIR}"
echo "Config:     ${CONFIG}"
echo "Run name:   ${RUN_NAME}"
echo "Log path:   ${LOG_PATH}"
echo "Started at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "============================================================"
echo

echo "==> uv environment configuration:"
echo "UV_LINK_MODE=${UV_LINK_MODE}"
echo "UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"

echo
echo "==> Git state:"
echo "Commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
echo "Branch: $(git branch --show-current 2>/dev/null || echo 'unknown')"

echo
echo "Working tree status:"
git status --short || true

echo
echo "==> GPU state:"
nvidia-smi || true

echo
echo "==> Python / uv state:"
uv --version || true
uv run python --version

echo
echo "==> Checking PyTorch CUDA:"

uv run python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("ERROR: CUDA is not available. Do not run training.")
PY

echo
echo "==> Project storage check:"

echo
echo "data/:"
ls -lah data || true

echo
echo "data/raw:"
ls -lah data/raw || true
echo "data/raw points to: $(readlink -f data/raw 2>/dev/null || echo 'not found')"
du -sh data/raw 2>/dev/null || true

echo
echo "outputs/:"
ls -lah outputs || true

echo
echo "outputs/experiments:"
ls -lah outputs/experiments || true
echo "outputs/experiments points to: $(readlink -f outputs/experiments 2>/dev/null || echo 'not found')"

echo
echo "outputs/runpod_logs:"
ls -lah outputs/runpod_logs || true
echo "outputs/runpod_logs points to: $(readlink -f outputs/runpod_logs 2>/dev/null || echo 'not found')"

echo
echo "outputs/splits:"
ls -lah outputs/splits || true

echo
echo "==> Starting training..."

# Make the repo root importable.
# This is needed because training is launched as a module:
#   python -m scripts.training.train
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

TRAIN_CMD=(uv run python -m scripts.training.train --config "${CONFIG}")

echo "PYTHONPATH=${PYTHONPATH}"
echo
echo "Command:"
printf '  %q' "${TRAIN_CMD[@]}"
echo
echo

"${TRAIN_CMD[@]}"

echo
echo "============================================================"
echo "Experiment finished successfully"
echo "============================================================"
echo "Finished at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "Log path: ${LOG_PATH}"
echo "============================================================"
