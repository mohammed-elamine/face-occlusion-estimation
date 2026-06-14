#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# RunPod project setup for face-occlusion-estimation
# ============================================================
#
# This script:
# 1. Bootstraps GitHub SSH access using /workspace/scripts/bootstrap_runtime_ssh.sh.
# 2. Clones or updates the private GitHub repo.
# 3. Keeps repo-level data/ and outputs/ folders intact.
# 4. Links only remote/persistent folders:
#       data/raw
#       outputs/experiments
#       outputs/runpod_logs
# 5. Preserves outputs/splits as a normal repo folder, because training reads it.
# 6. Installs useful system packages.
# 7. Installs dependencies with uv.
# 8. Checks GPU and PyTorch CUDA availability.
#
# Usage:
#   bash /workspace/scripts/setup_pod.sh
#
# ============================================================

REPO_PARENT="/workspace/repos"
REPO_DIR="/workspace/repos/face-occlusion-estimation"
REPO_URL="git@github-face-occlusion:mohammed-elamine/face-occlusion-estimation.git"

DATA_ROOT="/workspace/datasets/face-occlusion"
OUTPUT_ROOT="/workspace/outputs/face-occlusion-estimation"

SCRIPTS_DIR="/workspace/scripts"
BOOTSTRAP_SSH_SCRIPT="${SCRIPTS_DIR}/bootstrap_runtime_ssh.sh"

# ============================================================
# Helper: safely create or update a symlink
# ============================================================
#
# Behavior:
# - If link_path does not exist:
#       create symlink.
#
# - If link_path is already a symlink:
#       update it only if it points to the wrong target.
#
# - If link_path exists as a real file or directory:
#       do NOT delete it.
#       do NOT move it.
#       do NOT overwrite it.
#       just warn and skip.
#
# ============================================================

safe_link_path() {
    local target="$1"
    local link_path="$2"

    echo "==> Setting up ${link_path} -> ${target}"

    mkdir -p "${target}"
    mkdir -p "$(dirname "${link_path}")"

    if [ -L "${link_path}" ]; then
        local current_target
        current_target="$(readlink -f "${link_path}")"

        local expected_target
        expected_target="$(readlink -f "${target}")"

        if [ "${current_target}" = "${expected_target}" ]; then
            echo "    OK: ${link_path} already points to ${target}"
            return 0
        fi

        echo "    ${link_path} is a symlink but points to:"
        echo "      ${current_target}"
        echo "    Updating it to:"
        echo "      ${expected_target}"

        unlink "${link_path}"
        ln -s "${target}" "${link_path}"
        return 0
    fi

    if [ -e "${link_path}" ]; then
        echo "    SKIP: ${link_path} already exists and is not a symlink."
        echo "    I will not remove, move, or overwrite it."
        echo "    If this folder should be persistent, inspect it and convert it manually."
        return 0
    fi

    ln -s "${target}" "${link_path}"
    echo "    Created: ${link_path} -> ${target}"
}

# ============================================================
# 0. Check scripts directory and bootstrap SSH script
# ============================================================

echo "==> Checking RunPod scripts directory..."

if [ ! -d "${SCRIPTS_DIR}" ]; then
    echo "ERROR: Scripts directory not found:"
    echo "  ${SCRIPTS_DIR}"
    echo
    echo "Expected helper scripts to be stored in /workspace/scripts/"
    exit 1
fi

if [ ! -f "${BOOTSTRAP_SSH_SCRIPT}" ]; then
    echo "ERROR: SSH bootstrap script not found:"
    echo "  ${BOOTSTRAP_SSH_SCRIPT}"
    exit 1
fi

# ============================================================
# 1. Clone or update repository
# ============================================================

echo
echo "==> Bootstrapping SSH..."
echo "---------------------------------------------------"
bash "${BOOTSTRAP_SSH_SCRIPT}"
echo "---------------------------------------------------"

echo
echo "==> Checking repository..."

if [ ! -d "${REPO_DIR}/.git" ]; then
    echo "==> Repository not found. Cloning..."

    echo
    echo "==> Creating repo parent directory..."
    mkdir -p "${REPO_PARENT}"

    cd "${REPO_PARENT}"
    git clone "${REPO_URL}"

    echo
    echo "==> Repository ready at:"
    echo "  ${REPO_DIR}"
else
    echo "==> Repository already exists."

    cd "${REPO_DIR}"
    echo "==> Pulling latest changes..."
    git pull
fi

cd "${REPO_DIR}"

echo
echo "==> Git status:"
git status --short || true

# ============================================================
# 2. Create persistent directories
# ============================================================

echo
echo "==> Creating persistent directories..."

# Dataset ZIP/extracted dataset lives here through data/raw.
mkdir -p "${DATA_ROOT}/raw"

# Training runs and RunPod logs live here.
mkdir -p "${OUTPUT_ROOT}/experiments"
mkdir -p "${OUTPUT_ROOT}/runpod_logs"

# ============================================================
# 3. Keep repo-level folders intact
# ============================================================

echo
echo "==> Keeping repo data/ and outputs/ folders intact..."

mkdir -p data
mkdir -p outputs

# Do not force-create .gitkeep files here.
# If the repo already tracks placeholders, git pull will bring them.
# touch data/.gitkeep
# touch outputs/.gitkeep

# This folder is required by training and should remain a normal repo folder.
if [ ! -d "outputs/splits" ]; then
    echo "WARNING: outputs/splits does not exist."
    echo "Creating an empty outputs/splits folder, but you should verify split CSV files are present."
    mkdir -p outputs/splits
fi

echo
echo "==> Checking required split files..."
ls -lah outputs/splits || true

# ============================================================
# 4. Create only the required persistent symlinks
# ============================================================

echo
echo "==> Creating safe persistent symlinks..."

# Dataset
safe_link_path "${DATA_ROOT}/raw" "data/raw"

# Generated experiment outputs
safe_link_path "${OUTPUT_ROOT}/experiments" "outputs/experiments"

# RunPod-specific logs
safe_link_path "${OUTPUT_ROOT}/runpod_logs" "outputs/runpod_logs"

# Do NOT symlink outputs/splits.
# It is needed by training and should be read directly from the repo.

echo
echo "==> Structure check:"

echo
echo "data/:"
ls -lah data || true

echo
echo "outputs/:"
ls -lah outputs || true

echo
echo "Resolved paths:"
echo "data/raw:             $(readlink -f data/raw 2>/dev/null || echo 'not a symlink')"
echo "outputs/experiments:  $(readlink -f outputs/experiments 2>/dev/null || echo 'not a symlink')"
echo "outputs/runpod_logs:  $(readlink -f outputs/runpod_logs 2>/dev/null || echo 'not a symlink')"
echo "outputs/splits:       $(readlink -f outputs/splits 2>/dev/null || echo 'missing')"

# ============================================================
# 5. Install useful system packages
# ============================================================

echo
echo "==> Installing useful system packages..."

apt-get update
apt-get install -y unzip rsync tmux nano

# ============================================================
# 6. Install uv and sync environment
# ============================================================

echo
echo "==> Installing uv..."
python -m pip install --upgrade pip uv

echo
echo "==> Syncing uv environment..."

if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found in ${REPO_DIR}"
    exit 1
fi

uv sync

# ============================================================
# 7. Check GPU and PyTorch CUDA
# ============================================================

echo
echo "==> Checking GPU..."
nvidia-smi || true

echo
echo "==> Checking PyTorch CUDA from uv environment..."

uv run python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version used by torch:", torch.version.cuda)
else:
    raise SystemExit("ERROR: CUDA is not available in PyTorch.")
PY

# ============================================================
# 8. Disk usage summary
# ============================================================

echo
echo "==> Disk usage:"
df -h /workspace || true
du -sh /workspace/datasets/face-occlusion 2>/dev/null || true
du -sh /workspace/outputs/face-occlusion-estimation 2>/dev/null || true

echo
echo "==> Pod setup complete."

# ============================================================
# Configure persistent tmux settings
# ============================================================

echo
echo "==> Configuring persistent tmux settings..."

TMUX_CONF_PERSISTENT="/workspace/.tmux.conf"
TMUX_CONF_RUNTIME="${HOME}/.tmux.conf"

mkdir -p /workspace

# Create the persistent tmux config file if it does not exist.
touch "${TMUX_CONF_PERSISTENT}"

# Add mouse support only if it is not already configured.
if ! grep -qxF "set -g mouse on" "${TMUX_CONF_PERSISTENT}"; then
    echo "set -g mouse on" >> "${TMUX_CONF_PERSISTENT}"
    echo "    Added: set -g mouse on"
else
    echo "    OK: mouse support already configured"
fi

# Add a larger scrollback history only if it is not already configured.
if ! grep -qxF "set -g history-limit 50000" "${TMUX_CONF_PERSISTENT}"; then
    echo "set -g history-limit 50000" >> "${TMUX_CONF_PERSISTENT}"
    echo "    Added: set -g history-limit 50000"
else
    echo "    OK: history limit already configured"
fi

# Point ~/.tmux.conf to the persistent config.
if [ -L "${TMUX_CONF_RUNTIME}" ]; then
    CURRENT_TARGET="$(readlink -f "${TMUX_CONF_RUNTIME}")"
    EXPECTED_TARGET="$(readlink -f "${TMUX_CONF_PERSISTENT}")"

    if [ "${CURRENT_TARGET}" != "${EXPECTED_TARGET}" ]; then
        ln -sfn "${TMUX_CONF_PERSISTENT}" "${TMUX_CONF_RUNTIME}"
        echo "    Updated symlink: ${TMUX_CONF_RUNTIME} -> ${TMUX_CONF_PERSISTENT}"
    else
        echo "    OK: ${TMUX_CONF_RUNTIME} already points to persistent config"
    fi
elif [ -e "${TMUX_CONF_RUNTIME}" ]; then
    echo "    WARNING: ${TMUX_CONF_RUNTIME} exists and is not a symlink."
    echo "    I will not overwrite it. Persistent config is still available at:"
    echo "      ${TMUX_CONF_PERSISTENT}"
else
    ln -s "${TMUX_CONF_PERSISTENT}" "${TMUX_CONF_RUNTIME}"
    echo "    Created symlink: ${TMUX_CONF_RUNTIME} -> ${TMUX_CONF_PERSISTENT}"
fi

# Reload config if a tmux server is already running.
if command -v tmux >/dev/null 2>&1 && tmux ls >/dev/null 2>&1; then
    tmux source-file "${TMUX_CONF_PERSISTENT}"
    echo "    Reloaded tmux config"
else
    echo "    No active tmux server to reload"
fi

echo "==> tmux configuration complete."
