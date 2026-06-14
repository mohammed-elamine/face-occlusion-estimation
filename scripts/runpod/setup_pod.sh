#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# RunPod project setup for face-occlusion-estimation
# ============================================================
#
# This script:
# 1. Bootstraps GitHub SSH access using /workspace/scripts/bootstrap_runtime_ssh.sh.
# 2. Clones or updates the private GitHub repo.
# 3. Checks out the requested branch (--branch); the env (uv.lock/deps) is
#    branch-specific, so this drives which dependencies get installed.
# 4. Keeps repo-level data/ and outputs/ folders intact.
# 5. Links only remote/persistent folders:
#       data/raw
#       outputs/experiments
#       outputs/runpod_logs
# 6. Preserves outputs/splits as a normal repo folder, because training reads it.
# 7. Installs useful system packages.
# 8. Installs dependencies with uv (optional extras via --extra).
# 9. Checks GPU and PyTorch CUDA availability.
#
# Usage:
#   bash /workspace/scripts/setup_pod.sh [--branch <name>] [--extra <name> ...]
#
#   --branch, -b <name>   Branch to check out and sync (default: current/clone default).
#   --extra,  -e <name>   Optional dependency extra to install; repeatable
#                         (e.g. --extra synthetic --extra wandb).
#   --help,   -h          Show this help and exit.
#
# Env-var overrides (flags take precedence):
#   BRANCH=<name>         Same as --branch.
#   EXTRAS="a b"          Space-separated extras, same as repeated --extra.
#
# Examples:
#   bash setup_pod.sh --branch feat/lora
#   bash setup_pod.sh -b main --extra synthetic
#
# ============================================================

REPO_PARENT="/workspace/repos"
REPO_DIR="/workspace/repos/face-occlusion-estimation"
REPO_URL="git@github-face-occlusion:mohammed-elamine/face-occlusion-estimation.git"

DATA_ROOT="/workspace/datasets/face-occlusion"
OUTPUT_ROOT="/workspace/outputs/face-occlusion-estimation"

# Persistent uv cache + project environment on the /workspace volume. These MUST
# match run_experiment.sh and sync_repo_to_remote.sh so the install done here warms
# the exact cache + venv those scripts reuse (default cache is /root/.cache/uv,
# which is ephemeral, and the default .venv would be a separate environment).
export UV_LINK_MODE="copy"
export UV_CACHE_DIR="/workspace/cache/uv"
export UV_PROJECT_ENVIRONMENT="/workspace/venvs/face-occlusion-estimation"

SCRIPTS_DIR="/workspace/scripts"
BOOTSTRAP_SSH_SCRIPT="${SCRIPTS_DIR}/bootstrap_runtime_ssh.sh"

# ============================================================
# Parse arguments
# ============================================================
#
# The dependency environment is branch-specific (each branch carries its own
# pyproject.toml / uv.lock), so the branch and any optional extras are explicit
# inputs rather than whatever `git clone` happened to leave checked out.

BRANCH="${BRANCH:-}"
NONINTERACTIVE=0
# Seed extras from the EXTRAS env var (space-separated); flags append to this.
read -r -a EXTRAS <<< "${EXTRAS:-}"

usage() {
    cat <<'EOF'
Usage: setup_pod.sh [--branch <name>] [--extra <name> ...] [--no-interactive]

  --branch, -b <name>   Branch to check out and sync. If omitted and running
                        interactively, you are shown a picker of the repo's
                        branches (fzf if installed, else a numbered menu). The
                        env (uv.lock/deps) is branch-specific, so this drives
                        which dependencies get installed.
  --extra,  -e <name>   Optional dependency extra to install; repeatable
                        (e.g. --extra synthetic --extra wandb).
  --no-interactive      Never prompt; keep the current/clone-default branch when
                        no --branch is given (for cron/headless runs).
  --help,   -h          Show this help and exit.

Env-var overrides (flags take precedence):
  BRANCH=<name>         Same as --branch.
  EXTRAS="a b"          Space-separated extras, same as repeated --extra.

Examples:
  bash setup_pod.sh                       # pick a branch interactively
  bash setup_pod.sh --branch feat/lora
  bash setup_pod.sh -b main --extra synthetic
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -b|--branch)
            [ "$#" -ge 2 ] || { echo "ERROR: $1 requires a value"; exit 2; }
            BRANCH="$2"
            shift 2
            ;;
        -e|--extra)
            [ "$#" -ge 2 ] || { echo "ERROR: $1 requires a value"; exit 2; }
            EXTRAS+=("$2")
            shift 2
            ;;
        --no-interactive)
            NONINTERACTIVE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1"
            echo "Run with --help for usage."
            exit 2
            ;;
    esac
done

# Build the `uv sync` extra flags once (e.g. --extra synthetic --extra wandb).
UV_EXTRA_ARGS=()
for _extra in "${EXTRAS[@]:-}"; do
    [ -n "${_extra}" ] && UV_EXTRA_ARGS+=(--extra "${_extra}")
done

# ============================================================
# Helper: interactively pick a branch (sets BRANCH)
# ============================================================
#
# Lists the repo's remote branches and lets the user choose one. Prefers fzf
# (fuzzy + scroll) and tries a best-effort install if it is missing; otherwise
# falls back to the bash `select` numbered menu. Must run AFTER `git fetch` so the
# remote branches are known. Pressing Esc / Enter with no choice keeps the current
# branch (BRANCH stays empty).

select_branch_interactive() {
    local branches current i reply
    # Filter on the full refname so the origin/HEAD symref is dropped cleanly
    # (its short form is "origin", which a HEAD-only filter would miss).
    mapfile -t branches < <(
        git for-each-ref --format='%(refname)' refs/remotes/origin 2>/dev/null \
            | grep -v '/HEAD$' \
            | sed 's#^refs/remotes/origin/##' \
            | sort -u
    )
    if [ "${#branches[@]}" -eq 0 ]; then
        echo "    No remote branches found; keeping the current branch."
        return 0
    fi

    current="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
    echo
    echo "==> Select a branch to check out and sync (current: ${current})"

    # Best-effort: get fzf for the nicer scrollable picker if it is missing.
    if ! command -v fzf >/dev/null 2>&1; then
        apt-get install -y fzf >/dev/null 2>&1 \
            || { apt-get update >/dev/null 2>&1 && apt-get install -y fzf >/dev/null 2>&1; } \
            || true
    fi

    if command -v fzf >/dev/null 2>&1; then
        # `|| BRANCH=""` keeps set -e happy when the user presses Esc (fzf exits non-zero).
        BRANCH="$(printf '%s\n' "${branches[@]}" \
            | fzf --prompt='branch> ' --height='40%' --reverse --no-multi \
                  --header='type to filter | up/down to scroll | Enter to select | Esc to keep current')" \
            || BRANCH=""
    else
        echo "    (install fzf for a scrollable fuzzy picker; using a numbered menu)"
        for i in "${!branches[@]}"; do
            printf "  %2d) %s\n" "$((i + 1))" "${branches[$i]}"
        done
        reply=""
        read -r -p "Enter number (or just Enter to keep '${current}'): " reply || true
        # Blank or non-numeric or out-of-range -> keep current (BRANCH stays empty).
        if [[ "${reply}" =~ ^[0-9]+$ ]] \
            && [ "${reply}" -ge 1 ] && [ "${reply}" -le "${#branches[@]}" ]; then
            BRANCH="${branches[$((reply - 1))]}"
        fi
    fi

    if [ -n "${BRANCH}" ]; then
        echo "==> Selected branch: ${BRANCH}"
    else
        echo "==> No selection; keeping the current branch (${current})."
    fi
}

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
fi

cd "${REPO_DIR}"

# Make remote branches/commits known before checking out or fast-forwarding.
echo "==> Fetching remote..."
git fetch --all --prune || true

# No branch requested: offer an interactive picker when attached to a terminal.
# Skipped with --no-interactive or when stdin is not a TTY (cron/headless).
if [ -z "${BRANCH}" ] && [ "${NONINTERACTIVE}" -eq 0 ] && [ -t 0 ]; then
    select_branch_interactive
fi

# The branch determines which pyproject.toml / uv.lock (and thus which deps) get
# synced below, so check it out explicitly when requested.
if [ -n "${BRANCH}" ]; then
    echo "==> Checking out branch: ${BRANCH}"
    git checkout "${BRANCH}"
    echo "==> Fast-forwarding ${BRANCH}..."
    git pull --ff-only origin "${BRANCH}" \
        || echo "WARNING: could not fast-forward ${BRANCH}; staying at current commit."
else
    echo "==> No --branch given; fast-forwarding current branch..."
    git pull --ff-only \
        || echo "WARNING: could not fast-forward; staying at current commit."
fi

echo
echo "==> On branch: $(git rev-parse --abbrev-ref HEAD)"
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
apt-get install -y unzip rsync tmux nano fzf

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

# Ensure the persistent cache/venv locations exist before syncing into them.
mkdir -p "${UV_CACHE_DIR}"
mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"
echo "    UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "    UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"

if [ "${#UV_EXTRA_ARGS[@]}" -gt 0 ]; then
    echo "    extras: ${EXTRAS[*]}"
    uv sync "${UV_EXTRA_ARGS[@]}"
else
    uv sync
fi

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
