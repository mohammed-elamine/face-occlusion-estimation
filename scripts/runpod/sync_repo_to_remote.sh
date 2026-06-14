#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Sync RunPod repo to exact remote GitHub branch state
# ============================================================
#
# Purpose:
# - Switch to a chosen remote branch.
# - Make the local repo exactly match origin/<branch>.
# - Ask before removing untracked files/directories.
# - Create data/raw as a symlink to persistent dataset storage:
#       data/raw -> /workspace/datasets/face-occlusion/raw
# - Keep outputs/experiments, outputs/runpod_logs, and outputs/splits
#   as normal repo folders. They are persistent because the repo is
#   already inside /workspace.
# - Copy latest repo scripts/runpod/*.sh to /workspace/scripts/
#   without deleting extra files.
# - Run uv sync after updating code.
# - Use persistent uv cache/environment to avoid reinstalling everything
#   from scratch on each pod.
#
# This script never pushes anything.
#
# Usage:
#   bash /workspace/scripts/sync_repo_to_remote.sh main
#   bash /workspace/scripts/sync_repo_to_remote.sh feat/contrastive-learning
#
# ============================================================

REPO_DIR="/workspace/repos/face-occlusion-estimation"
REMOTE_NAME="origin"
BRANCH="${1:-main}"

WORKSPACE_SCRIPTS_DIR="/workspace/scripts"
REPO_RUNPOD_SCRIPTS_DIR="scripts/runpod"

DATA_ROOT="/workspace/datasets/face-occlusion"

# Persistent uv settings
export UV_LINK_MODE="copy"
export UV_CACHE_DIR="/workspace/cache/uv"
export UV_PROJECT_ENVIRONMENT="/workspace/venvs/face-occlusion-estimation"

# ---------- Colors ----------
if [ -t 1 ]; then
    RED="\033[0;31m"
    GREEN="\033[0;32m"
    YELLOW="\033[1;33m"
    BLUE="\033[0;34m"
    RESET="\033[0m"
else
    RED=""
    GREEN=""
    YELLOW=""
    BLUE=""
    RESET=""
fi

info() {
    echo -e "${BLUE}==>${RESET} $*"
}

ok() {
    echo -e "${GREEN}OK:${RESET} $*"
}

warn() {
    echo -e "${YELLOW}WARNING:${RESET} $*"
}

error() {
    echo -e "${RED}ERROR:${RESET} $*"
}

confirm_yes_no() {
    local prompt="$1"
    local answer=""

    echo
    echo -e "${YELLOW}${prompt}${RESET}"
    read -r -p "Type yes to confirm, or press Enter for no: " answer

    if [ "${answer}" = "yes" ]; then
        return 0
    fi

    return 1
}

show_git_state() {
    echo "Branch: $(git branch --show-current 2>/dev/null || echo 'unknown')"
    echo "Commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
    echo
    echo "Status:"
    git status --short || true
}

clean_untracked_interactively() {
    info "Checking untracked files/directories..."

    local clean_preview
    clean_preview="$(git clean -nd || true)"

    if [ -z "${clean_preview}" ]; then
        ok "No untracked files/directories to remove."
        return 0
    fi

    warn "The following untracked files/directories would be removed:"
    echo
    echo "${clean_preview}"
    echo
    warn "This removes only untracked files inside the Git repo."
    warn "It does not remove tracked files."
    warn "It does not remove ignored files."
    warn "It does not push anything."

    if confirm_yes_no "Do you want to run git clean -fd and remove these untracked files?"; then
        info "Removing untracked files/directories..."
        git clean -fd
        ok "Untracked files removed."
        return 0
    fi

    warn "Skipped git clean. Untracked files were preserved."
    return 1
}

confirm_discard_tracked_changes() {
    info "Checking tracked local modifications..."

    local tracked_changes
    tracked_changes="$(git status --short | grep -v '^??' || true)"

    if [ -z "${tracked_changes}" ]; then
        ok "No tracked local modifications."
        return 0
    fi

    warn "The following tracked files have local modifications:"
    echo
    echo "${tracked_changes}"
    echo
    warn "To match the remote branch exactly, these tracked changes must be discarded."

    if confirm_yes_no "Do you want to discard tracked local modifications with git reset --hard?"; then
        return 0
    fi

    error "Aborted. Tracked local modifications were preserved."
    exit 1
}

switch_to_branch() {
    local branch="$1"
    local remote_ref="$2"

    if git show-ref --verify --quiet "refs/heads/${branch}"; then
        git switch "${branch}"
    else
        git switch -c "${branch}" --track "${remote_ref}"
    fi
}

copy_runpod_scripts_to_workspace() {
    info "Copying RunPod helper scripts to persistent /workspace/scripts..."

    if [ ! -d "${REPO_RUNPOD_SCRIPTS_DIR}" ]; then
        warn "Repo RunPod scripts directory not found:"
        echo "  ${REPO_DIR}/${REPO_RUNPOD_SCRIPTS_DIR}"
        warn "Skipping script copy."
        return 0
    fi

    mkdir -p "${WORKSPACE_SCRIPTS_DIR}"

    if command -v rsync >/dev/null 2>&1; then
        # No --delete: preserve extra files in /workspace/scripts.
        rsync -av --no-owner --no-group \
            "${REPO_RUNPOD_SCRIPTS_DIR}/" \
            "${WORKSPACE_SCRIPTS_DIR}/"
    else
        warn "rsync not found. Falling back to cp -r."
        cp -r "${REPO_RUNPOD_SCRIPTS_DIR}/." "${WORKSPACE_SCRIPTS_DIR}/"
    fi

    chmod +x "${WORKSPACE_SCRIPTS_DIR}"/*.sh 2>/dev/null || true

    ok "RunPod scripts copied to ${WORKSPACE_SCRIPTS_DIR}"
    echo
    ls -lah "${WORKSPACE_SCRIPTS_DIR}"
}

ensure_data_raw_symlink() {
    local target="${DATA_ROOT}/raw"
    local link_path="data/raw"

    info "Ensuring dataset link: ${link_path} -> ${target}"

    mkdir -p "${target}"
    mkdir -p data

    # Case 1: already a symlink.
    if [ -L "${link_path}" ]; then
        local current_target
        current_target="$(readlink -f "${link_path}")"

        local expected_target
        expected_target="$(readlink -f "${target}")"

        if [ "${current_target}" = "${expected_target}" ]; then
            ok "${link_path} already points to ${target}"
            return 0
        fi

        warn "${link_path} is a symlink but points to:"
        echo "  ${current_target}"
        echo "Updating it to:"
        echo "  ${expected_target}"

        unlink "${link_path}"
        ln -s "${target}" "${link_path}"

        ok "Updated ${link_path}"
        return 0
    fi

    # Case 2: data/raw does not exist.
    if [ ! -e "${link_path}" ]; then
        ln -s "${target}" "${link_path}"
        ok "Created ${link_path} -> ${target}"
        return 0
    fi

    # Case 3: data/raw exists but is not a directory.
    if [ ! -d "${link_path}" ]; then
        warn "${link_path} exists but is not a directory or symlink."
        warn "Skipping to avoid overwriting local files."
        return 0
    fi

    # Case 4: data/raw contains tracked files.
    # This means the repo still tracks placeholders under data/raw.
    local tracked_files
    tracked_files="$(git ls-files -- "${link_path}" || true)"

    if [ -n "${tracked_files}" ]; then
        warn "${link_path} contains tracked Git files."
        warn "Skipping symlink conversion to avoid making the repo dirty."
        echo
        echo "Tracked files:"
        echo "${tracked_files}"
        echo
        echo "Recommended fix on your local machine:"
        echo "  git rm -r --cached data/raw"
        echo "  mkdir -p data"
        echo "  touch data/.gitkeep"
        echo "  git add .gitignore data/.gitkeep"
        echo "  git commit -m \"Keep data directory as placeholder only\""
        echo "  git push"
        return 0
    fi

    # Case 5: data/raw is a safe placeholder directory.
    # Safe means empty or only containing .gitkeep.
    local non_placeholder_content
    non_placeholder_content="$(find "${link_path}" -mindepth 1 ! -name ".gitkeep" -print -quit 2>/dev/null || true)"

    if [ -z "${non_placeholder_content}" ]; then
        info "${link_path} is an untracked placeholder directory. Converting it to a symlink..."

        rm -rf "${link_path}"
        ln -s "${target}" "${link_path}"

        ok "Converted ${link_path} -> ${target}"
        return 0
    fi

    # Case 6: data/raw has real local content.
    warn "${link_path} exists and contains real local files."
    warn "Skipping symlink conversion to avoid data loss."
    echo
    echo "First non-placeholder item found:"
    echo "  ${non_placeholder_content}"
    echo
    echo "If this is old local data and you are sure the real dataset is in:"
    echo "  ${target}"
    echo "then remove data/raw manually and rerun this script."
}

verify_outputs_layout() {
    info "Verifying outputs layout..."

    mkdir -p outputs/experiments
    mkdir -p outputs/runpod_logs
    mkdir -p outputs/slurm_logs
    mkdir -p outputs/splits

    echo
    echo "outputs/experiments: $(readlink -f outputs/experiments)"
    echo "outputs/runpod_logs: $(readlink -f outputs/runpod_logs)"
    echo "outputs/slurm_logs:  $(readlink -f outputs/slurm_logs)"
    echo "outputs/splits:      $(readlink -f outputs/splits)"
    echo
    ok "outputs/* folders are real repo folders. They persist because the repo is inside /workspace."
}

ensure_project_layout() {
    info "Ensuring RunPod project layout..."

    case "${REPO_DIR}" in
        /workspace/*)
            ok "Repository is inside /workspace, so repo-local outputs are persistent."
            ;;
        *)
            warn "Repository is not inside /workspace. Repo-local outputs may not persist after pod deletion."
            ;;
    esac

    ensure_data_raw_symlink
    verify_outputs_layout

    echo
    info "Project path check:"
    echo "data/raw:             $(readlink -f data/raw 2>/dev/null || echo 'missing')"
    echo "outputs/experiments:  $(readlink -f outputs/experiments 2>/dev/null || echo 'missing')"
    echo "outputs/runpod_logs:  $(readlink -f outputs/runpod_logs 2>/dev/null || echo 'missing')"
    echo "outputs/splits:       $(readlink -f outputs/splits 2>/dev/null || echo 'missing')"
}

echo "============================================================"
echo "Sync repo to remote branch"
echo "============================================================"
echo "Repo dir: ${REPO_DIR}"
echo "Remote:   ${REMOTE_NAME}"
echo "Branch:   ${BRANCH}"
echo "============================================================"
echo

if [ ! -d "${REPO_DIR}/.git" ]; then
    error "Git repository not found at:"
    echo "  ${REPO_DIR}"
    echo
    echo "Run setup_pod.sh first."
    exit 1
fi

cd "${REPO_DIR}"

info "Current Git state before sync:"
show_git_state
echo

info "Fetching latest remote refs..."
git fetch "${REMOTE_NAME}" --prune

REMOTE_REF="${REMOTE_NAME}/${BRANCH}"

if ! git rev-parse --verify "${REMOTE_REF}" >/dev/null 2>&1; then
    error "Remote branch not found: ${REMOTE_REF}"
    echo
    echo "Available remote branches:"
    git branch -r
    exit 1
fi

echo
info "Target remote branch exists: ${REMOTE_REF}"
echo "Target commit: $(git rev-parse "${REMOTE_REF}")"

# ============================================================
# Step 1: Ask before discarding tracked changes
# ============================================================

echo
confirm_discard_tracked_changes

# ============================================================
# Step 2: Try switching branch
# ============================================================

echo
info "Switching to branch: ${BRANCH}"

set +e
SWITCH_OUTPUT="$(switch_to_branch "${BRANCH}" "${REMOTE_REF}" 2>&1)"
SWITCH_STATUS=$?
set -e

if [ "${SWITCH_STATUS}" -ne 0 ]; then
    warn "Git could not switch branches cleanly."
    echo
    echo "Git reported:"
    echo "------------------------------------------------------------"
    echo "${SWITCH_OUTPUT}"
    echo "------------------------------------------------------------"
    echo
    warn "This is commonly caused by untracked files that would be overwritten by the target branch."

    if clean_untracked_interactively; then
        echo
        info "Retrying branch switch after cleaning untracked files..."
        switch_to_branch "${BRANCH}" "${REMOTE_REF}"
    else
        echo
        error "Aborted. Branch switch is still blocked because untracked files were preserved."
        echo
        echo "Inspect current files with:"
        echo "  git status --short"
        echo
        echo "Preview untracked deletion with:"
        echo "  git clean -nd"
        echo
        echo "Remove untracked files manually with:"
        echo "  git clean -fd"
        echo
        exit 1
    fi
else
    echo "${SWITCH_OUTPUT}"
fi

# ============================================================
# Step 3: Reset tracked files exactly to remote
# ============================================================

echo
info "Resetting tracked files to exact remote state:"
echo "  ${REMOTE_REF}"

git reset --hard "${REMOTE_REF}"

# ============================================================
# Step 4: Clean remaining untracked files, with confirmation
# ============================================================

echo
info "Checking remaining untracked files after reset..."

if ! clean_untracked_interactively; then
    warn "Untracked files remain. This may be fine if they are intentional local files."
fi

# ============================================================
# Step 5: Ensure project layout
# ============================================================

echo
ensure_project_layout

# ============================================================
# Step 6: Copy latest scripts/runpod/* to /workspace/scripts/
# ============================================================

echo
copy_runpod_scripts_to_workspace

# ============================================================
# Step 7: Final Git state
# ============================================================

echo
info "Final Git state after sync:"
show_git_state

# ============================================================
# Step 8: uv sync with persistent cache/env
# ============================================================

echo
info "Configuring persistent uv cache/environment..."

mkdir -p "${UV_CACHE_DIR}"
mkdir -p "$(dirname "${UV_PROJECT_ENVIRONMENT}")"

echo "UV_LINK_MODE=${UV_LINK_MODE}"
echo "UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"

echo
info "Checking uv availability..."

if ! command -v uv >/dev/null 2>&1; then
    warn "uv is not installed. Installing uv..."
    python -m pip install --upgrade pip uv
fi

echo
info "Syncing uv environment..."

if [ ! -f "pyproject.toml" ]; then
    error "pyproject.toml not found in ${REPO_DIR}"
    exit 1
fi

uv sync

# ============================================================
# Step 9: CUDA check
# ============================================================

echo
info "Checking PyTorch CUDA after uv sync..."

uv run python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise SystemExit(
        "ERROR: CUDA is not available. "
        "Do not run training until PyTorch/CUDA compatibility is fixed."
    )
PY

echo
echo "============================================================"
ok "Repo sync complete."
echo "============================================================"
echo "Branch: ${BRANCH}"
echo "Commit: $(git rev-parse HEAD)"
echo
echo "You can now run, for example:"
echo "  bash /workspace/scripts/run_experiment_tmux.sh configs/baseline.yaml"
