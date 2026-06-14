#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Launch one RunPod experiment inside tmux
# ============================================================
#
# Usage:
#   bash /workspace/scripts/run_experiment_tmux.sh CONFIG [SESSION_NAME]
#
# Example:
#   bash /workspace/scripts/run_experiment_tmux.sh configs/00_baseline.yaml
#
# Reattach later with:
#   tmux attach -t <session_name>
#
# Detach from tmux with:
#   Ctrl+b then d
#
# List sessions with:
#   tmux ls
# or equivalently with:
#   tmux list-sessions
# ============================================================

REPO_DIR="/workspace/repos/face-occlusion-estimation"
RUN_SCRIPT="/workspace/scripts/run_experiment.sh"

CONFIG="${1:-configs/baseline.yaml}"

cd "${REPO_DIR}"

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config file not found:"
    echo "  ${CONFIG}"
    exit 1
fi

if [ ! -f "${RUN_SCRIPT}" ]; then
    echo "ERROR: Missing run script:"
    echo "  ${RUN_SCRIPT}"
    exit 1
fi

RUN_NAME="$(basename "${CONFIG}" .yaml)"
SAFE_RUN_NAME="$(echo "${RUN_NAME}" | tr -c 'A-Za-z0-9_-' '_')"

SESSION_NAME="${2:-runpod_${SAFE_RUN_NAME}}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "==> tmux not found. Installing..."
    apt-get update
    apt-get install -y tmux
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "ERROR: tmux session already exists:"
    echo "  ${SESSION_NAME}"
    echo
    echo "Attach with:"
    echo "  tmux attach -t ${SESSION_NAME}"
    echo
    echo "Or choose another session name:"
    echo "  bash /workspace/scripts/run_experiment_tmux.sh ${CONFIG} ${SESSION_NAME}_2"
    exit 1
fi

echo "==> Starting tmux session: ${SESSION_NAME}"
echo "==> Config: ${CONFIG}"

tmux new-session -d -s "${SESSION_NAME}" \
    "cd '${REPO_DIR}' && bash '${RUN_SCRIPT}' '${CONFIG}'; echo; echo 'Experiment command finished. Press Ctrl+b then d to detach, or type exit to close.'; exec bash"

echo
echo "Experiment started in tmux."
echo
echo "Attach with:"
echo "  tmux attach -t ${SESSION_NAME}"
echo
echo "Detach from tmux with:"
echo "  Ctrl+b then d"
echo
echo "List tmux sessions with:"
echo "  tmux ls"
