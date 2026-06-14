#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Bootstrap GitHub SSH access on RunPod
# ============================================================
#
# Persistent deploy key:
#   /workspace/.ssh/github_runpod
#
# Runtime SSH key copy:
#   ~/.ssh/github_runpod
#
# Usage:
#   bash /workspace/scripts/bootstrap_runtime_ssh.sh
#
# ============================================================

PERSISTENT_KEY="/workspace/.ssh/github_runpod"
PERSISTENT_PUB_KEY="/workspace/.ssh/github_runpod.pub"
RUNTIME_SSH_DIR="${HOME}/.ssh"
RUNTIME_KEY="${RUNTIME_SSH_DIR}/github_runpod"
RUNTIME_PUB_KEY="${RUNTIME_SSH_DIR}/github_runpod.pub"

echo "==> Bootstrapping runtime SSH configuration..."

if [ ! -f "${PERSISTENT_KEY}" ]; then
    echo "ERROR: Persistent private key not found at:"
    echo "  ${PERSISTENT_KEY}"
    echo
    echo "Create it first with:"
    echo "  mkdir -p /workspace/.ssh"
    echo "  ssh-keygen -t ed25519 -C \"runpod-face-occlusion\" -f /workspace/.ssh/github_runpod"
    exit 1
fi

mkdir -p "${RUNTIME_SSH_DIR}"
chmod 700 "${RUNTIME_SSH_DIR}"

cp "${PERSISTENT_KEY}" "${RUNTIME_KEY}"
chmod 600 "${RUNTIME_KEY}"

if [ -f "${PERSISTENT_PUB_KEY}" ]; then
    cp "${PERSISTENT_PUB_KEY}" "${RUNTIME_PUB_KEY}"
    chmod 644 "${RUNTIME_PUB_KEY}"
fi

cat > "${RUNTIME_SSH_DIR}/config" <<'EOF'
Host github-face-occlusion
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_runpod
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new

Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_runpod
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
EOF

chmod 600 "${RUNTIME_SSH_DIR}/config"

echo "==> SSH files:"
ls -lah "${RUNTIME_SSH_DIR}"

echo
echo "==> Testing GitHub SSH access..."
set +e
ssh -T git@github-face-occlusion
STATUS=$?
set -e

echo
echo "GitHub SSH test finished with exit code: ${STATUS}"
echo "If you see 'successfully authenticated' or a repo-specific success message, SSH is working."
echo "GitHub may return a non-zero code because it does not provide shell access."

echo
echo "==> Runtime SSH bootstrap complete."
