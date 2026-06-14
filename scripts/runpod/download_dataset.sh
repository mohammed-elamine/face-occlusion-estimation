#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Download the face occlusion dataset
# ============================================================
#
# Dataset URL:
#   https://partage.imt.fr/index.php/s/ntYk27ZFCbeKGqW/download
#
# The dataset will be stored persistently in:
#   /workspace/datasets/face-occlusion/
#
# Through the repo symlink, this is also:
#   /workspace/repos/face-occlusion-estimation/data/
#
# ============================================================

REPO_DIR="/workspace/repos/face-occlusion-estimation"
DATA_DIR="/workspace/datasets/face-occlusion"
ZIP_PATH="${DATA_DIR}/crops.zip"
DATASET_URL="https://partage.imt.fr/index.php/s/ntYk27ZFCbeKGqW/download"

echo "==> Checking repository..."
if [ ! -d "${REPO_DIR}" ]; then
    echo "ERROR: Repository not found at ${REPO_DIR}"
    exit 1
fi

mkdir -p "${DATA_DIR}"

echo "==> Dataset directory:"
echo "  ${DATA_DIR}"

if [ -f "${ZIP_PATH}" ]; then
    echo "==> crops.zip already exists:"
    ls -lh "${ZIP_PATH}"
    echo
    read -r -p "Do you want to re-download it? [y/N] " ANSWER
    ANSWER="${ANSWER:-N}"

    if [[ "${ANSWER}" =~ ^[Yy]$ ]]; then
        rm -f "${ZIP_PATH}"
    else
        echo "==> Keeping existing zip."
    fi
fi

if [ ! -f "${ZIP_PATH}" ]; then
    echo "==> Downloading dataset..."
    wget -O "${ZIP_PATH}" "${DATASET_URL}"
fi

echo
echo "==> Download complete:"
ls -lh "${ZIP_PATH}"

echo
read -r -p "Do you want to unzip crops.zip now? [Y/n] " UNZIP_ANSWER
UNZIP_ANSWER="${UNZIP_ANSWER:-Y}"

if [[ "${UNZIP_ANSWER}" =~ ^[Yy]$ ]]; then
    echo "==> Unzipping dataset..."

    # unzip is often installed, but install it if missing.
    if ! command -v unzip >/dev/null 2>&1; then
        apt-get update
        apt-get install -y unzip
    fi

    unzip -n "${ZIP_PATH}" -d "${DATA_DIR}"

    echo
    echo "==> Unzip complete."
fi

echo
echo "==> Dataset size:"
du -sh "${DATA_DIR}"

echo
echo "==> Dataset files:"
find "${DATA_DIR}" -maxdepth 3 -type f | head -50

echo
echo "==> Dataset ready."
