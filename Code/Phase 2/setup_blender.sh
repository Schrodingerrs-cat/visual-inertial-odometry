#!/usr/bin/env bash
# setup_blender.sh — download and extract Blender 4.0 LTS into blender_app/ so
# the cluster need not have a system Blender. Idempotent: will skip the
# download if the expected directory already exists.
#
# Usage:
#   bash setup_blender.sh
#
# After completion, invoke Blender as:
#   ./blender_app/blender-4.0.2-linux-x64/blender --background --python ...

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

BLENDER_VERSION="4.0.2"
TARBALL="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
URL="https://download.blender.org/release/Blender4.0/${TARBALL}"
DEST_DIR="blender_app"
EXTRACTED="${DEST_DIR}/blender-${BLENDER_VERSION}-linux-x64"

mkdir -p "$DEST_DIR"

if [[ -x "${EXTRACTED}/blender" ]]; then
    echo "[setup_blender] Blender already extracted at ${EXTRACTED}/blender"
    exit 0
fi

if [[ ! -f "${DEST_DIR}/${TARBALL}" ]]; then
    echo "[setup_blender] Downloading ${URL}"
    if command -v curl >/dev/null 2>&1; then
        curl -L -o "${DEST_DIR}/${TARBALL}" "$URL"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${DEST_DIR}/${TARBALL}" "$URL"
    else
        echo "[setup_blender] Neither curl nor wget is installed."
        exit 1
    fi
else
    echo "[setup_blender] Tarball already present, re-using ${DEST_DIR}/${TARBALL}"
fi

echo "[setup_blender] Extracting"
tar -xf "${DEST_DIR}/${TARBALL}" -C "${DEST_DIR}"

if [[ ! -x "${EXTRACTED}/blender" ]]; then
    echo "[setup_blender] FAIL: extraction did not produce ${EXTRACTED}/blender"
    exit 1
fi

# Symlink for convenience so scripts can use a stable path.
ln -sfn "blender-${BLENDER_VERSION}-linux-x64" "${DEST_DIR}/current"

echo "[setup_blender] Done. Binary at: ${EXTRACTED}/blender"
echo "[setup_blender] Stable symlink:  ${DEST_DIR}/current/blender"
