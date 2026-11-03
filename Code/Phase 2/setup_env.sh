#!/usr/bin/env bash
# setup_env.sh — create a Python 3.10 venv for Phase 2 and install pinned deps.
#
# Idempotent: running this script a second time re-uses the existing venv and
# re-runs pip install (pip will skip already-satisfied requirements).
#
# Usage:
#   bash setup_env.sh
#   source .venv/bin/activate
#
# Requires python3.10 on PATH. On Turing:
#   module load python/3.10

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VENV_DIR=".venv"
PY_BIN="${PYTHON:-python3.10}"

if ! command -v "$PY_BIN" >/dev/null 2>&1; then
    echo "[setup_env] $PY_BIN not found on PATH."
    echo "[setup_env] On Turing run:  module load python/3.10"
    exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[setup_env] Creating venv at $VENV_DIR"
    "$PY_BIN" -m venv "$VENV_DIR"
else
    echo "[setup_env] Re-using existing venv at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools

echo "[setup_env] Installing pinned requirements"
python -m pip install -r requirements.txt

# Optional: clone Depth-Anything-V2 as a submodule-ish sibling. We do NOT add
# it to requirements.txt (heavy install and not strictly required). It is
# available under third_party/ if any downstream experiment wants to use it.
THIRD_PARTY="third_party"
mkdir -p "$THIRD_PARTY"
if [[ ! -d "$THIRD_PARTY/Depth-Anything-V2" ]]; then
    echo "[setup_env] Cloning Depth-Anything-V2 into $THIRD_PARTY/"
    git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2.git \
        "$THIRD_PARTY/Depth-Anything-V2" || {
            echo "[setup_env] WARN: Depth-Anything-V2 clone failed; continuing without it."
        }
else
    echo "[setup_env] Depth-Anything-V2 already present under $THIRD_PARTY/"
fi

python - <<'PY'
import importlib, sys
required = ["torch", "torchvision", "numpy", "scipy", "cv2", "PIL",
            "imageio", "matplotlib", "tqdm", "kornia", "evo", "yaml",
            "tensorboard", "einops", "transformers", "huggingface_hub", "pytest"]
missing = []
for m in required:
    try:
        importlib.import_module(m)
    except ImportError as exc:
        missing.append((m, str(exc)))
if missing:
    print("[setup_env] FAIL: missing modules:", missing)
    sys.exit(1)
print("[setup_env] OK — all required modules import.")
PY

echo "[setup_env] Done. Activate with:  source $VENV_DIR/bin/activate"
