#!/bin/bash
# _common.sh — sourced by every Phase 2 slurm script.
#
# Responsibilities:
#   1. Source conda.sh and activate the `vio` env (fallback: local .venv/).
#   2. Export LD_LIBRARY_PATH with conda's sysroot (matches user's template).
#   3. cd into the Phase 2 root (auto-detected from $BASH_SOURCE unless
#      $PHASE2_ROOT is already set).
#   4. Ensure output directories exist.
#   5. Emit a banner and an EXIT trap for the end-of-job summary.
#
# Override hooks:
#   PHASE2_ROOT    — absolute path to "Code/Phase 2" (auto-detected otherwise)
#   CONDA_ENV      — conda env name (default: vio)
#   USE_VENV=1     — force the venv path even if conda is available

# Only enable strict mode for batch invocations. Interactive sourcing must
# not inherit set -e, or a failing command inside an interactive shell will
# kill the shell (and ssh session).
(return 0 2>/dev/null) && _COMMON_SOURCED=1 || _COMMON_SOURCED=0
if [[ "$_COMMON_SOURCED" == "0" ]]; then
    set -euo pipefail
fi

# --- Root detection ----------------------------------------------------------
if [[ -z "${PHASE2_ROOT:-}" ]]; then
    _this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PHASE2_ROOT="$(cd "$_this_dir/.." && pwd)"
fi
export PHASE2_ROOT

# --- Python env --------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-vio}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

_activated=0
if [[ "${USE_VENV:-0}" != "1" && -f "$CONDA_SH" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
        conda activate "$CONDA_ENV"
        export LD_LIBRARY_PATH="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
        _activated=1
    fi
fi
if [[ "$_activated" == "0" ]]; then
    if [[ -f "$PHASE2_ROOT/.venv/bin/activate" ]]; then
        # shellcheck disable=SC1090
        source "$PHASE2_ROOT/.venv/bin/activate"
        # Blender needs libs that live in conda's sysroot even when we're on
        # the venv fallback path. If a sibling (base) conda env is on the
        # shell, use its sysroot; otherwise try the system default conda.
        _CONDA_ROOT="${CONDA_PREFIX:-}"
        if [[ -z "$_CONDA_ROOT" && -d "$HOME/miniconda3" ]]; then
            _CONDA_ROOT="$HOME/miniconda3"
        fi
        if [[ -n "$_CONDA_ROOT" ]]; then
            if [[ -d "$_CONDA_ROOT/x86_64-conda-linux-gnu/sysroot/usr/lib64" ]]; then
                export LD_LIBRARY_PATH="$_CONDA_ROOT/x86_64-conda-linux-gnu/sysroot/usr/lib64:$_CONDA_ROOT/lib:${LD_LIBRARY_PATH:-}"
            elif [[ -d "$_CONDA_ROOT/lib" ]]; then
                export LD_LIBRARY_PATH="$_CONDA_ROOT/lib:${LD_LIBRARY_PATH:-}"
            fi
        fi
        unset _CONDA_ROOT
    else
        echo "[slurm] no conda env '$CONDA_ENV' and no .venv at $PHASE2_ROOT/.venv — aborting." >&2
        if [[ "$_COMMON_SOURCED" == "1" ]]; then return 1; else exit 1; fi
    fi
fi

# --- Working dir + output tree ----------------------------------------------
cd "$PHASE2_ROOT"
mkdir -p outputs slurm/logs data/raw data/processed

BLENDER="${BLENDER:-$PHASE2_ROOT/blender_app/current/blender}"
export BLENDER

# --- Banner + exit trap ------------------------------------------------------
# Use shell-builtin basename (${var##*/}) instead of /usr/bin/basename so that
# interactive sourcing — where $0 is "-bash" — doesn't blow up with
# "basename: invalid option -- 'b'" and kill the shell under `set -e`.
_JOB_SRC="${SLURM_JOB_NAME:-${BASH_SOURCE[0]:-${0:-interactive}}}"
JOB_NAME="${_JOB_SRC##*/}"
unset _JOB_SRC
echo "========================================="
echo "Phase2 job: $JOB_NAME"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "PHASE2_ROOT=$PHASE2_ROOT"
echo "python=$(which python)"
echo "BLENDER=$BLENDER"
echo "========================================="
trap '_rc=$?; echo "========================================="; echo "End: $(date)"; echo "Exit code: $_rc"; echo "========================================="' EXIT

nvidia-smi || true
