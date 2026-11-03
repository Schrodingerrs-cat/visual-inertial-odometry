#!/bin/bash
# status.sh — print current Phase 2 slurm jobs for the calling user.
set -euo pipefail
squeue -u "$USER" --format="%.10i %.20j %.10T %.10M %.9l %R" \
    | awk 'NR==1 || $2 ~ /^p2_/'
