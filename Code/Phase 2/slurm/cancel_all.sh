#!/bin/bash
# cancel_all.sh — scancel every queued/running Phase 2 job owned by the caller.
set -euo pipefail

mapfile -t JIDS < <(squeue -u "$USER" -h -o "%i %j" | awk '$2 ~ /^p2_/ {print $1}')

if [[ ${#JIDS[@]} -eq 0 ]]; then
    echo "[cancel] no Phase 2 jobs found for user $USER."
    exit 0
fi

echo "[cancel] cancelling ${#JIDS[@]} job(s): ${JIDS[*]}"
scancel "${JIDS[@]}"
