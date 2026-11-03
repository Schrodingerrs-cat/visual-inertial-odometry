#!/usr/bin/env bash
# generate_all_data.sh — render all 8 Phase 2 trajectories
# (4 train + 3 test + 1 minitest validation).
#
# Mapping (see blender/generate_data.py TRAJ_ID_MAP):
#   0..3 → sinusoid training trajectories
#   100  → flat_spiral (test)
#   101  → figure_8    (test)
#   102  → clover      (test)
#   200  → minitest    (validation; short clover, reduced amplitude)
#
# Usage:
#   bash scripts/generate_all_data.sh
#
# Run `python scripts/prepare_textures.py` ONCE before this script so the
# texture pools under Data/textures_train/ and Data/textures_test/ exist.
#
# Environment:
#   BLENDER   — path to the Blender binary. Defaults to the locally-installed
#               ./blender_app/current/blender (created by setup_blender.sh).
#   DATA_ROOT — output root. Defaults to data/raw.
#   SKIP_EXIST — if set to "1", skip trajectories that already have metadata.json.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$HERE"

BLENDER="${BLENDER:-./blender_app/current/blender}"
DATA_ROOT="${DATA_ROOT:-data/raw}"
SKIP_EXIST="${SKIP_EXIST:-1}"

if [[ ! -x "$BLENDER" ]]; then
    echo "[gen] Blender binary not found at $BLENDER"
    echo "[gen] Run bash setup_blender.sh first, or set \$BLENDER to your install."
    exit 1
fi

mkdir -p "$DATA_ROOT"

TRAIN_IDS=(0 1 2 3)
TEST_IDS=(100 101 102)
VAL_IDS=(200)

render_one () {
    local id="$1"
    echo "[gen] Rendering traj $id"
    local extra=""
    if [[ "$SKIP_EXIST" == "1" ]]; then extra="--skip_if_exists"; fi
    "$BLENDER" --background --python blender/generate_data.py -- \
        --trajectory_id "$id" \
        --output_dir "$DATA_ROOT" \
        $extra
}

for id in "${TRAIN_IDS[@]}"; do render_one "$id"; done
for id in "${TEST_IDS[@]}";  do render_one "$id"; done
for id in "${VAL_IDS[@]}";   do render_one "$id"; done

echo "[gen] Done. 8 trajectories under $DATA_ROOT/"
