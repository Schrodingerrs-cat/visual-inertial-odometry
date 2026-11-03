#!/bin/bash
#SBATCH --job-name=p2_gen0
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/gen0.out
#SBATCH --error=slurm/logs/gen0.err

# Phase 2 data shard 0 (of 4): sinusoid traj 0 + flat_spiral test (100).
# 100 s trajectories at 1000 Hz → 10001 rendered frames per trajectory.
# Post 2026-04-24 overhaul (640×480 EEVEE taa=64):
#   ~2× render time vs 320×240 → ~160 min / trajectory → ~320 min / shard.
#   12 h SLURM cap leaves ~2× safety.
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

SHARD_IDS=(0 100)

for ID in "${SHARD_IDS[@]}"; do
    echo "===== gen traj $ID (shard 0) ====="
    "$BLENDER" --background --python blender/generate_data.py -- \
        --trajectory_id "$ID" \
        --output_dir data/raw \
        --skip_if_exists
done
