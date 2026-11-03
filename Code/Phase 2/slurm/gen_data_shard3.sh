#!/bin/bash
#SBATCH --job-name=p2_gen3
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/gen3.out
#SBATCH --error=slurm/logs/gen3.err

# Phase 2 data shard 3 (of 4): sinusoid traj 3 + minitest validation (200).
# Minitest is 2 s (201 frames), rendering is quick; this shard takes ~90 min
# because the sinusoid trajectory is full-length 100 s.
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

SHARD_IDS=(3 200)

for ID in "${SHARD_IDS[@]}"; do
    echo "===== gen traj $ID (shard 3) ====="
    "$BLENDER" --background --python blender/generate_data.py -- \
        --trajectory_id "$ID" \
        --output_dir data/raw \
        --skip_if_exists
done
