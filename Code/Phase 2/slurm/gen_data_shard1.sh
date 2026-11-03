#!/bin/bash
#SBATCH --job-name=p2_gen1
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/gen1.out
#SBATCH --error=slurm/logs/gen1.err

# Phase 2 data shard 1 (of 4): sinusoid traj 1 + figure_8 test (101).
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

SHARD_IDS=(1 101)

for ID in "${SHARD_IDS[@]}"; do
    echo "===== gen traj $ID (shard 1) ====="
    "$BLENDER" --background --python blender/generate_data.py -- \
        --trajectory_id "$ID" \
        --output_dir data/raw \
        --skip_if_exists
done
