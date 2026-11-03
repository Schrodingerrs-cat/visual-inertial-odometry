#!/bin/bash
#SBATCH --job-name=step_j_render
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/step_j_render.out
#SBATCH --error=slurm/logs/step_j_render.err

# Step J render: 7 simple-shape trajectories at 30 s × 1000 Hz (~10 min/traj
# at EEVEE taa=64 → ~70 min total + buffer). All on tex_03.webp into
# data/raw_g/. Trajs 0..3, 100..102, 200, 300 are NOT re-rendered
# (--skip_if_exists).

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

mkdir -p data/raw_g

for ID in 301 302 303 304 305 306 307; do
    echo "===== gen traj $ID (Step J; --force_texture tex_03.webp) ====="
    "$BLENDER" --background --python blender/generate_data.py -- \
        --trajectory_id "$ID" \
        --output_dir data/raw_g \
        --force_texture tex_03.webp \
        --skip_if_exists
done

echo "===== Step J render complete ====="
for ID in 301 302 303 304 305 306 307; do
    if [[ -d "data/raw_g/traj_0$ID" ]]; then
        n=$(ls "data/raw_g/traj_0$ID/frames/" 2>/dev/null | wc -l)
        echo "traj_0$ID: $n frames"
    else
        echo "traj_0$ID: MISSING"
    fi
done
