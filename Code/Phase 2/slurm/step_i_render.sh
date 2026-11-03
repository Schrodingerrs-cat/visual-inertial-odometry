#!/bin/bash
#SBATCH --job-name=step_i_render
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/step_i_render.out
#SBATCH --error=slurm/logs/step_i_render.err

# Step I render: render the new simple_figure8 trajectory (id 300) into
# data/raw_g/, the same data dir as Step G/H. We force the same texture
# (tex_03.webp) used by trajs 0..3 so V's photometric statistics match
# what it saw at training time (Step H V is reused via symlink).
#
# Wallclock budget: 30 s × 100 Hz / RENDER_STRIDE 10 = 3001 frames at
# 640×480 EEVEE taa=64. The Step G shard0 numbers gave ~0.2 s/frame on
# this GPU class, so ~10 minutes total. 1 h cap is 6× safety.

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

mkdir -p data/raw_g

ID=300
echo "===== gen traj $ID (Step I; --force_texture tex_03.webp) ====="
"$BLENDER" --background --python blender/generate_data.py -- \
    --trajectory_id "$ID" \
    --output_dir data/raw_g \
    --force_texture tex_03.webp \
    --skip_if_exists

echo "===== Step I render complete ====="
ls -la data/raw_g/traj_0300/ 2>/dev/null || echo "WARNING: traj_0300 not found"
