#!/bin/bash
#SBATCH --job-name=step_g_render
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/step_g_render.out
#SBATCH --error=slurm/logs/step_g_render.err

# Step G render: re-render trajs 0, 1, 2, 3 each with tex_03.webp into
# data/raw_g/. Same trajectory seeds as the originals (poses_full.npy and
# imu.npy will be byte-identical to data/raw/traj_000{0..3}/), only the PNGs
# in frames/ change because the rendered texture is forced to tex_03.
#
# Why tex_03 specifically: this preserves traj 3's plane geometry exactly as
# in Step F (28×18.69 m), so I_traj_3 should still ≈3.41 m on raw_g —
# a clean cross-check that the re-render didn't break IMU/poses. Trajs 0/1/2
# get new motion patterns due to their different planes (tex_00..02 had
# different aspect ratios), but they're training data and V learns from
# whatever it sees.
#
# Time budget: ~30-45 min per 100 s trajectory at 640×480 EEVEE taa=64
# (per gen_data_shard0 comment). 4 trajectories sequential ≈ 2-3 h. 8 h cap
# leaves >2× safety.

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

mkdir -p data/raw_g

for ID in 0 1 2 3; do
    echo "===== gen traj $ID (Step G; --force_texture tex_03.webp) ====="
    "$BLENDER" --background --python blender/generate_data.py -- \
        --trajectory_id "$ID" \
        --output_dir data/raw_g \
        --force_texture tex_03.webp \
        --skip_if_exists
done

echo "===== Step G render complete ====="
ls -la data/raw_g/
