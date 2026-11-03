#!/bin/bash
#SBATCH --job-name=step_k_ext_render
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/step_k_extended_render.out
#SBATCH --error=slurm/logs/step_k_extended_render.err

# Step K extended render: 4 held-out within-family trajectories at
# intermediate (unseen) speeds. ~5 min per traj on rtx_pro_6000_b →
# ~20 min batch wall-clock. 1 h cap is ~3× safety margin.
#
# Trajectories rendered:
#   325 simple_figure8 @ 0.30 m/s (seed 6001)
#   326 simple_circle  @ 0.25 m/s (seed 6002)
#   327 simple_diamond @ 0.30 m/s (seed 6003)
#   328 simple_zigzag  @ 0.30 m/s (seed 6004)
#
# Trajs are rendered with --force_texture tex_03.webp into data/raw_g/
# so they share the texture used by Step K's training set.

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

mkdir -p data/raw_g

STEP_K_EXT_IDS=(325 326 327 328)

for ID in "${STEP_K_EXT_IDS[@]}"; do
    echo "===== gen traj $ID (Step K extended; --force_texture tex_03.webp) ====="
    "$BLENDER" --background --python blender/generate_data.py -- \
        --trajectory_id "$ID" \
        --output_dir data/raw_g \
        --force_texture tex_03.webp \
        --skip_if_exists
done

echo ""
echo "===== Step K extended render complete — per-traj frame counts ====="
for ID in "${STEP_K_EXT_IDS[@]}"; do
    DIR="data/raw_g/traj_0$ID"
    if [[ -d "$DIR" ]]; then
        n=$(ls "$DIR/frames/" 2>/dev/null | wc -l)
        TPS=$(python3 -c "
import json
m=json.load(open('$DIR/metadata.json'))
tps=m.get('target_peak_speed')
emp=m.get('empirical_peak_velocity')
err=m.get('speed_target_relative_error')
print(f'tps={tps} emp={emp} rel_err={err}')
" 2>/dev/null)
        echo "traj_0$ID: $n frames    $TPS"
    else
        echo "traj_0$ID: MISSING"
    fi
done
