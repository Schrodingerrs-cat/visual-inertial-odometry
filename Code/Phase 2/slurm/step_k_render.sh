#!/bin/bash
#SBATCH --job-name=step_k_render
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/step_k_render.out
#SBATCH --error=slurm/logs/step_k_render.err

# Step K render: 13 trajectories at 30 s × 1000 Hz, EEVEE taa=64, 640×480.
# Per-traj wall-clock at this resolution is ~5 min on rtx_pro_6000_b →
# ~65 min for the batch. 2 h cap is ~2× safety margin. All on tex_03.webp
# into data/raw_g/. Earlier-step trajs (0..307) are NOT re-rendered
# (--skip_if_exists guards each call individually).
#
# Trajectories rendered (Step K speed-diverse):
#   312-319 train (4 shapes × 2 speeds): circle×2, figure8×2, diamond×2,
#                                         zigzag×2
#   320     val   (ellipse @ 0.3 m/s — drives best.pt selection)
#   321-323 test  (ellipse fast, triangle mid + fast)
#   324     test  (NEW shape: simple_sinusoid_30s @ 0.5 m/s)

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

mkdir -p data/raw_g

STEP_K_IDS=(312 313 314 315 316 317 318 319 320 321 322 323 324)

for ID in "${STEP_K_IDS[@]}"; do
    echo "===== gen traj $ID (Step K; --force_texture tex_03.webp) ====="
    "$BLENDER" --background --python blender/generate_data.py -- \
        --trajectory_id "$ID" \
        --output_dir data/raw_g \
        --force_texture tex_03.webp \
        --skip_if_exists
done

echo ""
echo "===== Step K render complete — per-traj frame counts ====="
for ID in "${STEP_K_IDS[@]}"; do
    DIR="data/raw_g/traj_0$ID"
    if [[ -d "$DIR" ]]; then
        n=$(ls "$DIR/frames/" 2>/dev/null | wc -l)
        # Also surface speed metadata so a quick eyeball confirms scaling.
        TPS=$(python3 -c "
import json, sys
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
