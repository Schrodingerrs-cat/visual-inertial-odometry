#!/bin/bash
#SBATCH --job-name=step_s_render
#SBATCH --partition=long
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/step_s_render.out
#SBATCH --error=slurm/logs/step_s_render.err

# Step S render: 13 trajectories (400-408 + 410-413) at 30 s × 1000 Hz,
# EEVEE taa=64, 640×480, all on tex_03.webp into data/raw_g/. Per-traj
# wall-clock ~5 min on rtx_pro_6000_b → ~65 min for the batch. 2 h cap is
# ~2× safety margin. Earlier-step trajs are NOT re-rendered
# (--skip_if_exists guards each call individually).
#
# Renders 13 trajs through the corrected IMU pipeline (Option B + coordinated
# turn + R-chain ω + carry-forward fallback, eps=1e-3 m/s). Locked speeds:
#   400 simple_circle  @ 0.5  (train)
#   401 simple_circle  @ 1.2  (train)
#   402 simple_figure8 @ 0.5  (train)
#   403 simple_figure8 @ 1.0  (train)
#   404 simple_diamond @ 0.8  (train)
#   405 simple_diamond @ 1.05 (train; plane-fit ceiling on seed 7006)
#   406 simple_zigzag  @ 0.8  (train)
#   407 simple_zigzag  @ 1.0  (train)
#   408 simple_ellipse @ 1.0  (val)
#   410 simple_ellipse @ 0.3  (test, OOD-slow)
#   411 simple_ellipse @ 0.5  (test)
#   412 simple_ellipse @ 0.8  (test)
#   413 simple_ellipse @ 1.2  (test, train ceiling)

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

mkdir -p data/raw_g

STEP_S_IDS=(400 401 402 403 404 405 406 407 408 410 411 412 413)

for ID in "${STEP_S_IDS[@]}"; do
    echo "===== gen traj $ID (Step S; --force_texture tex_03.webp) ====="
    "$BLENDER" --background --python blender/generate_data.py -- \
        --trajectory_id "$ID" \
        --output_dir data/raw_g \
        --force_texture tex_03.webp \
        --skip_if_exists
done

echo ""
echo "===== Step S render complete — per-traj frame counts + speed metadata ====="
for ID in "${STEP_S_IDS[@]}"; do
    DIR=$(printf "data/raw_g/traj_%04d" "$ID")
    if [[ -d "$DIR" ]]; then
        n=$(ls "$DIR/frames/" 2>/dev/null | wc -l)
        TPS=$(python3 -c "
import json
m = json.load(open('$DIR/metadata.json'))
tps = m.get('target_peak_speed')
emp = m.get('empirical_peak_velocity')
err = m.get('speed_target_relative_error')
print(f'tps={tps} emp={emp} rel_err={err}')
" 2>/dev/null)
        echo "$DIR: $n frames    $TPS"
    else
        echo "$DIR: MISSING"
    fi
done
