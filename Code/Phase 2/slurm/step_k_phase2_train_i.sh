#!/bin/bash
#SBATCH --job-name=step_k_p2_i
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/step_k_phase2_train_i.out
#SBATCH --error=slurm/logs/step_k_phase2_train_i.err

# Phase 2: retrain Network I with the new 100-sample causal IMU context
# window (data.imu_context_length=100 in configs/step_k_phase2.yaml).
# Loss is IMUPoseLoss_hover from Phase 1; architecture (2-layer LSTM,
# hidden 128) is unchanged. V and VI are NOT retouched.

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

# Pre-flight: confirm training trajs are rendered.
echo "===== Pre-flight: training trajs rendered ====="
for ID in 312 313 314 315 316 317 318 319 320 322; do
    if [[ ! -d "data/raw_g/traj_0$ID" ]]; then
        echo "ERROR: data/raw_g/traj_0$ID missing." >&2
        exit 1
    fi
done
echo "  all training/val/minitest trajs present in data/raw_g/"

mkdir -p outputs/step_k_phase2

python scripts/train_i.py --config configs/step_k_phase2.yaml

echo ""
echo "===== Phase 2 I checkpoint ====="
ls -la outputs/step_k_phase2/i/best.pt outputs/step_k_phase2/i/last.pt 2>/dev/null || true
echo ""
echo "===== Last 5 epoch rows of train.log ====="
tail -5 outputs/step_k_phase2/i/train.log 2>/dev/null || echo "(train.log not found)"
