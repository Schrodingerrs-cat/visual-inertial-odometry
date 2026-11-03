#!/bin/bash
#SBATCH --job-name=step_s_i
#SBATCH --partition=long
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=10:00:00
#SBATCH --output=slurm/logs/step_s_train_i.out
#SBATCH --error=slurm/logs/step_s_train_i.err

# Step S I: vanilla NetworkI (LSTM hidden=128, 2 layers, 10 ms IMU window)
# trained from scratch with original IMUPoseLoss. NOT IMUPoseLoss_hover
# (Phase 1 hover masking did not help on the Step K within-family test
# set; revert for a clean diagnostic comparison) and NOT the Phase 2
# 100 ms causal context (data.imu_context_length absent → dataloader
# emits Phase 0 length-10 batch["imu"] tensor).
#
# Loss class is selected by configs/step_s.yaml: loss.imu_use_hover=false.
# train_i.py prints the resolved loss class on startup; verify before
# letting the run go to completion.

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

TRAIN_AND_VAL=(400 401 402 403 404 405 406 407 408)
for ID in "${TRAIN_AND_VAL[@]}"; do
    DIR=$(printf "data/raw_g/traj_%04d" "$ID")
    if [[ ! -d "$DIR" ]]; then
        echo "ERROR: $DIR missing — Step S render did not run." >&2
        exit 1
    fi
done

python scripts/train_i.py --config configs/step_s.yaml
