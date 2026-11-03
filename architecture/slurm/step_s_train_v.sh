#!/bin/bash
#SBATCH --job-name=step_s_v
#SBATCH --partition=long
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=10:00:00
#SBATCH --output=slurm/logs/step_s_train_v.out
#SBATCH --error=slurm/logs/step_s_train_v.err

# Step S V: NetworkV_tdirect (VGG8 backbone, t-direct head — same as
# Steps J/K) trained from scratch with TDirectPoseLoss. NO RAFT
# distillation — Step K confirmed RAFT does not engage on this scene
# type (>99.6% FB rejection), so Step S uses the no-distill tdirect
# path (train_v_tdirect.py, NOT train_v_distill.py). This preserves
# the J/K-validated head architecture while isolating the speed-bucket
# change as the sole independent variable.

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

# Defensive checks: training data dirs (train + val) must exist.
TRAIN_AND_VAL=(400 401 402 403 404 405 406 407 408)
for ID in "${TRAIN_AND_VAL[@]}"; do
    DIR=$(printf "data/raw_g/traj_%04d" "$ID")
    if [[ ! -d "$DIR" ]]; then
        echo "ERROR: $DIR missing — Step S render did not run." >&2
        exit 1
    fi
done

python scripts/train_v_tdirect.py --config configs/step_s.yaml
