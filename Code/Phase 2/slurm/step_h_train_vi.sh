#!/bin/bash
#SBATCH --job-name=step_h_vi
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=14:00:00
#SBATCH --output=slurm/logs/step_h_train_vi.out
#SBATCH --error=slurm/logs/step_h_train_vi.err

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

# Pretrained V: freshly trained t-direct V from this Step H run.
# Pretrained I: symlink at outputs/step_h/i -> ../step_g/i (NetworkI ckpt
#               is variant-agnostic so the Step G I checkpoint loads cleanly).
python scripts/train_vi_tdirect.py \
    --config configs/step_h.yaml \
    --pretrained_v outputs/step_h/v/best.pt \
    --pretrained_i outputs/step_h/i/best.pt
