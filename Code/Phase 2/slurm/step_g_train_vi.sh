#!/bin/bash
#SBATCH --job-name=step_g_vi
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=14:00:00
#SBATCH --output=slurm/logs/step_g_train_vi.out
#SBATCH --error=slurm/logs/step_g_train_vi.err

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

python scripts/train_vi.py \
    --config configs/step_g.yaml \
    --pretrained_v outputs/step_g/v/best.pt \
    --pretrained_i outputs/step_g/i/best.pt
