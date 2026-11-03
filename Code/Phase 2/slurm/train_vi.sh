#!/bin/bash
#SBATCH --job-name=p2_train_vi
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=14:00:00
#SBATCH --output=slurm/logs/train_vi.out
#SBATCH --error=slurm/logs/train_vi.err

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

python scripts/train_vi.py \
    --config configs/default.yaml \
    --pretrained_v outputs/v/best.pt \
    --pretrained_i outputs/i/best.pt
