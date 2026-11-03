#!/bin/bash
#SBATCH --job-name=p2_train_i
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=10:00:00
#SBATCH --output=slurm/logs/train_i.out
#SBATCH --error=slurm/logs/train_i.err

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

python scripts/train_i.py --config configs/default.yaml
