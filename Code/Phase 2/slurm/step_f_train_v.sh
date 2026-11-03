#!/bin/bash
#SBATCH --job-name=step_f_v
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/step_f_train_v.out
#SBATCH --error=slurm/logs/step_f_train_v.err

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

python scripts/train_v.py --config configs/step_f.yaml
