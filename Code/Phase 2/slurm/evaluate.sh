#!/bin/bash
#SBATCH --job-name=p2_evaluate
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/evaluate.out
#SBATCH --error=slurm/logs/evaluate.err

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

python scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoints_dir outputs/ \
    --split test \
    --skip_missing
