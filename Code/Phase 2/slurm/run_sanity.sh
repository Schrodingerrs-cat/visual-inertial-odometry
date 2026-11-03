#!/bin/bash
#SBATCH --job-name=p2_sanity
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/sanity.out
#SBATCH --error=slurm/logs/sanity.err

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

python scripts/run_sanity_checks.py \
    --data_dir data/raw \
    --config configs/default.yaml \
    --soft_fail motion_histogram,motion_distribution
