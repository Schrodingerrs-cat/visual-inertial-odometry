#!/bin/bash
#SBATCH --job-name=p2_ablation
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/ablation.out
#SBATCH --error=slurm/logs/ablation.err

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

python scripts/ablation_study.py --config configs/default.yaml
