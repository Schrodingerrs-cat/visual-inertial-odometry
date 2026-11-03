#!/bin/bash
#SBATCH --job-name=p2_motion_mag
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=20:00:00
#SBATCH --output=slurm/logs/motion_magnitude.out
#SBATCH --error=slurm/logs/motion_magnitude.err

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

python scripts/motion_magnitude_study.py --config configs/default.yaml
