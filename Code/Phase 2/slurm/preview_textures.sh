#!/bin/bash
#SBATCH --job-name=p2_preview
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/preview.out
#SBATCH --error=slurm/logs/preview.err

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

"$BLENDER" --background --python scripts/preview_textures.py
