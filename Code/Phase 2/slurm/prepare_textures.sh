#!/bin/bash
#SBATCH --job-name=p2_texprep
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000
#SBATCH --time=00:05:00
#SBATCH --output=slurm/logs/prepare_textures.out
#SBATCH --error=slurm/logs/prepare_textures.err

# Normalize the 7 user-provided texture images in Data/textures_raw/
# into Data/textures_processed/ and split into train (5) / test (2) pools.
# CPU-only; runs in seconds.

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

python scripts/prepare_textures.py
