#!/bin/bash
#SBATCH --job-name=step_j_i
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=10:00:00
#SBATCH --output=slurm/logs/step_j_train_i.out
#SBATCH --error=slurm/logs/step_j_train_i.err

# Step J I: NetworkI from scratch on simple shapes.

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

for ID in 301 302 303 304 305; do
    if [[ ! -d "data/raw_g/traj_0$ID" ]]; then
        echo "ERROR: data/raw_g/traj_0$ID missing — render did not run." >&2
        exit 1
    fi
done

python scripts/train_i.py --config configs/step_j.yaml
