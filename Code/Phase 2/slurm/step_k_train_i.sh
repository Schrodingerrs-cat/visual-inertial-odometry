#!/bin/bash
#SBATCH --job-name=step_k_i
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=10:00:00
#SBATCH --output=slurm/logs/step_k_train_i.out
#SBATCH --error=slurm/logs/step_k_train_i.err

# Step K I — FALLBACK SCRIPT (NOT IN THE DEFAULT SUBMIT DAG).
#
# The recommended path is to symlink Step J's I instead:
#     ln -sfn ../step_j/i outputs/step_k/i
#     ls -L outputs/step_k/i/best.pt   # verify symlink resolves (L10)
#
# Step J's I was trained on the same simple-shape data and is unaffected
# by RAFT distillation (which only modifies V's loss). Reusing it saves
# ~6 h of compute and gives a clean Step J ↔ Step K controlled comparison.
#
# This script exists as a fallback in case the symlink is undesired
# (e.g. you want a per-step retrain for variance estimation). To use,
# launch with: sbatch slurm/step_k_train_i.sh

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

for ID in 312 313 314 315 316 317 318 319 320; do
    if [[ ! -d "data/raw_g/traj_0$ID" ]]; then
        echo "ERROR: data/raw_g/traj_0$ID missing." >&2
        exit 1
    fi
done

python scripts/train_i.py --config configs/step_k.yaml
