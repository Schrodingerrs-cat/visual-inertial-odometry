#!/bin/bash
#SBATCH --job-name=step_k_v
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/step_k_train_v.out
#SBATCH --error=slurm/logs/step_k_train_v.err

# Step K V: NetworkV_tdirect trained from scratch with RAFT distillation.
# Architecture identical to Step J's V; only the loss function and the
# training distribution change (8 speed-diverse train trajs instead of
# Step J's 4).

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

# L9 + R3: defensive checks. Training data dirs must exist AND every traj
# (train + val) must have a fresh raft_predictions.npz cache. The dataloader
# does the hash check at startup; this is a faster pre-check.
TRAIN_AND_VAL=(312 313 314 315 316 317 318 319 320)
for ID in "${TRAIN_AND_VAL[@]}"; do
    if [[ ! -d "data/raw_g/traj_0$ID" ]]; then
        echo "ERROR: data/raw_g/traj_0$ID missing — render did not run." >&2
        exit 1
    fi
    if [[ ! -f "data/raw_g/traj_0$ID/raft_predictions.npz" ]]; then
        echo "ERROR: data/raw_g/traj_0$ID/raft_predictions.npz missing — " >&2
        echo "       run slurm/step_k_precompute_raft.sh first." >&2
        exit 1
    fi
done
# Minitest traj 322 must also exist (Step K reads frames during train).
if [[ ! -d "data/raw_g/traj_0322" ]]; then
    echo "ERROR: data/raw_g/traj_0322 missing (minitest traj) — render did not run." >&2
    exit 1
fi

python scripts/train_v_distill.py --config configs/step_k.yaml
