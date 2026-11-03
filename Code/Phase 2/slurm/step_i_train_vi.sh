#!/bin/bash
#SBATCH --job-name=step_i_vi
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=14:00:00
#SBATCH --output=slurm/logs/step_i_train_vi.out
#SBATCH --error=slurm/logs/step_i_train_vi.err

# Step I VI training: NetworkVI_anchored trained from scratch on data/raw_g.
# Two-stage schedule (10 + 20 epochs) identical to Step H, but the model is
# different — it has two new gate MLPs (gate_t, gate_R) initialised neutral
# (sigmoid output 0.5 at t=0) and learns from L_gate_supervision.
#
# Pretrained branches:
#   --pretrained_v outputs/step_i/v/best.pt   → symlink → step_h/v/best.pt
#   --pretrained_i outputs/step_i/i/best.pt   → symlink → step_g/i/best.pt
#
# Sanity-check symlinks BEFORE this script runs (the submit sequence in
# CHANGES.md prints the verification command). If either symlink is broken
# we'd silently train on randomly-initialised V and I, and the gate-
# supervision targets would be meaningless.

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

# Defensive checks at job start — fail fast with a clear message rather than
# silently train on missing branches.
if [[ ! -f outputs/step_i/v/best.pt ]]; then
    echo "ERROR: outputs/step_i/v/best.pt is missing. Did the symlink fail?" >&2
    echo "Expected: outputs/step_i/v -> ../step_h/v (with step_h/v/best.pt)" >&2
    exit 1
fi
if [[ ! -f outputs/step_i/i/best.pt ]]; then
    echo "ERROR: outputs/step_i/i/best.pt is missing. Did the symlink fail?" >&2
    echo "Expected: outputs/step_i/i -> ../step_g/i (with step_g/i/best.pt)" >&2
    exit 1
fi
if [[ ! -d data/raw_g/traj_0300 ]]; then
    echo "ERROR: data/raw_g/traj_0300/ is missing — render step did not run." >&2
    exit 1
fi

python scripts/train_vi_anchored.py \
    --config configs/step_i.yaml \
    --pretrained_v outputs/step_i/v/best.pt \
    --pretrained_i outputs/step_i/i/best.pt
