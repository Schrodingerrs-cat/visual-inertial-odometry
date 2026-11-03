#!/bin/bash
#SBATCH --job-name=step_s_vi
#SBATCH --partition=long
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/step_s_train_vi.out
#SBATCH --error=slurm/logs/step_s_train_vi.err

# Step S VI: NetworkVI_tdirect (same as Steps J/K), warm-started from
# Step S V (outputs/step_s/v/best.pt) and Step S I (outputs/step_s/i/
# best.pt). NOT Step K's checkpoints — Step S is a fresh V/I/VI trio
# top-to-bottom; the only thing inherited from Step K is the
# architecture and hyperparameters.
#
# stage2_lr_mult=0.0 (lesson L1) freezes both backbones across both
# stages — the residual refiner is what trains.

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

if [[ ! -f outputs/step_s/v/best.pt ]]; then
    echo "ERROR: outputs/step_s/v/best.pt missing — Step S V training did not finish." >&2
    exit 1
fi
if [[ ! -f outputs/step_s/i/best.pt ]]; then
    echo "ERROR: outputs/step_s/i/best.pt missing — Step S I training did not finish." >&2
    exit 1
fi

python scripts/train_vi_tdirect.py \
    --config configs/step_s.yaml \
    --pretrained_v outputs/step_s/v/best.pt \
    --pretrained_i outputs/step_s/i/best.pt
