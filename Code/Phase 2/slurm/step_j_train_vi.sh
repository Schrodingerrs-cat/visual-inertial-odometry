#!/bin/bash
#SBATCH --job-name=step_j_vi
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=14:00:00
#SBATCH --output=slurm/logs/step_j_train_vi.out
#SBATCH --error=slurm/logs/step_j_train_vi.err

# Step J VI: NetworkVI_tdirect from scratch on simple shapes, warm-started
# from the freshly-trained Step J V and I. stage2_lr_mult=0.0 → V and I
# stay frozen across both stages (lesson L1 from Step I audit).

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

if [[ ! -f outputs/step_j/v/best.pt ]]; then
    echo "ERROR: outputs/step_j/v/best.pt missing — Step J V training did not finish." >&2
    exit 1
fi
if [[ ! -f outputs/step_j/i/best.pt ]]; then
    echo "ERROR: outputs/step_j/i/best.pt missing — Step J I training did not finish." >&2
    exit 1
fi

python scripts/train_vi_tdirect.py \
    --config configs/step_j.yaml \
    --pretrained_v outputs/step_j/v/best.pt \
    --pretrained_i outputs/step_j/i/best.pt
