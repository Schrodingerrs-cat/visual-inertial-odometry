#!/bin/bash
#SBATCH --job-name=step_k_vi
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=14:00:00
#SBATCH --output=slurm/logs/step_k_train_vi.out
#SBATCH --error=slurm/logs/step_k_train_vi.err

# Step K VI: NetworkVI_tdirect, warm-started from Step K's distilled V and
# Step J's I (symlinked). VI itself does NOT use distillation — only V does.
# stage2_lr_mult=0.0 (lesson L1) freezes both backbones across both stages.

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

# L10: symlink integrity. ``ls -L`` resolves through symlinks; if the target
# is missing, the test fails even though the link itself exists.
if [[ ! -e outputs/step_k/v/best.pt ]] || ! ls -L outputs/step_k/v/best.pt > /dev/null 2>&1; then
    echo "ERROR: outputs/step_k/v/best.pt missing or symlink broken." >&2
    echo "       Step K V training did not finish, or symlink target is gone." >&2
    exit 1
fi
if [[ ! -e outputs/step_k/i/best.pt ]] || ! ls -L outputs/step_k/i/best.pt > /dev/null 2>&1; then
    echo "ERROR: outputs/step_k/i/best.pt missing or symlink broken." >&2
    echo "       Expected: outputs/step_k/i symlinked to ../step_j/i" >&2
    echo "       (or run slurm/step_k_train_i.sh as a fallback)." >&2
    exit 1
fi

python scripts/train_vi_tdirect.py \
    --config configs/step_k.yaml \
    --pretrained_v outputs/step_k/v/best.pt \
    --pretrained_i outputs/step_k/i/best.pt
