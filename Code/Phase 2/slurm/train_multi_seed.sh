#!/bin/bash
#SBATCH --job-name=p2_multiseed
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/multiseed.out
#SBATCH --error=slurm/logs/multiseed.err

# Run scripts/train_multi_seed.py sequentially for the requested target and
# seeds in a single SBATCH job (so a 3-seed sweep uses one of the cluster's
# concurrent-job slots, not three).
#
# Invocation:
#   TARGET=vi SEEDS="1 2 3" sbatch slurm/train_multi_seed.sh
#
# Optional:
#   PRETRAINED_V=outputs/v/best.pt PRETRAINED_I=outputs/i/best.pt
#       TARGET=vi SEEDS="1 2 3" sbatch slurm/train_multi_seed.sh

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

: "${TARGET:?set TARGET=v|i|vi}"
: "${SEEDS:?set SEEDS=\"s1 s2 s3\"}"
PRETRAINED_V="${PRETRAINED_V:-}"
PRETRAINED_I="${PRETRAINED_I:-}"

EXTRA=()
if [[ -n "$PRETRAINED_V" ]]; then EXTRA+=(--pretrained_v "$PRETRAINED_V"); fi
if [[ -n "$PRETRAINED_I" ]]; then EXTRA+=(--pretrained_i "$PRETRAINED_I"); fi

python scripts/train_multi_seed.py --target "$TARGET" --seeds $SEEDS "${EXTRA[@]}"
