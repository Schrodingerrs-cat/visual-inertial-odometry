#!/bin/bash
#SBATCH --job-name=step_g_eval
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/step_g_eval.out
#SBATCH --error=slurm/logs/step_g_eval.err

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

python scripts/evaluate.py \
    --config configs/step_g.yaml \
    --checkpoints_dir outputs/step_g \
    --split test \
    --methods v i vi \
    --skip_missing

echo "===== Step G eval results (results.csv) ====="
RESULTS=outputs/step_g/results.csv
if [[ -f "$RESULTS" ]]; then
    cat "$RESULTS"
    echo ""
    echo "===== Headline numbers ====="
    grep -E '^(v|i|vi),3,(ate_aligned|endpoint)' "$RESULTS" || true
else
    echo "WARNING: $RESULTS not found"
fi
