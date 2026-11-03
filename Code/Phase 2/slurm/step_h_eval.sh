#!/bin/bash
#SBATCH --job-name=step_h_eval
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/step_h_eval.out
#SBATCH --error=slurm/logs/step_h_eval.err

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

python scripts/evaluate_tdirect.py \
    --config configs/step_h.yaml \
    --checkpoints_dir outputs/step_h \
    --split test \
    --methods v i vi \
    --skip_missing

echo "===== Step H eval results (results.csv) ====="
RESULTS=outputs/step_h/results.csv
if [[ -f "$RESULTS" ]]; then
    cat "$RESULTS"
    echo ""
    echo "===== Headline numbers ====="
    grep -E '^(v|i|vi),3,(ate_aligned|endpoint)' "$RESULTS" || true
    echo ""
    echo "===== Cross-check: I_traj_3 should equal Step G's 1.78 m ====="
    grep -E '^i,3,ate_aligned' "$RESULTS" || true
else
    echo "WARNING: $RESULTS not found"
fi
