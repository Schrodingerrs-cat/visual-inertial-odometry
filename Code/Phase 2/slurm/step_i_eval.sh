#!/bin/bash
#SBATCH --job-name=step_i_eval
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/step_i_eval.out
#SBATCH --error=slurm/logs/step_i_eval.err

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

python scripts/evaluate_anchored.py \
    --config configs/step_i.yaml \
    --checkpoints_dir outputs/step_i \
    --split test \
    --methods v i vi \
    --skip_missing

echo "===== Step I eval results (results.csv) ====="
RESULTS=outputs/step_i/results.csv
if [[ -f "$RESULTS" ]]; then
    cat "$RESULTS"
    echo ""
    echo "===== Headline numbers ====="
    grep -E '^(v|i|vi),(3|300),(ate_aligned|endpoint)' "$RESULTS" || true
    echo ""
    echo "===== Cross-check: I_traj_3 should equal Step G's 1.78 m ====="
    grep -E '^i,3,ate_aligned' "$RESULTS" || true
    echo ""
    echo "===== Cross-check: I_traj_300 should be in 0.5-2 m range ====="
    grep -E '^i,300,ate_aligned' "$RESULTS" || true
else
    echo "WARNING: $RESULTS not found"
fi

GATE=outputs/step_i/eval/gate_stats.json
echo ""
echo "===== Gate behaviour summary (gate_stats.json) ====="
if [[ -f "$GATE" ]]; then
    cat "$GATE"
else
    echo "WARNING: $GATE not found"
fi
