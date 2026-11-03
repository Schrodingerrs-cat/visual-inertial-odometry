#!/bin/bash
#SBATCH --job-name=step_j_eval
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/step_j_eval.out
#SBATCH --error=slurm/logs/step_j_eval.err

# Step J eval: V/I/VI on test_ids = [305, 306, 307, 300]. Produces per-method
# per-traj plots AND per-traj V/I/VI comparison overlays in
# outputs/step_j/eval/plots/.

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

python scripts/evaluate_tdirect.py \
    --config configs/step_j.yaml \
    --checkpoints_dir outputs/step_j \
    --split test \
    --methods v i vi \
    --skip_missing

RESULTS=outputs/step_j/results.csv
echo "===== Step J eval results (results.csv) ====="
if [[ -f "$RESULTS" ]]; then
    cat "$RESULTS"
    echo ""
    echo "===== Headline rows (ate_aligned only) ====="
    grep -E ',ate_aligned,' "$RESULTS" || true
    echo ""
    echo "===== Per-method averages across test trajs (ate_aligned) ====="
    for m in v i vi; do
        avg=$(awk -F, -v m="$m" '
            $1==m && $3=="ate_aligned" { s+=$4; n+=1 }
            END { if (n>0) printf "%.3f", s/n; else print "n/a" }' "$RESULTS")
        echo "$m: avg ate_aligned = $avg m"
    done
    echo ""
    echo "===== L8 cross-check: I on val/minitest (305) should be sane (no NaN) ====="
    grep -E '^i,305,ate_aligned,' "$RESULTS" || echo "  (no row found)"
else
    echo "WARNING: $RESULTS not found"
fi

echo ""
echo "===== Comparison plot files ====="
ls outputs/step_j/eval/plots/comparison_* 2>/dev/null || echo "  (no comparison plots found)"
