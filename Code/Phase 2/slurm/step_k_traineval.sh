#!/bin/bash
#SBATCH --job-name=step_k_traineval
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/step_k_traineval.out
#SBATCH --error=slurm/logs/step_k_traineval.err

# Step K — eval Step K's trained V/I/VI checkpoints on the *training*
# trajs (312-319). Produces per-traj pred-vs-GT comparison plots
# (comparison_{tid}_xy.png) for visualising how each method fits the data
# it was trained on.
#
# Pre-req: data/raw_g/traj_03{12..19} must already be rendered (they were
# created by slurm/step_k_render.sh).
#
# Output:
#   outputs/step_k_traineval/results.csv
#   outputs/step_k_traineval/eval/trajectories/{gt,v,i,vi}_3{12..19}.tum
#   outputs/step_k_traineval/eval/plots/comparison_3{12..19}_xy.png  ← KEY

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

# Pre-flight: confirm Step K's checkpoints exist.
echo "===== Pre-flight: Step K checkpoints ====="
for m in v i vi; do
    ckpt="outputs/step_k/${m}/best.pt"
    if [[ -f "$ckpt" ]]; then
        echo "  $ckpt: present"
    else
        echo "  $ckpt: MISSING"
    fi
done
echo ""

# Pre-flight: confirm training trajs are rendered.
echo "===== Pre-flight: training trajs rendered ====="
for tid in 312 313 314 315 316 317 318 319; do
    dir=$(printf "data/raw_g/traj_%04d" "$tid")
    if [[ -d "$dir/frames" ]]; then
        n_frames=$(ls "$dir/frames" 2>/dev/null | wc -l)
        echo "  $dir: $n_frames frames"
    else
        echo "  $dir: MISSING"
    fi
done
echo ""

python scripts/evaluate_tdirect.py \
    --config configs/step_k_traineval.yaml \
    --checkpoints_dir outputs/step_k \
    --split test \
    --methods v i vi \
    --skip_missing

RESULTS=outputs/step_k_traineval/results.csv
echo ""
echo "===== Train-eval results.csv ====="
if [[ -f "$RESULTS" ]]; then
    cat "$RESULTS"
    echo ""
    echo "===== Per-method per-traj ate_aligned ====="
    grep -E ',ate_aligned,' "$RESULTS" || true

    echo ""
    echo "===== V on each training traj ====="
    for tid in 312 313 314 315 316 317 318 319; do
        v_ate=$(awk -F, -v t="$tid" '$1=="v" && $2==t && $3=="ate_aligned" {print $4}' "$RESULTS")
        v_seg=$(awk -F, -v t="$tid" '$1=="v" && $2==t && $3=="seg2s_median" {print $4}' "$RESULTS")
        case "$tid" in
            312) label="circle    @ 0.10 m/s";;
            313) label="circle    @ 0.40 m/s";;
            314) label="figure8   @ 0.15 m/s";;
            315) label="figure8   @ 0.50 m/s";;
            316) label="diamond   @ 0.20 m/s";;
            317) label="diamond   @ 0.40 m/s";;
            318) label="zigzag    @ 0.20 m/s";;
            319) label="zigzag    @ 0.50 m/s";;
        esac
        echo "  traj $tid ($label): V ate=${v_ate:-n/a}m  seg2s=${v_seg:-n/a}m"
    done

    echo ""
    echo "===== Per-method averages over training set (312-319) ====="
    for m in v i vi; do
        avg=$(awk -F, -v m="$m" '
            $1==m && $3=="ate_aligned" { s+=$4; n+=1 }
            END { if (n>0) printf "%.3f", s/n; else print "n/a" }' "$RESULTS")
        echo "  $m: avg ate_aligned = $avg m"
    done
else
    echo "WARNING: $RESULTS not found"
fi

echo ""
echo "===== Comparison plots ====="
ls outputs/step_k_traineval/eval/plots/comparison_*_xy.png 2>/dev/null \
    || echo "  (no comparison plots found)"
