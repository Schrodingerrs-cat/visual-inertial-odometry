#!/bin/bash
#SBATCH --job-name=step_j_crossfamily_eval
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/step_j_crossfamily_eval.out
#SBATCH --error=slurm/logs/step_j_crossfamily_eval.err

# Step J cross-family eval. Re-uses outputs/step_j/{v,i,vi}/best.pt and runs
# evaluate_tdirect.py on test_ids = [3, 100, 101, 102]:
#
#   traj 3    — sinusoid, tex_03 (Step F/G traj that Step H's V scored 3.86 m on)
#   traj 100  — flat_spiral, tex_04 (held-out texture + shape)
#   traj 101  — figure_8 (different parametric form), tex_04 (held-out texture)
#   traj 102  — clover, tex_00 (seen-pool texture, unseen by Step J's V which
#                                only trained on tex_03)
#
# Reads Step J's trained checkpoints from outputs/step_j/. Writes a fresh
# results.csv to outputs/step_j_crossfamily/results.csv (does NOT touch
# outputs/step_j/results.csv).

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

# Pre-flight: confirm Step J's checkpoints exist where we expect them. The
# eval will skip missing methods (--skip_missing) but it's nicer to fail loud
# here if all three are absent.
echo "===== Pre-flight: Step J checkpoints ====="
for m in v i vi; do
    ckpt="outputs/step_j/${m}/best.pt"
    if [[ -f "$ckpt" ]]; then
        echo "  $ckpt: present ($(stat -c %s "$ckpt") bytes)"
    else
        echo "  $ckpt: MISSING"
    fi
done
echo ""

# Pre-flight: confirm test trajectories exist in data/raw/.
echo "===== Pre-flight: test trajectories in data/raw/ ====="
for tid in 3 100 101 102; do
    dir=$(printf "data/raw/traj_%04d" "$tid")
    if [[ -d "$dir/frames" ]]; then
        n_frames=$(ls "$dir/frames" 2>/dev/null | wc -l)
        echo "  $dir: present ($n_frames frame files)"
    else
        echo "  $dir: MISSING — eval will fail at dataloader build."
    fi
done
echo ""

python scripts/evaluate_tdirect.py \
    --config configs/step_j_crossfamily.yaml \
    --checkpoints_dir outputs/step_j \
    --split test \
    --methods v i vi \
    --skip_missing

RESULTS=outputs/step_j_crossfamily/results.csv
echo ""
echo "===== Step J cross-family eval results (results.csv) ====="
if [[ -f "$RESULTS" ]]; then
    cat "$RESULTS"
    echo ""
    echo "===== Headline rows (ate_aligned only) ====="
    grep -E ',ate_aligned,' "$RESULTS" || true
    echo ""
    echo "===== Per-method averages across cross-family test trajs (ate_aligned) ====="
    for m in v i vi; do
        avg=$(awk -F, -v m="$m" '
            $1==m && $3=="ate_aligned" { s+=$4; n+=1 }
            END { if (n>0) printf "%.3f", s/n; else print "n/a" }' "$RESULTS")
        echo "  $m: avg ate_aligned = $avg m"
    done
    echo ""
    echo "===== Comparison vs Step H V_3 baseline ====="
    echo "  Step H V_3   = 3.86 m  (V trained on sinusoids, evaluated on traj 3)"
    echo "  Step J V_avg = 0.089 m (V trained on simple shapes, in-family eval)"
    v3_now=$(awk -F, '$1=="v" && $2=="3" && $3=="ate_aligned" {print $4}' "$RESULTS")
    echo "  Step J V_3   = ${v3_now:-n/a} m  (V trained on simple shapes, sinusoid eval)"
else
    echo "WARNING: $RESULTS not found"
fi

echo ""
echo "===== Comparison plot files ====="
ls outputs/step_j_crossfamily/eval/plots/comparison_* 2>/dev/null \
    || echo "  (no comparison plots found — check stderr)"
