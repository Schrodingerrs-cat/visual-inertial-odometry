#!/bin/bash
#SBATCH --job-name=step_k_eval
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/step_k_eval.out
#SBATCH --error=slurm/logs/step_k_eval.err

# Step K eval: V/I/VI on test_ids = [320, 321, 322, 323, 324]. Reuses
# scripts/evaluate_tdirect.py unchanged. Then runs a SECOND eval against
# the Step F/G cross-family test set ([3, 100, 101, 102]) — free
# information that lets us compare K's V vs J's V on the original
# aggressive-sinusoid trajs that prompted Step K in the first place.
#
# Output layout:
#   outputs/step_k/results.csv                — primary (held-out shapes)
#   outputs/step_k_crossfamily/results.csv    — secondary (Step F/G trajs)

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

# ---------------------------------------------------------------------
# Primary eval — Step K held-out shapes at varied speeds.
# ---------------------------------------------------------------------
echo "===== STAGE 1/2: primary eval (held-out shapes) ====="
python scripts/evaluate_tdirect.py \
    --config configs/step_k.yaml \
    --checkpoints_dir outputs/step_k \
    --split test \
    --methods v i vi \
    --skip_missing

RESULTS=outputs/step_k/results.csv
echo ""
echo "===== Primary results.csv ====="
if [[ -f "$RESULTS" ]]; then
    cat "$RESULTS"
    echo ""
    echo "===== Per-method per-traj ate_aligned ====="
    grep -E ',ate_aligned,' "$RESULTS" || true

    echo ""
    echo "===== Per-method averages over ALL test trajs (320-324) ====="
    for m in v i vi; do
        avg=$(awk -F, -v m="$m" '
            $1==m && $3=="ate_aligned" { s+=$4; n+=1 }
            END { if (n>0) printf "%.3f", s/n; else print "n/a" }' "$RESULTS")
        echo "  $m: avg ate_aligned = $avg m"
    done

    # The genuinely held-out subset = {321, 323, 324}.
    # 320 (val) and 322 (minitest) are model-selection diagnostics.
    echo ""
    echo "===== Per-method averages over GENUINELY held-out (321, 323, 324) ====="
    for m in v i vi; do
        avg=$(awk -F, -v m="$m" '
            $1==m && $3=="ate_aligned" && ($2=="321" || $2=="323" || $2=="324") { s+=$4; n+=1 }
            END { if (n>0) printf "%.3f", s/n; else print "n/a" }' "$RESULTS")
        echo "  $m: avg ate_aligned = $avg m  (genuinely held-out only)"
    done

    # Per-axis breakdown for finer-grained interpretation
    # (matches the decision tree in the prompt).
    echo ""
    echo "===== V on each held-out trajectory ====="
    for tid in 320 321 322 323 324; do
        v_ate=$(awk -F, -v t="$tid" '$1=="v" && $2==t && $3=="ate_aligned" {print $4}' "$RESULTS")
        # Map traj_id → human-readable label.
        case "$tid" in
            320) label="ellipse  mid-speed (val)";;
            321) label="ellipse  fast (OOD speed)";;
            322) label="triangle mid-speed (minitest)";;
            323) label="triangle fast (OOD speed)";;
            324) label="sinusoid_30s @ 0.5 (NEW SHAPE)";;
            *)   label="unknown";;
        esac
        echo "  traj $tid ($label): V = ${v_ate:-n/a} m"
    done

    # Headline V comparison vs Step J.
    echo ""
    echo "===== Headline: V distillation effect (Step K vs Step J) ====="
    K_V_AVG=$(awk -F, '$1=="v" && $3=="ate_aligned" { s+=$4; n+=1 }
                       END { if (n>0) printf "%.3f", s/n }' "$RESULTS")
    J_RES=outputs/step_j/results.csv
    if [[ -f "$J_RES" ]]; then
        J_V_AVG=$(awk -F, '$1=="v" && $3=="ate_aligned" { s+=$4; n+=1 }
                          END { if (n>0) printf "%.3f", s/n }' "$J_RES")
        echo "  Step K V_avg (test_ids 320-324):    $K_V_AVG m"
        echo "  Step J V_avg (test_ids 305-307):    $J_V_AVG m"
        echo "  (different test sets — comparison is apples-to-pears.)"
    else
        echo "  Step K V_avg: $K_V_AVG m    (Step J results.csv not found)"
    fi

    # I/VI delta — does fusion add value with the diverse training set?
    echo ""
    echo "===== Fusion check: VI vs V on primary test set ====="
    VI_AVG=$(awk -F, '$1=="vi" && $3=="ate_aligned" { s+=$4; n+=1 }
                      END { if (n>0) printf "%.3f", s/n }' "$RESULTS")
    if [[ -n "$K_V_AVG" && -n "$VI_AVG" ]]; then
        echo "  V_avg=$K_V_AVG m   VI_avg=$VI_AVG m"
    fi
else
    echo "WARNING: $RESULTS not found"
fi

# ---------------------------------------------------------------------
# Secondary eval — Step F/G cross-family trajs (free information).
# ---------------------------------------------------------------------
echo ""
echo "===== STAGE 2/2: cross-family eval (Step F/G trajs 3, 100, 101, 102) ====="

# Pre-flight: confirm Step K's checkpoints exist.
for m in v i vi; do
    ckpt="outputs/step_k/${m}/best.pt"
    if [[ -f "$ckpt" ]]; then
        echo "  $ckpt: present"
    else
        echo "  $ckpt: MISSING"
    fi
done
echo ""

# Pre-flight: confirm test trajs are rendered in data/raw/.
for tid in 3 100 101 102; do
    dir=$(printf "data/raw/traj_%04d" "$tid")
    if [[ -d "$dir/frames" ]]; then
        n_frames=$(ls "$dir/frames" 2>/dev/null | wc -l)
        echo "  $dir: $n_frames frames"
    else
        echo "  $dir: MISSING — cross-family eval will fail at dataloader build."
    fi
done
echo ""

python scripts/evaluate_tdirect.py \
    --config configs/step_k_crossfamily.yaml \
    --checkpoints_dir outputs/step_k \
    --split test \
    --methods v i vi \
    --skip_missing

CROSS_RESULTS=outputs/step_k_crossfamily/results.csv
echo ""
echo "===== Cross-family results.csv ====="
if [[ -f "$CROSS_RESULTS" ]]; then
    cat "$CROSS_RESULTS"
    echo ""
    echo "===== Per-method averages across cross-family test trajs ====="
    for m in v i vi; do
        avg=$(awk -F, -v m="$m" '
            $1==m && $3=="ate_aligned" { s+=$4; n+=1 }
            END { if (n>0) printf "%.3f", s/n; else print "n/a" }' "$CROSS_RESULTS")
        echo "  $m: avg ate_aligned = $avg m"
    done

    echo ""
    echo "===== KEY ABLATION: traj 324 (controlled-speed sinusoid) vs traj 3 (aggressive sinusoid) ====="
    K_324=$(awk -F, '$1=="v" && $2=="324" && $3=="ate_aligned" {print $4}' "$RESULTS")
    K_3=$(awk -F, '$1=="v" && $2=="3" && $3=="ate_aligned" {print $4}' "$CROSS_RESULTS")
    echo "  V on traj 324 (30 s, 0.5 m/s, sinusoid_30s):    ${K_324:-n/a} m  ← SHAPE only"
    echo "  V on traj 3   (100 s, sum-of-sinusoids):        ${K_3:-n/a}   m  ← shape+length+accel"
    echo "  If 324 << 3:  length/magnitude dominate cross-family failure (NOT shape)."
    echo "  If 324 ≈ 3:   sinusoidal shape itself is the issue."

    # Compare vs Step J cross-family.
    J_CROSS=outputs/step_j_crossfamily/results.csv
    if [[ -f "$J_CROSS" ]]; then
        echo ""
        echo "===== Cross-family: Step K V vs Step J V ====="
        for tid in 3 100 101 102; do
            K=$(awk -F, -v t="$tid" '$1=="v" && $2==t && $3=="ate_aligned" {print $4}' "$CROSS_RESULTS")
            J=$(awk -F, -v t="$tid" '$1=="v" && $2==t && $3=="ate_aligned" {print $4}' "$J_CROSS")
            echo "  traj $tid: Step K V = ${K:-n/a} m    Step J V = ${J:-n/a} m"
        done
    fi
else
    echo "WARNING: $CROSS_RESULTS not found"
fi

echo ""
echo "===== Comparison plot files ====="
ls outputs/step_k/eval/plots/comparison_* 2>/dev/null \
    || echo "  (no primary comparison plots found)"
ls outputs/step_k_crossfamily/eval/plots/comparison_* 2>/dev/null \
    || echo "  (no cross-family comparison plots found)"
