#!/bin/bash
#SBATCH --job-name=step_k_ext_eval
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/step_k_extended_eval.out
#SBATCH --error=slurm/logs/step_k_extended_eval.err

# Step K extended eval — V/I/VI on the 4 held-out within-family
# probe trajs (325-328). Reuses Step K's trained checkpoints in
# outputs/step_k/{v,i,vi}/best.pt; --skip_missing tolerates partial
# checkpoint sets.
#
# Output:
#   outputs/step_k_extended/results.csv
#   outputs/step_k_extended/eval/{trajectories,plots}/

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

# Pre-flight: confirm extended trajs have been rendered.
echo "===== Pre-flight: extended trajs ====="
for tid in 325 326 327 328; do
    dir=$(printf "data/raw_g/traj_%04d" "$tid")
    if [[ -d "$dir/frames" ]]; then
        n_frames=$(ls "$dir/frames" 2>/dev/null | wc -l)
        echo "  $dir: $n_frames frames"
    else
        echo "  $dir: MISSING — eval will fail at dataloader build."
    fi
done
echo ""

python scripts/evaluate_tdirect.py \
    --config configs/step_k_extended.yaml \
    --checkpoints_dir outputs/step_k \
    --split test \
    --methods v i vi \
    --skip_missing

RESULTS=outputs/step_k_extended/results.csv
echo ""
echo "===== Extended results.csv ====="
if [[ -f "$RESULTS" ]]; then
    cat "$RESULTS"
    echo ""
    echo "===== Per-method per-traj ate_aligned ====="
    grep -E ',ate_aligned,' "$RESULTS" || true

    echo ""
    echo "===== V on each held-out within-family traj ====="
    for tid in 325 326 327 328; do
        v_ate=$(awk -F, -v t="$tid" '$1=="v" && $2==t && $3=="ate_aligned" {print $4}' "$RESULTS")
        v_seg=$(awk -F, -v t="$tid" '$1=="v" && $2==t && $3=="seg2s_median" {print $4}' "$RESULTS")
        case "$tid" in
            325) label="simple_figure8 @ 0.30 m/s";;
            326) label="simple_circle  @ 0.25 m/s";;
            327) label="simple_diamond @ 0.30 m/s";;
            328) label="simple_zigzag  @ 0.30 m/s";;
            *)   label="unknown";;
        esac
        echo "  traj $tid ($label): V ate_aligned = ${v_ate:-n/a} m  seg2s_median = ${v_seg:-n/a} m"
    done

    echo ""
    echo "===== Per-method averages over extended set (325-328) ====="
    for m in v i vi; do
        avg=$(awk -F, -v m="$m" '
            $1==m && $3=="ate_aligned" { s+=$4; n+=1 }
            END { if (n>0) printf "%.3f", s/n; else print "n/a" }' "$RESULTS")
        echo "  $m: avg ate_aligned = $avg m"
    done

    # Q1 decision-tree summary
    echo ""
    echo "===== Q1 decision-tree summary ====="
    python3 - <<'PY'
import csv
rows = list(csv.DictReader(open('outputs/step_k_extended/results.csv')))
def get(method, tid, metric):
    for r in rows:
        if r['method']==method and r['traj_id']==str(tid) and r['metric_name']==metric:
            return float(r['value'])
    return None
ates  = [get('v', t, 'ate_aligned')   for t in (325,326,327,328)]
segs  = [get('v', t, 'seg2s_median')  for t in (325,326,327,328)]
print(f"  V ate_aligned per-traj: {[f'{x:.3f}' if x else 'n/a' for x in ates]}")
print(f"  V seg2s_median per-traj: {[f'{x:.3f}' if x else 'n/a' for x in segs]}")
ok_ate = [a is not None and a < 0.5 for a in ates]
ok_seg = [s is not None and s < 0.1 for s in segs]
if all(ok_ate) and all(ok_seg):
    verdict = "BROAD: within-family generalisation extends to ALL 4 trained shapes"
elif any(ok_ate) and any(s is None or s > 1.0 for s in segs):
    verdict = "SHAPE-SPECIFIC: some shapes generalise, others fail"
elif all(a is not None and a > 1.0 for a in ates):
    verdict = "NARROW: 0.184m success was specific to ellipse / triangle / sinusoid"
else:
    verdict = "MIXED: see per-traj numbers above"
print(f"  Verdict: {verdict}")
PY
else
    echo "WARNING: $RESULTS not found"
fi

echo ""
echo "===== Comparison plot files ====="
ls outputs/step_k_extended/eval/plots/comparison_* 2>/dev/null \
    || echo "  (no comparison plots found)"
