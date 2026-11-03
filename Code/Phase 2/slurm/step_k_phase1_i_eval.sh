#!/bin/bash
#SBATCH --job-name=step_k_p1_i_eval
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/step_k_phase1_i_eval.out
#SBATCH --error=slurm/logs/step_k_phase1_i_eval.err

# Phase 1 — re-evaluate V/I/VI on the Step K test set (320-324) AND on the
# cross-family test set (3, 100, 101, 102) after retraining I with
# IMUPoseLoss_hover. Backs up the prior eval artifacts first so the
# baseline-vs-Phase-1 comparison is preserved on disk.
#
# Pre-req: outputs/step_k/i/best.pt has been retrained with the new loss.
# V/VI checkpoints are NOT touched in Phase 1 (per spec).

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

# --- 1. Backup old eval artifacts (idempotent: only on first run) -------
echo "===== Backup old eval artifacts ====="
BACKUP=outputs/step_k_phase0_baseline_eval
if [[ ! -d "$BACKUP" ]]; then
    mkdir -p "$BACKUP"
    [[ -f outputs/step_k/results.csv ]] && cp outputs/step_k/results.csv "$BACKUP/"
    [[ -d outputs/step_k/eval ]] && cp -r outputs/step_k/eval "$BACKUP/"
    [[ -f outputs/step_k_crossfamily/results.csv ]] && \
        cp outputs/step_k_crossfamily/results.csv "$BACKUP/results_crossfamily.csv"
    [[ -d outputs/step_k_crossfamily/eval ]] && \
        cp -r outputs/step_k_crossfamily/eval "$BACKUP/eval_crossfamily"
    echo "  backed up to $BACKUP"
else
    echo "  $BACKUP exists — skipping backup (preserving original baseline)"
fi

# --- 2. Wipe stale results.csv so re-eval doesn't append duplicate rows -
# Trajectories/plots get overwritten by evaluate_tdirect.py.
rm -f outputs/step_k/results.csv outputs/step_k_crossfamily/results.csv

# --- 3. Re-eval primary (Step K test set) -------------------------------
echo ""
echo "===== STAGE 1/2: Step K primary eval (test_ids 320-324) ====="
python scripts/evaluate_tdirect.py \
    --config configs/step_k.yaml \
    --checkpoints_dir outputs/step_k \
    --split test \
    --methods v i vi \
    --skip_missing

# --- 4. Re-eval cross-family --------------------------------------------
echo ""
echo "===== STAGE 2/2: cross-family eval (3, 100, 101, 102) ====="
python scripts/evaluate_tdirect.py \
    --config configs/step_k_crossfamily.yaml \
    --checkpoints_dir outputs/step_k \
    --split test \
    --methods v i vi \
    --skip_missing

# --- 5. Phase-1 vs baseline comparison summary --------------------------
echo ""
echo "===== Phase 1 vs baseline (I network) ====="
python3 - <<'PY'
import csv
def load(p):
    d = {}
    try:
        with open(p) as f:
            for r in csv.DictReader(f):
                d[(r['method'], int(r['traj_id']), r['metric_name'])] = float(r['value'])
    except FileNotFoundError:
        pass
    return d

primary_old = load('outputs/step_k_phase0_baseline_eval/results.csv')
primary_new = load('outputs/step_k/results.csv')
cross_old   = load('outputs/step_k_phase0_baseline_eval/results_crossfamily.csv')
cross_new   = load('outputs/step_k_crossfamily/results.csv')

def fmt(x): return f'{x:7.3f}' if x is not None else '   n/a '
def block(label, old, new, tids):
    print(f'\n--- {label} ---')
    print(f"{'method':>6} {'traj':>5} {'ate_old':>8} {'ate_new':>8} {'Δate':>8}    "
          f"{'seg2s_old':>10} {'seg2s_new':>10} {'Δseg2s':>9}")
    for m in ('i','v','vi'):
        for t in tids:
            ao = old.get((m,t,'ate_aligned')); an = new.get((m,t,'ate_aligned'))
            so = old.get((m,t,'seg2s_median')); sn = new.get((m,t,'seg2s_median'))
            da = (an-ao) if (ao is not None and an is not None) else None
            ds = (sn-so) if (so is not None and sn is not None) else None
            print(f"{m:>6} {t:>5} {fmt(ao)} {fmt(an)} {fmt(da)}    "
                  f"{fmt(so)} {fmt(sn)} {fmt(ds)}")
block('Step K test set', primary_old, primary_new, [320,321,322,323,324])
block('cross-family',    cross_old,   cross_new,   [3,100,101,102])

# I averages
def avg(d, m, tids, metric='ate_aligned'):
    vals = [d.get((m,t,metric)) for t in tids]
    vals = [v for v in vals if v is not None]
    return sum(vals)/len(vals) if vals else None

print('\n=== I aggregate ===')
print(f'  Step K test set  I_avg: old = {fmt(avg(primary_old, "i", [320,321,322,323,324]))}'
      f' → new = {fmt(avg(primary_new, "i", [320,321,322,323,324]))}')
print(f'  cross-family     I_avg: old = {fmt(avg(cross_old,   "i", [3,100,101,102]))}'
      f' → new = {fmt(avg(cross_new,   "i", [3,100,101,102]))}')

# Decision-tree check
new_avg = avg(primary_new, 'i', [320,321,322,323,324])
print('\n=== Phase-1 decision ===')
if new_avg is None:
    print('  (no new I results — eval failed)')
elif new_avg < 0.6:
    print(f'  STOP: new I_avg = {new_avg:.3f} < 0.6 — loss fix sufficient.')
elif new_avg < 1.0:
    print(f'  MARGINAL: new I_avg = {new_avg:.3f} in [0.6, 1.0] — defer to user.')
else:
    print(f'  PROCEED to Phase 2: new I_avg = {new_avg:.3f} >= 1.0.')
PY

echo ""
echo "===== Plot comparison files ====="
ls outputs/step_k/eval/plots/comparison_*_xy.png 2>/dev/null | head
ls outputs/step_k_crossfamily/eval/plots/comparison_*_xy.png 2>/dev/null | head
