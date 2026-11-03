#!/bin/bash
#SBATCH --job-name=step_k_p2_eval
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/step_k_phase2_eval.out
#SBATCH --error=slurm/logs/step_k_phase2_eval.err

# Phase 2 eval — V/I/VI on Step K test set (320-324) and cross-family
# (3, 100, 101, 102).
#
# Checkpoint routing:
#   V  ← outputs/step_k/v/best.pt           (UNCHANGED across phases)
#   I  ← outputs/step_k_phase2/i/best.pt    (Phase 2 retrain)
#   VI ← outputs/step_k/vi/best.pt          (UNCHANGED across phases)
#
# The eval pipeline routes batch["imu"] (length 10) to V/VI and
# batch["imu_context"] (length 100, set via configs/step_k_phase2.yaml's
# data.imu_context_length) to I. VI compatibility is asserted by the
# sanity check at the end.

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

CKPT_DIR=outputs/step_k_phase2_ckpt_combined

# ---------------------------------------------------------------------
# 1. Build a unified checkpoints dir layout that evaluate_tdirect.py
#    expects: {ckpt_dir}/{v,i,vi}/best.pt. We symlink V and VI from the
#    unchanged Phase 0 checkpoints, and I from the Phase 2 retrain.
# ---------------------------------------------------------------------
echo "===== Build unified checkpoint directory ====="
mkdir -p "$CKPT_DIR"
for sub in v i vi; do
    rm -rf "$CKPT_DIR/$sub"
done
ln -sfn "$(realpath outputs/step_k/v)"        "$CKPT_DIR/v"
ln -sfn "$(realpath outputs/step_k_phase2/i)" "$CKPT_DIR/i"
ln -sfn "$(realpath outputs/step_k/vi)"       "$CKPT_DIR/vi"

for m in v i vi; do
    if [[ ! -f "$CKPT_DIR/$m/best.pt" ]]; then
        echo "ERROR: $CKPT_DIR/$m/best.pt missing (symlink chain broken)." >&2
        exit 1
    fi
    target=$(readlink -f "$CKPT_DIR/$m/best.pt")
    echo "  $m: $CKPT_DIR/$m/best.pt → $target"
done

# ---------------------------------------------------------------------
# 2. Wipe stale results.csv so re-eval doesn't append duplicate rows.
# ---------------------------------------------------------------------
rm -f outputs/step_k_phase2/results.csv outputs/step_k_phase2_crossfamily/results.csv

# ---------------------------------------------------------------------
# 3. Primary eval — Step K within-family test set (320-324).
# ---------------------------------------------------------------------
echo ""
echo "===== STAGE 1/2: Step K primary eval (test_ids 320-324) ====="
python scripts/evaluate_tdirect.py \
    --config configs/step_k_phase2.yaml \
    --checkpoints_dir "$CKPT_DIR" \
    --split test \
    --methods v i vi \
    --skip_missing

# ---------------------------------------------------------------------
# 4. Cross-family eval (3, 100, 101, 102).
# ---------------------------------------------------------------------
echo ""
echo "===== STAGE 2/2: cross-family eval (3, 100, 101, 102) ====="
python scripts/evaluate_tdirect.py \
    --config configs/step_k_phase2_crossfamily.yaml \
    --checkpoints_dir "$CKPT_DIR" \
    --split test \
    --methods v i vi \
    --skip_missing

# ---------------------------------------------------------------------
# 5. VI sanity check — Phase 2 VI numbers MUST match Phase 0/1 VI within
#    1e-4 m. If they don't, option-(a) plumbing is wrong (VI's IMU
#    branch was fed the longer context tensor instead of the legacy
#    length-10 tensor).
# ---------------------------------------------------------------------
echo ""
echo "===== VI sanity check (Phase 2 VI must equal Phase 1/0 VI within 1e-4) ====="
python3 - <<'PY'
import csv, os, sys
def load(p):
    if not os.path.exists(p): return {}
    d = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            d[(r['method'], int(r['traj_id']), r['metric_name'])] = float(r['value'])
    return d

p2_primary = load('outputs/step_k_phase2/results.csv')
p2_cross   = load('outputs/step_k_phase2_crossfamily/results.csv')

# Phase 0/1 reference: outputs/step_k/results.csv contains the most-recent
# eval of the unchanged VI checkpoint. (Phase 1 eval overwrote this with
# its re-eval of v/i/vi.) Use it as the VI-invariant reference.
ref_primary = load('outputs/step_k/results.csv')
ref_cross   = load('outputs/step_k_crossfamily/results.csv')

ok = True
print(f"  {'traj':>5} {'metric':>14} {'P2 vi':>10} {'ref vi':>10} {'|Δ|':>8}")
for src_p2, src_ref, label, tids in [
    (p2_primary, ref_primary, 'primary',     [320, 321, 322, 323, 324]),
    (p2_cross,   ref_cross,   'cross-family',[3, 100, 101, 102]),
]:
    print(f"  --- {label} ---")
    for t in tids:
        for m in ('ate_aligned', 'seg2s_median'):
            a = src_p2.get(('vi', t, m))
            b = src_ref.get(('vi', t, m))
            if a is None or b is None:
                print(f"  {t:>5} {m:>14} MISSING (a={a}, b={b})")
                ok = False
                continue
            diff = abs(a - b)
            flag = '' if diff < 1e-4 else '  *** MISMATCH ***'
            if diff >= 1e-4: ok = False
            print(f"  {t:>5} {m:>14} {a:>10.5f} {b:>10.5f} {diff:>8.6f}{flag}")
if ok:
    print("\n  PASS: Phase 2 VI numbers match reference within 1e-4 m.")
    print("  Option-(a) plumbing is correct: VI consumed batch['imu'] (length 10).")
else:
    print("\n  FAIL: Phase 2 VI numbers diverged from reference. Option-(a)")
    print("  plumbing is wrong — VI's IMU branch may have been fed batch")
    print("  ['imu_context'] instead of batch['imu']. Investigate before")
    print("  trusting the I-branch numbers in the comparison table below.")
    sys.exit(2)
PY

# ---------------------------------------------------------------------
# 6. Phase 0 / Phase 1 / Phase 2 comparison table.
# ---------------------------------------------------------------------
echo ""
echo "===== Phase 0 / Phase 1 / Phase 2 comparison ====="
python3 - <<'PY'
import csv, os
def load(p):
    if not os.path.exists(p): return {}
    d = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            d[(r['method'], int(r['traj_id']), r['metric_name'])] = float(r['value'])
    return d

# Source paths:
#   Phase 0 baseline: outputs/step_k_phase0_baseline_eval/  (created by
#                     slurm/step_k_phase1_i_eval.sh on its first run)
#   Phase 1:           outputs/step_k/  (overwritten by Phase 1 eval)
#   Phase 2:           outputs/step_k_phase2/  (this run)
# We tag whichever exists.
phase0_primary = load('outputs/step_k_phase0_baseline_eval/results.csv')
phase0_cross   = load('outputs/step_k_phase0_baseline_eval/results_crossfamily.csv')
phase1_primary = load('outputs/step_k/results.csv')
phase1_cross   = load('outputs/step_k_crossfamily/results.csv')
phase2_primary = load('outputs/step_k_phase2/results.csv')
phase2_cross   = load('outputs/step_k_phase2_crossfamily/results.csv')

def fmt(x): return f'{x:7.3f}' if x is not None else '   n/a '
def block(label, p0, p1, p2, tids):
    print(f"\n  --- I network — {label} ---")
    print(f"  {'traj':>5} {'P0 ate':>9} {'P1 ate':>9} {'P2 ate':>9}    "
          f"{'P0 seg2s':>10} {'P1 seg2s':>10} {'P2 seg2s':>10}")
    for t in tids:
        a0 = p0.get(('i', t, 'ate_aligned'))
        a1 = p1.get(('i', t, 'ate_aligned'))
        a2 = p2.get(('i', t, 'ate_aligned'))
        s0 = p0.get(('i', t, 'seg2s_median'))
        s1 = p1.get(('i', t, 'seg2s_median'))
        s2 = p2.get(('i', t, 'seg2s_median'))
        print(f"  {t:>5} {fmt(a0)} {fmt(a1)} {fmt(a2)}    "
              f"{fmt(s0)} {fmt(s1)} {fmt(s2)}")

block('Step K primary',  phase0_primary, phase1_primary, phase2_primary,
      [320, 321, 322, 323, 324])
block('cross-family',    phase0_cross,   phase1_cross,   phase2_cross,
      [3, 100, 101, 102])

# Aggregates + decision
def avg(d, m, tids, metric='ate_aligned'):
    vals = [d.get((m, t, metric)) for t in tids]
    vals = [v for v in vals if v is not None]
    return sum(vals)/len(vals) if vals else None

print('\n  === I aggregate ===')
for tag, d, tids in [
    ('primary  P0', phase0_primary, [320, 321, 322, 323, 324]),
    ('primary  P1', phase1_primary, [320, 321, 322, 323, 324]),
    ('primary  P2', phase2_primary, [320, 321, 322, 323, 324]),
    ('crossfam P0', phase0_cross,   [3, 100, 101, 102]),
    ('crossfam P1', phase1_cross,   [3, 100, 101, 102]),
    ('crossfam P2', phase2_cross,   [3, 100, 101, 102]),
]:
    a = avg(d, 'i', tids)
    print(f'  {tag} I_avg = {fmt(a)}')

new_avg = avg(phase2_primary, 'i', [320, 321, 322, 323, 324])
print('\n  === Phase 2 decision ===')
if new_avg is None:
    print('  (no Phase 2 I results — eval failed)')
elif new_avg < 0.6:
    print(f'  Phase 2 worked: I_avg = {new_avg:.3f} < 0.6 m.')
    print('  Window length was the bottleneck.')
elif new_avg < 1.0:
    print(f'  MARGINAL: I_avg = {new_avg:.3f} in [0.6, 1.0).')
    print('  Architectural capacity bump may help; document and stop.')
else:
    print(f'  Phase 2 also insufficient: I_avg = {new_avg:.3f} >= 1.0.')
    print('  Bottleneck is not loss, not window length. Document')
    print('  architecture or representation as the residual.')
PY

echo ""
echo "===== Plot files ====="
ls outputs/step_k_phase2/eval/plots/comparison_*_xy.png 2>/dev/null | head
ls outputs/step_k_phase2_crossfamily/eval/plots/comparison_*_xy.png 2>/dev/null | head
