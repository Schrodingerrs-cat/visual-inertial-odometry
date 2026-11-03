#!/bin/bash
#SBATCH --job-name=step_s_eval
#SBATCH --partition=long
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/step_s_eval.out
#SBATCH --error=slurm/logs/step_s_eval.err

# Step S eval — two stages, both using scripts/evaluate_tdirect.py
# (Step S V/VI are NetworkV_tdirect / NetworkVI_tdirect, same as J/K).
#
# Stage 1 — "Step S primary" (the headline). Pure speed sweep on a
#   single held-out shape (simple_ellipse) at 0.5, 0.8, 1.0, 1.2 m/s.
#   Test trajs: [410, 411, 412, 413].
#   Output:     outputs/step_s/results.csv
#
# Stage 2 — "Step S on Step K test set" (slow-speed regression check).
#   Tests whether fast-trained Step S preserves slow-speed performance.
#   Test trajs: [320, 321, 322, 323, 324]  (Step K's existing test set).
#   Output:     outputs/step_s_kxtest/results.csv
#
# Final summary prints two comparison tables:
#   1. Step K (slow train, slow test) vs Step S (fast train, fast test)
#   2. Step K (slow train, slow test) vs Step S on Step K trajs
#                                        (fast train, slow test)

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

# --- Pre-flight: confirm Step S checkpoints exist ----------------------
echo "===== Pre-flight: checkpoint presence ====="
for m in v i vi; do
    ckpt="outputs/step_s/${m}/best.pt"
    if [[ -f "$ckpt" ]]; then
        echo "  $ckpt: present"
    else
        echo "  $ckpt: MISSING"
    fi
done
echo ""

# --- Pre-flight: confirm test trajs are rendered -----------------------
echo "===== Pre-flight: test traj presence ====="
for tid in 410 411 412 413 320 321 322 323 324; do
    dir=$(printf "data/raw_g/traj_%04d" "$tid")
    if [[ -d "$dir/frames" ]]; then
        n_frames=$(ls "$dir/frames" 2>/dev/null | wc -l)
        echo "  $dir: $n_frames frames"
    else
        echo "  $dir: MISSING"
    fi
done
echo ""

# --- Wipe stale results.csv so re-eval doesn't append duplicate rows ---
rm -f outputs/step_s/results.csv outputs/step_s_kxtest/results.csv

# ======================================================================
# STAGE 1 — Step S primary (held-out shapes at 1 m/s + interpolation).
# ======================================================================
echo "===== STAGE 1/2: Step S primary eval (test_ids 410, 411, 412, 413 — ellipse speed sweep) ====="
python scripts/evaluate_tdirect.py \
    --config configs/step_s.yaml \
    --checkpoints_dir outputs/step_s \
    --split test \
    --methods v i vi \
    --skip_missing

PRIMARY=outputs/step_s/results.csv
echo ""
echo "===== Stage 1 results.csv ====="
if [[ -f "$PRIMARY" ]]; then
    cat "$PRIMARY"
else
    echo "WARNING: $PRIMARY not found"
fi

# ======================================================================
# STAGE 2 — Step S on Step K test set (slow-speed regression check).
# ======================================================================
echo ""
echo "===== STAGE 2/2: Step S on Step K test set (test_ids 320, 321, 322, 323, 324) ====="
python scripts/evaluate_tdirect.py \
    --config configs/step_s_kxtest.yaml \
    --checkpoints_dir outputs/step_s \
    --split test \
    --methods v i vi \
    --skip_missing

KXTEST=outputs/step_s_kxtest/results.csv
echo ""
echo "===== Stage 2 results.csv ====="
if [[ -f "$KXTEST" ]]; then
    cat "$KXTEST"
else
    echo "WARNING: $KXTEST not found"
fi

# ======================================================================
# Comparison tables — Step K vs Step S.
# ======================================================================
echo ""
echo "===== Comparison: Step K vs Step S ====="
python3 - <<'PY'
import csv
from pathlib import Path

def load(p):
    d = {}
    if not Path(p).is_file():
        return d
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                tid = int(r['traj_id'])
            except (KeyError, ValueError):
                continue
            d[(r['method'], tid, r['metric_name'])] = float(r['value'])
    return d

step_k        = load('outputs/step_k/results.csv')               # slow train, slow test
step_s_prim   = load('outputs/step_s/results.csv')               # fast train, fast test (Stage 1)
step_s_kxtest = load('outputs/step_s_kxtest/results.csv')        # fast train, slow test (Stage 2)

def fmt(x):
    return f'{x:7.3f}' if isinstance(x, float) else '   n/a '

def avg(d, m, tids, metric='ate_aligned'):
    vals = [d.get((m, t, metric)) for t in tids]
    vals = [v for v in vals if v is not None]
    return sum(vals)/len(vals) if vals else None

# ---- Table 1: Step K (slow/slow) vs Step S Stage 1 (speed sweep) -----
print()
print('--- Table 1: Step K vs Step S Stage 1 (ellipse speed sweep) ---')
print('Step K averages over its own test_ids 320-324; Step S Stage 1 over 410-413.')
print('Step S Stage 1 also broken out per-speed so the sweep is visible.')
K_TIDS  = [320, 321, 322, 323, 324]
S1_TIDS = [410, 411, 412, 413]
S1_SPEEDS = {410: 0.5, 411: 0.8, 412: 1.0, 413: 1.2}
print(f"{'method':>6} {'K_avg_ate':>10} {'S1_avg_ate':>11}")
for m in ('v', 'i', 'vi'):
    k = avg(step_k,      m, K_TIDS)
    s = avg(step_s_prim, m, S1_TIDS)
    print(f"{m:>6} {fmt(k)}    {fmt(s)}")
print()
print('--- Step S Stage 1 per-speed (ellipse) ---')
print(f"{'method':>6} " + " ".join(f"{S1_SPEEDS[t]:>7}m/s" for t in S1_TIDS))
for m in ('v', 'i', 'vi'):
    cells = [step_s_prim.get((m, t, 'ate_aligned')) for t in S1_TIDS]
    print(f"{m:>6} " + " ".join(fmt(c) + "    " for c in cells))

# ---- Table 2: Step K vs Step S Stage 2 (regression on Step K trajs) --
print()
print('--- Table 2: Step K (slow train, slow test) vs Step S Stage 2 (fast train, slow test) ---')
print('Same trajs (320-324) on both columns; only the trained network differs.')
print(f"{'method':>6} {'traj':>5} {'K_ate':>7} {'S2_ate':>7} {'Δate':>7}")
for m in ('v', 'i', 'vi'):
    for t in K_TIDS:
        k = step_k.get((m, t, 'ate_aligned'))
        s = step_s_kxtest.get((m, t, 'ate_aligned'))
        d = (s - k) if (k is not None and s is not None) else None
        print(f"{m:>6} {t:>5} {fmt(k)} {fmt(s)} {fmt(d)}")
print(f"{'method':>6} {'avg':>5} {'K_ate':>7} {'S2_ate':>7} {'Δate':>7}")
for m in ('v', 'i', 'vi'):
    k = avg(step_k,        m, K_TIDS)
    s = avg(step_s_kxtest, m, K_TIDS)
    d = (s - k) if (k is not None and s is not None) else None
    print(f"{m:>6}   avg {fmt(k)} {fmt(s)} {fmt(d)}")

# ---- Decision matrix --------------------------------------------------
def vavg(d, tids):
    return avg(d, 'v', tids)

s1_v = vavg(step_s_prim,   S1_TIDS)
s2_v = vavg(step_s_kxtest, K_TIDS)
extrap_v = step_s_prim.get(('v', 413, 'ate_aligned'))   # 1.2 m/s only
print()
print('--- Decision matrix (V averages) ---')
print(f"  Stage 1 V_avg (ellipse 0.5-1.2 m/s sweep):           {fmt(s1_v)} m")
print(f"  Stage 1 V    (ellipse 1.2 m/s — extrapolation only): {fmt(extrap_v)} m")
print(f"  Stage 2 V_avg (Step K test set, slow):               {fmt(s2_v)} m")
if s1_v is None or s2_v is None:
    print("  (insufficient data — eval did not produce one of the result sets)")
else:
    s1_ok = s1_v < 0.5
    s2_ok = s2_v < 0.3
    extrap_ok = (extrap_v is not None and extrap_v < 0.6)
    if s1_ok and s2_ok and extrap_ok:
        print("  → speed-diverse training generalises BOTH ways AND extrapolates "
              "cleanly to 1.2 m/s. Strong finding.")
    elif s1_ok and s2_ok and not extrap_ok:
        print("  → in-range generalisation works but 1.2 m/s extrapolation breaks. "
              "Document as a motion-magnitude ceiling at the training upper bound.")
    elif s1_ok and not s2_ok:
        print("  → tradeoff: speed-diverse training destroys slow capability. "
              "Document as a generalization limit.")
    elif (not s1_ok) and s2_ok:
        print("  → architecture ceiling within trained speeds. Investigate.")
    else:
        print("  → both stages failed. Investigate Step S training quality.")
PY

echo ""
echo "===== Plot files ====="
ls outputs/step_s/eval/plots/comparison_*.png 2>/dev/null | head \
    || echo "  (no Stage 1 comparison plots found)"
ls outputs/step_s_kxtest/eval/plots/comparison_*.png 2>/dev/null | head \
    || echo "  (no Stage 2 comparison plots found)"
