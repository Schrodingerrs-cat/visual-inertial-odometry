#!/bin/bash
#SBATCH --job-name=step_k_raft
#SBATCH --partition=short
#SBATCH -A fyuan3
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/step_k_precompute_raft.out
#SBATCH --error=slurm/logs/step_k_precompute_raft.err

# Step K precompute: run RAFT-large (frozen) over every frame pair in
# trajs 312-320 (8 train + val_traj_id=320). Writes
# data/raw_g/traj_{N}/raft_predictions.npz with R_b/t_b/sigma_b/valid_mask
# + metadata including a frames SHA-1 (R3 stale-cache detection).
#
# Test-only trajs 321-324 do NOT need RAFT cache — the teacher is a
# training-time regulariser only. The val traj (320) is included because
# train_v_distill computes val loss with the same loss function, which
# consults the cache.
#
# Estimated runtime: 9 trajs × ~3000 pairs × ~150 ms/pair ≈ 70 min on
# rtx_pro_6000_b. 6h cap is ~5× safety.

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source slurm/_common.sh

PRECOMPUTE_IDS=(312 313 314 315 316 317 318 319 320)

# L9: defensive checks — training data must exist before precompute.
for ID in "${PRECOMPUTE_IDS[@]}"; do
    if [[ ! -d "data/raw_g/traj_0$ID" ]]; then
        echo "ERROR: data/raw_g/traj_0$ID missing — Step K render did not run." >&2
        exit 1
    fi
    if [[ ! -d "data/raw_g/traj_0$ID/frames" ]]; then
        echo "ERROR: data/raw_g/traj_0$ID/frames missing." >&2
        exit 1
    fi
done

python scripts/precompute_raft.py \
    --config configs/step_k.yaml \
    --traj_ids "${PRECOMPUTE_IDS[@]}"

echo ""
echo "===== Step K RAFT precompute summary ====="
SUMMARY=outputs/step_k/raft_precompute_summary.json
if [[ -f "$SUMMARY" ]]; then
    cat "$SUMMARY"
else
    echo "WARNING: $SUMMARY not found"
fi

echo ""
echo "===== Per-traj cache verification ====="
for ID in "${PRECOMPUTE_IDS[@]}"; do
    NPZ="data/raw_g/traj_0$ID/raft_predictions.npz"
    if [[ -f "$NPZ" ]]; then
        size=$(du -h "$NPZ" | cut -f1)
        echo "OK: $NPZ ($size)"
    else
        echo "MISSING: $NPZ"
    fi
done

echo ""
echo "===== F: pre-flight RAFT-domain-mismatch warning ====="
# Surface mean_sigma_px_valid per traj so the user can decide whether
# the carpet is too uniform for RAFT before launching the train job.
# Heuristic: mean_sigma > 30 px on all trajs ⇒ RAFT can't see the carpet
# at all (texture too uniform); training with distillation will not help
# and the student loss will collapse to pose-only.
if [[ -f "$SUMMARY" ]]; then
    python3 - <<'PY'
import json
import sys
from pathlib import Path
p = Path("outputs/step_k/raft_precompute_summary.json")
blob = json.loads(p.read_text())
sigmas = []
for tid, info in blob.items():
    if info.get("status") != "ok":
        continue
    s = info.get("mean_sigma_px_valid")
    if s is None:
        continue
    sigmas.append((tid, float(s)))
    flag = " (HIGH — carpet may be too uniform)" if s > 20.0 else ""
    print(f"  traj {tid}: mean_sigma_px_valid = {s:.2f} px{flag}")
if sigmas:
    avg = sum(s for _, s in sigmas) / len(sigmas)
    print(f"  AVERAGE across trajs: {avg:.2f} px")
    if all(s > 30.0 for _, s in sigmas):
        print("  RED FLAG: ALL trajs have mean_sigma > 30 px. RAFT cannot")
        print("            see this carpet. Distillation will not help.")
        print("            Consider: try fallback weights (Raft_Large_Weights.")
        print("            C_T_V2) or skip distillation entirely.")
        sys.exit(2)
PY
fi
