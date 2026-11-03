# Step K — RAFT teacher knowledge distillation

## Hypothesis

Steps F → J have not pushed V's held-out ATE below ~3.86 m. The Step J
re-test on simple trajectories (figure-8 / circle / ellipse) tests whether
data complexity was the bottleneck. Step K tests an orthogonal hypothesis:
**V's small VGG8 backbone cannot acquire sufficient motion priors from
limited training data alone**. A pretrained optical-flow network (RAFT,
trained on millions of motion patterns from FlyingChairs + FlyingThings3D
+ Sintel + KITTI) can serve as a teacher to constrain V's predictions
during training.

If V_test_avg in Step K is meaningfully lower than Step J's, the
representation-learning bottleneck hypothesis is confirmed.

## Architecture

```
TRAINING TIME ONLY:
    img_a, img_b ─► RAFT-large (frozen, 12 iterations)
                       │
                       ├─► flow_fwd, flow_bwd
                       │       │
                       │       └─► forward-backward consistency
                       │            (per-pair median pixel residual = σ_b)
                       │
                       └─► flow_fwd ─► grid correspondences (stride 16)
                                        │
                                        └─► cv2.findHomography (USAC_MAGSAC, 2px)
                                              │
                                              └─► decompose_homography(H, K, prior_normal)
                                                     │
                                                     └─► (R_b, t_b/d, n_b)
                                                           │
                                                           └─► t_b = (t_b/d) · depth_gt

INFERENCE TIME:
    img_a, img_b ─► NetworkV_tdirect ─► (R_a, t_a)        # student only

LOSS (training):
    σ_t = σ_b_px · (depth_gt / fx)               # pixel σ → metric σ
    σ_R = σ_b_px / fx                             # pixel σ → angular σ
    L = w_rot · L_rot(R_a, R_gt) + w_t · L_t(t_a, t_gt)        # standard pose
        + λ · valid_mask · 1/(σ_t² + ε_t²) · ‖t_a − t_b‖²       # distill T
        + λ · valid_mask · 1/(σ_R² + ε_R²) · geodesic(R_a, R_b)² # distill R
```

The unit-correct σ conversion via `(depth/fx, 1/fx)` is critical. RAFT's
FB residual is in **pixels**; pose residuals are in **metres** and
**radians**. The naive form `(t_a − t_b)² / σ_b²` would silently scale by
`1/(focal² · ...)`, mixing units.

## Lessons applied from prior audits

- **L1 — `stage2_lr_mult: 0.0`**: V/I freeze across both VI stages.
- **L2 — eval-time depth**: tdirect variant doesn't use depth at eval; moot.
- **L3 — NaN guards**: `RAFTTeacher` `nan_to_num`'s flow output; the
  distillation loss `nan_to_num`'s σ, weights, per-term losses, and the
  scalar total. Triple guard.
- **L4 — SLURM hygiene**: 5 new scripts verified for LF line endings,
  trailing newline, and `bash -n`.
- **L5 — TRAJ_ID_MAP / TEXTURE_ASSIGNMENT**: no new traj IDs needed
  (Step J already added 301..307; Step K reuses).
- **L6 — Reuse existing scripts**: `train_i.py`, `train_vi_tdirect.py`,
  `evaluate_tdirect.py` reused unchanged with `--config configs/step_k.yaml`.
- **L7 — I-baseline cross-check**: `step_k_eval.sh` compares
  `i_305_step_k` against `i_305_step_j`; if I is symlinked, they must
  match exactly.
- **L8 — comprehensive plots**: training-time and eval-time plots from
  Step J's pipeline carry over. ATE in title; start/end markers; V/I/VI
  comparison overlays per traj.
- **L9 — defensive SLURM checks**: every SLURM script tests for required
  inputs (data dirs, RAFT cache, V/I checkpoints) before launching python.
- **L10 — symlinks verified**: `step_k_train_vi.sh` uses `ls -L` to test
  that `outputs/step_k/i/best.pt` actually resolves through the symlink.

## Changes Made

### NEW modules

- `models/raft_teacher.py`: `RAFTTeacher(nn.Module)` wraps `raft_large`
  (frozen, eval mode). Methods: `flow`, `bidirectional_flow`,
  `fb_inconsistency_pixel`, `per_pair_sigma_px`,
  `per_pair_flow_magnitude_px`, `predict_full`. Returns
  `TeacherPrediction(R_b, t_b, sigma_b_px, valid_mask, flow_med_px)`.
  Composite `valid_mask` requires: flow magnitude > 1 px AND FB σ < 30 px
  AND ≥8 RANSAC inliers AND homography decomposition succeeded AND
  per-pair |t_b| ≤ 1 m AND geodesic(R_b) ≤ π/4. Default weights:
  `Raft_Large_Weights.C_T_SKHT_V2` (FlyingChairs + FlyingThings3D +
  Sintel/KITTI/HD1K fine-tune).

- `models/losses_distill.py`: `TDirectDistillLoss(nn.Module)` composes
  the existing `TDirectPoseLoss` with the inverse-variance distillation
  term. Unit-correct σ projection via `K` and `depth_gt`. Triple NaN
  guard. Logs `valid_frac, mean_sigma_t, mean_sigma_R, l_distill_t,
  l_distill_R` separately so we can spot teacher dominance / collapse
  during training.

- `scripts/precompute_raft.py`: one-time RAFT precompute over trajs
  301..305 (train + val). Saves
  `data/raw_g/traj_{N:04d}/raft_predictions.npz` with arrays + metadata
  including `frames_sha1` (R3 stale-cache detection). Test trajs 306,
  307, 300 are NOT cached — teacher is training-time only.

- `scripts/train_v_distill.py`: mirrors `train_v_tdirect.py` exactly
  except (1) loss is `TDirectDistillLoss`, (2) loaders are built with
  `raft_cache=True`, (3) progress bar shows `pose / distill / valid_frac`
  alongside the total loss. Architecture is unchanged
  (`NetworkV_tdirect`), so the resulting `best.pt` is a drop-in
  replacement for Step J's V.

- `configs/step_k.yaml`: copy of `configs/step_j.yaml` + new
  `distill:` and `raft:` blocks + `paths.outputs_dir = outputs/step_k`.
  `stage2_lr_mult: 0.0` retained (lesson L1).

- `slurm/step_k_precompute_raft.sh`, `slurm/step_k_train_v.sh`,
  `slurm/step_k_train_i.sh` (FALLBACK; not in default DAG),
  `slurm/step_k_train_vi.sh`, `slurm/step_k_eval.sh`.

- `tests/test_raft_teacher.py`: synthetic smoke test for FB
  inconsistency, σ reduction, homography → pose round-trip on a known
  homography. CPU-only, ~30 s.

### MODIFIED (minimal)

- `utils/dataloader.py`:
  + `_compute_frames_sha1(traj)` helper (R3).
  + `FramePairDataset.__init__` adds `raft_cache, raft_cache_strict`
    kwargs; `_load_raft_cache(traj_id)` loads + hash-verifies the cache.
  + `__getitem__` emits `raft_R, raft_t, raft_sigma, raft_valid` keys
    when cache is enabled and the traj's cache exists.
  + `build_train_loader` accepts `raft_cache` kwarg; default False so
    every other step's config remains untouched.

- `scripts/_train_common.py`:
  + `build_val_loader_single_traj` accepts `raft_cache` kwarg
    (defaults False) so Step K can opt the val/minitest loaders into
    cache loading without breaking Step F/G/H/I/J's val loops.

### REUSE unchanged

- `scripts/train_i.py` (no distillation for I).
- `scripts/train_vi_tdirect.py` (no distillation for VI).
- `scripts/evaluate_tdirect.py` (eval is identical: V is still
  `NetworkV_tdirect`, same forward signature).
- `models/network_v_tdirect.py`, `network_i.py`, `network_vi_tdirect.py`
  — architecture unchanged.
- `models/losses_tdirect.py::TDirectPoseLoss` — wrapped by
  `TDirectDistillLoss`, not modified.
- `utils/eval.py` — plotting comprehensive from Step J.
- `utils/geometry.py::decompose_homography` — already has the
  `prior_normal` disambiguation needed by the teacher.

### Pipeline

```
step_k_precompute_raft   (one-time; ~40-60 min on rtx_pro_6000_b)
        │
        └─► step_k_train_v    (12 h cap; loads cached RAFT per batch)
              │
              ├─► [SYMLINK] outputs/step_k/i  →  ../step_j/i
              │   (manual `ln -sfn` before submit; verified by
              │    step_k_train_vi.sh's ls -L check)
              │
              └─► step_k_train_vi  (14 h cap; uses Step K V + Step J I)
                     │
                     └─► step_k_eval   (1 h cap)
```

Wallclock: ~1 h precompute + ~10 h V + ~10 h VI + ~30 min eval ≈ **~22 h**.

## Note on photometric augmentation (R6)

`PhotometricAugment` is enabled at training time
(`configs/step_k.yaml::augmentation.photometric.enabled: true`). The RAFT
cache is precomputed on **clean** (un-augmented) frames. The student then
trains on *augmented* frames but is asked to match the *clean*-frame
teacher target. This is **intentional augmentation invariance**, not a
leak: the teacher's pose is a geometric property of the frame pair, and
we want the student to be robust to photometric perturbations while still
matching that geometric target. RAFT itself was trained assuming
brightness constancy, so caching on clean frames gives the cleanest
teacher signal.

## Fallback path (R1: carpet uniformity)

If `outputs/step_k/raft_precompute_summary.json` shows
`mean_sigma_px_valid > 20` for any training traj, RAFT's flow is
uninformative on the carpet texture and distillation will fail (or hurt).
In that case:

1. Re-run precompute with the alternative weights enum:
   ```
   python scripts/precompute_raft.py --config configs/step_k.yaml \
       --weights Raft_Large_Weights.C_T_V2 --force
   ```
   (`C_T_V2` is the Things-only checkpoint, no Sintel fine-tune; sometimes
   generalises better to atypical textures.)
2. If σ is still > 20 px, RAFT-large isn't suited to this domain. Step K
   is then a controlled-negative: report and move on.

## Results (TBD after eval)

| Method | traj 305 | traj 306 | traj 307 | traj 300 | Avg |
|---|---|---|---|---|---|
| V (Step K, distilled) | TBD | TBD | TBD | TBD | TBD |
| V (Step J baseline)   | TBD | TBD | TBD | TBD | TBD |
| I  (Step J symlink)   | TBD | TBD | TBD | TBD | TBD |
| VI (Step K)           | TBD | TBD | TBD | TBD | TBD |
| VI (Step J baseline)  | TBD | TBD | TBD | TBD | TBD |

## Comparison vs Steps F / G / H / I / J

| Step | Approach                                  | V_test_avg | VI_test_avg |
|------|-------------------------------------------|------------|-------------|
| F    | sinusoid 100 s, t/d decomp                | 11.81 m    | 6.94 m      |
| G    | sinusoid 100 s, shared texture, t/d       | 5.29 m     | 2.35 m      |
| H    | sinusoid 100 s, t-direct                  | 3.86 m     | 3.36 m      |
| I    | reused H+G + anchored fusion gates        | 3.86 m\*   | 3.37 m / 4.27 m |
| J    | simple 30 s, t-direct, fresh V/I/VI       | TBD        | TBD         |
| K    | simple 30 s + RAFT distillation on V      | TBD        | TBD         |

\* symlinked from Step H.

## Decision Tree

After eval lands, compare V_test_avg in Step K vs Step J across the same
4 test trajectories.

- **K's `v_avg` < J's `v_avg` − 0.5 m AND K's `v_avg` < 1.5 m** →
  **DISTILLATION SUCCESS**. RAFT meaningfully improved V's
  representation. If `vi_avg < i_avg` too, this is a headline result.

- **K's `v_avg` ≈ J's `v_avg` ± 0.3 m** → **NEUTRAL**. Teacher knowledge
  didn't transfer, or V was already at architectural capacity.
  Investigate: train.log's `l_distill` curve — if it dropped to ~0
  quickly, the teacher signal was too easy to satisfy and didn't
  constrain V; the σ floor or λ may need tuning. If `valid_frac` was
  consistently low (< 0.3), the FB sanity-mask was over-strict.

- **K's `v_avg` > J's `v_avg` + 0.5 m** → **DISTILLATION HURT**. Likely
  the σ projection is wrong-flavor or RAFT's homography path produces
  systematically biased poses on carpet (R1 / R2). Check
  `mean_sigma_px_valid` in the precompute summary; if it's > 20 px,
  RAFT had no real signal and the inverse-variance term was just noise.
  Try the C_T_V2 fallback or roll back to Step J.

## Cross-checks before applying the tree

- `i_305` in Step K must match Step J's `i_305` exactly (symlink
  integrity; printed by `step_k_eval.sh`).
- `valid_fraction` per traj in `raft_precompute_summary.json` should be
  > 0.7 for trajs 301-304. If some traj has `valid_fraction < 0.5`, the
  precompute settings may need tuning before training.
- `train.log` rows: `l_pose, l_distill_t, l_distill_R, valid_frac,
  mean_sigma_t, mean_sigma_R` should all be finite throughout training
  (NaN guards verified, but worth eyeballing).
