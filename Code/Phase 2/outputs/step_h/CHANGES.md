# Step H — t-direct V Experiment

## Hypothesis

Steps F (held-out sinusoid with texture confound) and G (texture controlled)
showed `V_traj_3 = 5.29 m` and `VI_traj_3 = 2.35 m` — V partially generalizes
but VI is *worse* than I alone (`I_traj_3 = 1.78 m`), suggesting the fusion
inherits problems from the V branch.

The (R, t/d, n) decomposition is a non-standard parameterization of relative
pose: V predicts a scale-free direction t/d together with the plane normal,
and the metric translation is reconstructed at use-time as `t = (t/d) * d`.
The hypothesis being tested in Step H is whether this decomposition is
itself the source of the V error.

Step H replaces the decomposition with direct metric prediction: V predicts
`(R, t)` directly (no n, no scale-free direction, no per-frame altitude
multiplication). Same backbone (VGG8), same training data (`data/raw_g`,
shared texture), same train/val/test split. If V_traj_3 drops below ~3 m,
the (t/d) parameterization was the bug. If it stays near 5 m, the bottleneck
is elsewhere (backbone capacity, training procedure, fusion architecture)
and Step I (RAFT end-to-end) is warranted.

## Changes Made

### Code Changes — new files only (per Step H DO NOT list)

- `models/network_v_tdirect.py`: NEW. `NetworkV_tdirect` class. VGG8 backbone
  with two MLP heads — `head_rot` (6D continuous rotation) and `head_t`
  (3D metric translation). 9-dim head total. No `head_td`, no `head_n`.
- `models/network_vi_tdirect.py`: NEW. `NetworkVI_tdirect` class. Wraps
  `NetworkV_tdirect` + `NetworkI` + `_ResidualRefiner` (reused from
  `models/network_vi.py`). Fusion is residual only:
  `R_final = dR @ R_v`, `t_final = t_v + dt`. No scale head, no scale mixer
  — those existed only because the (t/d) V branch produced a direction; with
  metric translation they have no role.
- `models/losses_tdirect.py`: NEW.
  - `TDirectPoseLoss`: V-branch loss = `w_rot · L_rot + w_t · L_t`. L_t is
    L1 over the 3 axes, smooth-hover-weighted by `hovering_weight(|t_gt|)`
    (the same sigmoid mask as the (t/d) loss; it was always computed from
    `|t_rel|` in metres, not from `|t/d|`, so no recomputation was needed).
  - `VILoss_tdirect`: V-branch loss + IMU-branch loss + fused-pose
    `w_rot · L_rot_fused + w_trans · L_t_fused`. No `l_d` (no scale head),
    no `l_warp` (no plane-normal supervision).
  - Removed sub-losses vs `models/losses.py`: `l_n`, `l_td`, `l_warp`, `l_d`.
- `configs/step_h.yaml`: NEW. Copy of `configs/step_g.yaml` with:
  - `data.raw_dir: data/raw_g` (reuses Step G data — no re-render).
  - `data.minitest_traj_id: 3` (drives best.pt by ATE on traj 3).
  - `model.variant: tdirect` (asserted by trainers and eval script).
  - `loss.w_t: 5.0` (replaces `w_td: 20.0`).
  - Removed: `loss.w_td`, `loss.w_n`, `loss.w_warp`, `loss.w_d`.
  - Removed: `eval.fallback_threshold` (no `||td_v||` collapse to fall back
    from).
  - `paths.outputs_dir: outputs/step_h` (and the four sibling paths
    rerooted).
- `scripts/train_v_tdirect.py`: NEW. Mirrors `train_v.py`. Imports
  `NetworkV_tdirect` and `TDirectPoseLoss`. `_forward_rel` returns `(R, t)`
  from the model directly (no `td * d_gt` step). No `K` argument (no warp
  loss). Asserts `cfg.model.variant == 'tdirect'`.
- `scripts/train_vi_tdirect.py`: NEW. Mirrors `train_vi.py`. Same two-stage
  freeze/unfreeze schedule. Stage-2 head_params is just
  `model.refiner.parameters()` (no scale_head, no mixer). No `K`. Asserts
  `cfg.model.variant == 'tdirect'`.
- `scripts/evaluate_tdirect.py`: NEW. Methods `v / i / vi`. For `v`, the
  model returns metric `t` directly (no `t = td * batch["depth"]`
  reconstruction). For `i` and `vi`, prediction is unchanged. No
  `use_imu_fallback` flag (no near-zero direction-collapse failure mode in
  the t-direct vision branch).

### Code Changes — pre-existing files left untouched

Per the Step H DO NOT list:
- `models/network_v.py`, `models/losses.py` — unchanged.
- `scripts/train_v.py`, `scripts/evaluate.py` — unchanged.
- `configs/default.yaml`, `configs/step_f.yaml`, `configs/step_g.yaml`
  — unchanged.
- `outputs/step_f/`, `outputs/step_g/`, `data/raw/`, `data/raw_g/`
  — unchanged.

### Data

- Reused `data/raw_g/` from Step G as-is. No re-render. The hypothesis is
  about V's parameterization, not about texture or scene; the same RGB
  frames, same poses, same IMU give the cleanest comparison vs Step G.

### Pipeline

- I reused from Step G via symlink: `outputs/step_h/i -> ../step_g/i`.
  Justification: I doesn't depend on V's parameterization at all
  (`NetworkI` consumes only IMU). Retraining would waste ~6 h compute and
  introduce an unnecessary source of seed variation. The cross-check is
  that `I_traj_3` in Step H eval must equal Step G's `1.78 m` exactly.
  If it doesn't, the symlink broke or the eval script is loading the wrong
  checkpoint.
- V trained from scratch with t-direct head (no warm-start from Step G's V,
  since the head shapes differ).
- VI trained from scratch with the new V (Step H V) and reused I (symlink).
  Identical two-stage schedule to Step G's VI, just with the new vision
  branch and the new (no scale-head) fusion head.

### SLURM submission graph

```
step_h_train_v   (no deps)
step_h_train_vi  (after step_h_train_v)
step_h_eval      (after step_h_train_vi)
```

I doesn't appear — symlinked from Step G.

## Results

(Filled in by `slurm/step_h_eval.sh` and pasted here after the run.)

```
Step H eval — results.csv (excerpt):
  method  traj_id  metric         value
  i       3        ate_aligned    TBD       (should equal Step G's 1.78)
  v       3        ate_aligned    TBD       (NEW)
  vi      3        ate_aligned    TBD       (NEW)
```

## Comparison vs Steps F, G

| Variant                            | V_traj_3 | I_traj_3 | VI_traj_3 |
|------------------------------------|----------|----------|-----------|
| Step F (different texture, t/d)    | 7.41     | 3.41     | 8.25      |
| Step G (shared texture, t/d)       | 5.29     | 1.78     | 2.35      |
| Step H (shared texture, t-direct)  | TBD      | 1.78\*   | TBD       |

\*I reused from Step G via symlink (`outputs/step_h/i -> ../step_g/i`).

## Interpretation

(To be filled in after eval — was the (t/d) decomposition the bug, or not?)

## Decision

Apply the decision tree below to the `V_traj_3` ATE-aligned value reported
in `outputs/step_h/results.csv`:

- **V_traj_3 < 3 m** — CONFIRMED: (t/d) was the bug. Adopt t-direct as
  primary V variant; re-run multi-trajectory eval (trajs 100/101/102) to
  verify generalization outside the sinusoid family; update final report.
- **3 m ≤ V_traj_3 < 5 m** — AMBIGUOUS: (t/d) was part of the issue. Ship
  with t-direct (better than 5.29) or push further with Step I.
- **V_traj_3 ≥ 5 m** — REJECTED: (t/d) was not the dominant bug. Bottleneck
  is backbone capacity, training procedure, or fusion. Proceed to Step I
  (RAFT end-to-end).

Cross-check before applying the tree: `I_traj_3` must equal the Step G
value of `1.78 m`. If it doesn't, the symlink broke or the eval is loading
the wrong I checkpoint — fix that before drawing conclusions about V.
