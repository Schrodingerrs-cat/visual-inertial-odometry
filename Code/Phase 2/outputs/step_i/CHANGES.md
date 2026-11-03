# Step I — Photometric self-validation + IMU-anchored fusion

## Hypothesis

Step H eval (texture controlled, t-direct V):

```
V_traj_3  = 3.86 m
I_traj_3  = 1.78 m   (symlinked from Step G)
VI_traj_3 = 3.36 m
```

The (t/d) decomposition was a 27 % bug — t-direct is now the V baseline.
The remaining problem is that **VI is worse than I**: the fusion has no
quality signal that tells it *when* V is unreliable, so V's error leaks
into the fused output even on frames where I would have been a better
anchor.

Step I addresses that by adding two ingredients:

1. **Photometric self-validation.** Free per-frame confidence ``rho_v ∈
   [0, 1]``: warp ``img_a`` to ``img_b`` under the network's predicted
   ``(R_v, t_v)`` and the known ground-plane assumption, then squash the
   per-sample mean residual: ``rho_v = exp(-8 · MAE)``. Big residual → low
   ``rho_v`` → V is probably wrong on this frame.
2. **Soft IMU-anchored fusion.** Two small gating MLPs produce per-sample
   weights ``alpha_t, alpha_R ∈ [0, 1]``:
   ```
   t_fused = alpha_t · t_v + (1 − alpha_t) · t_i
   R_fused = SLERP(R_i, R_v, alpha_R)
   ```
   Gate inputs include ``rho_v``, the V↔I direction agreement, and the
   per-branch translation magnitudes. Both gates are initialised neutral
   (final-layer weight + bias zeroed → ``sigmoid(0) = 0.5`` for every
   sample at step 0) and learn from ``L_gate_supervision``: a binary cross-
   entropy targeting "trust the branch closer to GT on this sample."

We additionally add a **second eval point** — the simple 30 s figure-8
trajectory ``traj_id 300`` — to test whether VI clearly beats I on benign
motion where IMU drift is short and fusion has the most room to help.

V's architecture is **not** touched (V stays at t-direct); ``data/raw_g/``
is **not** re-rendered for trajs 0..3; only the new traj 300 is rendered,
re-using ``tex_03.webp`` to keep the photometric distribution consistent.

## Changes Made

### Code Changes — additive only (per Step I DO NOT list)

- ``blender/trajectory_library.py``: + ``SIMPLE_DURATION_S=30``,
  ``SIMPLE_N_SAMPLES=30001``, ``SIMPLE_ALTITUDE_M=4.0``;
  + ``_simple_figure8_sampler`` (one X period, two Y crossings → smooth
  figure-8 in XY at constant altitude, ``|v|_max ≲ 0.6 m/s``,
  ``|a|_max ≲ 0.18 m/s²``); + ``simple_figure8_trajectory``;
  + ``FAMILY_SAMPLERS["simple_figure8"]``.
- ``blender/generate_data.py``: + ``TRAJ_ID_MAP[300] = ("test",
  "simple_figure8", 2003)``; + ``TEXTURE_ASSIGNMENT[300] = ("train",
  "tex_03.webp")``.
- ``utils/geometry.py``: + ``slerp_matrices(R0, R1, alpha)`` — geodesic
  interpolation on SO(3) via ``R0 · exp(α · log(R0ᵀ R1))``, using the
  existing ``matrix_to_rodrigues`` and ``rodrigues_to_matrix``.
- ``models/network_vi_anchored.py``: NEW. ``NetworkVI_anchored`` =
  ``NetworkV_tdirect`` + ``NetworkI`` + two gate MLPs + photometric self-
  validation block (intrinsics K registered as buffer, OpenGL→CV basis
  flip applied inline; same homography formula as
  ``losses.py::photometric_warp_residual`` but reduced per-sample so
  ``rho_v`` is shape ``(B,)``). Gates initialised neutral; load_pretrained
  warm-starts V and I from external checkpoints with ``strict=False``.
- ``models/losses_anchored.py``: NEW. ``GateSupervisionLoss``
  (``BCE(alpha, target)`` with ``target = (err_other > err_self).float()``,
  detached) and ``VILoss_anchored``:
  ```
  L = w_rot · L_rot_fused + w_t · L_t_fused
    + 0.5 · (w_rot · L_rot_v + w_t · L_t_v)
    + 0.5 · (w_rot · L_rot_i + w_t · L_t_i)
    + w_gate · (L_gate_t + L_gate_R)
  ```
  Translation losses are L1, hover-weighted by the same
  ``hovering_weight(|t_gt|)`` sigmoid mask used everywhere else.
- ``configs/step_i.yaml``: NEW. Copy of ``configs/step_h.yaml`` with
  ``data.test_ids: [3, 300]``, ``model.variant: anchored``,
  ``loss.w_gate: 1.0``, plus ``model.default_altitude_m`` /
  ``model.rho_decay`` for the eval-time photometric warp.
- ``scripts/train_vi_anchored.py``: NEW. Mirrors
  ``scripts/train_vi_tdirect.py``. Forward pass receives
  ``depth_gt = batch["depth"]`` so training-time ``rho_v`` uses the true
  per-frame altitude. Stage 2 head_params is
  ``model.gate_t.parameters() + model.gate_R.parameters()``.
- ``scripts/evaluate_anchored.py``: NEW. Mirrors
  ``scripts/evaluate_tdirect.py``. For ``vi`` it accumulates per-frame
  ``rho_v / alpha_t / alpha_R / t_v`` and writes a sidecar
  ``outputs/step_i/eval/gate_stats.json`` with the per-trajectory summary
  consumed by the *Gate Behaviour* table below.
- ``slurm/step_i_render.sh``: NEW. Render traj 300 with
  ``--force_texture tex_03.webp``. 1 h cap (~10 min runtime).
- ``slurm/step_i_train_vi.sh``: NEW. Train ``NetworkVI_anchored``
  (depends on render). 14 h cap. Defensive checks at job start verify
  the V/I symlinks resolve and traj 300 was rendered.
- ``slurm/step_i_eval.sh``: NEW. Run ``evaluate_anchored.py`` on
  ``test_ids = [3, 300]``. 1 h cap. Echoes ``results.csv`` and
  ``gate_stats.json``.

### Code Changes — pre-existing files left untouched

Per the Step I DO NOT list:

- ``models/network_v.py``, ``models/network_v_tdirect.py``,
  ``models/network_i.py``, ``models/network_vi.py``,
  ``models/network_vi_tdirect.py`` — unchanged. Step I introduces
  ``NetworkVI_anchored`` alongside.
- ``models/losses.py``, ``models/losses_tdirect.py`` — unchanged.
  Step I adds ``models/losses_anchored.py``.
- ``configs/default.yaml``, ``configs/step_f.yaml``,
  ``configs/step_g.yaml``, ``configs/step_h.yaml`` — unchanged.
- ``outputs/step_f/``, ``outputs/step_g/``, ``outputs/step_h/``,
  ``data/raw/`` — unchanged. Existing trajectories in ``data/raw_g/``
  are **not** re-rendered; only the new ``data/raw_g/traj_0300/`` is
  added by ``step_i_render.sh``.

### Data

- Reused ``data/raw_g/traj_000{0..3}/`` as-is (Step G renders, all using
  ``tex_03.webp``).
- Added ``data/raw_g/traj_0300/`` — rendered by Step I from the new
  ``simple_figure8`` family with ``tex_03.webp``. 30 s × 100 Hz IMU →
  3001 frames at 640×480.

### Pipeline

- V reused from Step H via symlink:
  ``outputs/step_i/v -> ../step_h/v``.
  Justification: V's architecture is unchanged in Step I; retraining
  would only add seed variance. Cross-check: ``v_traj_3`` in Step I eval
  must match Step H's ``3.86 m``.
- I reused from Step G via symlink (chained through Step H):
  ``outputs/step_i/i -> ../step_g/i``.
  Cross-check: ``i_traj_3`` in Step I eval must equal Step G's ``1.78 m``
  exactly. Same NetworkI checkpoint, no architectural change.
- VI trained from scratch (``NetworkVI_anchored``). Vision and IMU
  branches warm-start from the symlinked V/I checkpoints; the two new
  gate MLPs start from the neutral initialisation described above.
  Two-stage schedule (10 + 20 epochs), Stage 1 freezes the backbones so
  only the gates train, Stage 2 unfreezes with the backbone LR scaled by
  ``stage2_lr_mult = 0.25``.

### SLURM submission graph

```
step_i_render   (no deps; renders traj 300)
step_i_train_vi (after step_i_render; uses symlinked V/I)
step_i_eval     (after step_i_train_vi; tests on trajs 3 + 300)
```

V and I do not appear — both symlinked from earlier steps.

## Results

(Filled in by ``slurm/step_i_eval.sh`` and pasted here after the run.)

```
Step I eval — results.csv (excerpt):
  method  traj_id  metric         value
  i       3        ate_aligned    TBD       (cross-check: must equal 1.78 m)
  i       300      ate_aligned    TBD       (sanity: short traj, expect 0.5–2 m)
  v       3        ate_aligned    TBD       (sanity: must equal Step H's 3.86 m)
  v       300      ate_aligned    TBD
  vi      3        ate_aligned    TBD       (NEW)
  vi      300      ate_aligned    TBD       (NEW)
```

## Gate Behaviour

(Filled in from ``outputs/step_i/eval/gate_stats.json``.)

| Traj | mean alpha_t | std alpha_t | mean alpha_R | mean rho_v | corr(rho_v, ‖t_v − t_gt‖) | frac(α_t > 0.5) |
|------|--------------|-------------|--------------|------------|----------------------------|-----------------|
| 3    | TBD          | TBD         | TBD          | TBD        | TBD                        | TBD             |
| 300  | TBD          | TBD         | TBD          | TBD        | TBD                        | TBD             |

How to read this table:

- **mean alpha_t / mean alpha_R**: > 0.5 means the gate is trusting V,
  < 0.5 means it's trusting I, ≈ 0.5 means the gate hasn't learnt a
  preference (or the supervision target is itself ≈ 0.5 because V and I
  are similarly bad).
- **std alpha_t**: a higher std means the gate is making per-frame
  decisions; a near-zero std means the gate has collapsed to a constant.
- **mean rho_v**: average photometric self-confidence — sanity-check that
  it isn't pegged at 0 or 1.
- **corr(rho_v, ‖t_v − t_gt‖)**: this should be **negative** if
  ``rho_v`` is doing its job (high confidence → low error). A value
  near 0 means the photometric signal has no actual information for
  this trajectory; a positive value would be alarming.
- **frac(α_t > 0.5)**: the share of frames where the gate let V drive
  translation. Should track which branch is actually winning per traj.

## Comparison vs Steps F, G, H

| Variant                                      | V_traj_3 | I_traj_3 | VI_traj_3 |
|----------------------------------------------|----------|----------|-----------|
| Step F (different texture, t/d)              | 7.41     | 3.41     | 8.25      |
| Step G (shared texture, t/d)                 | 5.29     | 1.78     | 2.35      |
| Step H (shared texture, t-direct)            | 3.86     | 1.78\*   | 3.36      |
| Step I (anchored fusion + photometric gate)  | 3.86\*   | 1.78\*   | TBD       |

\* V symlinked from Step H, I symlinked from Step G. These cells must
match exactly — if they don't, the symlink broke.

| Variant                                      | I_traj_300 | V_traj_300 | VI_traj_300 |
|----------------------------------------------|------------|------------|-------------|
| Step I                                        | TBD        | TBD        | TBD         |

## Interpretation

(To be filled in after eval.)

## Decision

After eval, three numbers matter: ``VI_traj_3``, ``VI_traj_300``, and the
gate statistics in the table above.

- **VI_traj_3 < 1.78 m AND VI_traj_300 < I_traj_300** → **SUCCESS**.
  Soft fusion works. VI beats I on both. Headline result for the report.
- **VI_traj_3 within 0.2 m of I_traj_3 AND VI_traj_300 ≤ I_traj_300**
  → **PARTIAL SUCCESS**. VI no longer regresses on the harder traj
  while at least matching I on the simple one. Report as "fusion is now
  neutral; future work to make it productively positive."
- **VI_traj_3 > I_traj_3 + 0.5 m** → **FAILURE**. Investigate the gate
  stats:
  * Are ``alpha_t`` values reasonable (some high, some low)?
  * Is ``rho_v`` correlating with V's actual per-frame error
    (corr should be < 0)?
  * If both gates are stuck near 0.5: the gate-supervision loss isn't
    teaching them — diagnose ``target_alpha_t_mean`` /
    ``target_alpha_R_mean`` in the train log to see whether the
    supervision targets are themselves degenerate.

Cross-checks before applying the tree:

- ``i_traj_3`` must equal Step G's ``1.78 m`` (symlink integrity).
- ``v_traj_3`` must equal Step H's ``3.86 m`` (symlink integrity).
- ``i_traj_300`` should fall in the ``0.5–2 m`` range — short trajectory,
  IMU integration window short enough that drift is small.
