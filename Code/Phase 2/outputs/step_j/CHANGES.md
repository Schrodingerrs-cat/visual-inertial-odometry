# Step J — Retrain V/I/VI on simple trajectories

## Hypothesis

V's representation ceiling across Steps F→I (best 3.86 m on held-out
sinusoid in Step H, never lower) may be due to training on **100 s
sum-of-sinusoid trajectories** that are too complex for the VGG8 backbone
to learn clean motion features. Simpler 30 s trajectories (figure-8,
circle, ellipse) may let V converge to better features.

If V can hit < 1.5 m on held-out simple trajectories, the bottleneck was
data complexity, not architecture. We use this as a clean substrate to
rebuild VI on top of a V that actually works.

## Lessons applied from prior audits

- **L1 — `stage2_lr_mult: 0.0`**: Step I used 0.25 which silently
  fine-tuned V and I during VI training, muddying the comparison. Step J
  freezes V/I across Stage 1 *and* Stage 2 (Adam · lr=0 = no weight
  update; backbones still in optimizer per L10).
- **L2 — eval-time depth**: anchored variant is not used in Step J
  (variant=tdirect), so this lesson is moot here, but the dataloader
  always provides `batch["depth"]` if needed.
- **L3 — NaN guards**: no exp/division/kornia code paths touched in this
  step.
- **L4 — TRAJ_ID_MAP and TEXTURE_ASSIGNMENT both updated**: trajs
  301..307 added to both tables in `blender/generate_data.py`.
- **L5/L6/L7 — Unix line endings, trailing newline, bash -n**: all 5
  new SLURM scripts verified `file` → "ASCII text" / "UTF-8 text" (no
  CRLF) and pass `bash -n`.
- **L8 — eval cross-checks**: `slurm/step_j_eval.sh` prints the
  `i,305,ate_aligned` row so we can sanity-check that I on the
  validation trajectory is not NaN / catastrophically off.
- **L9 — SLERP convention**: not used in Step J (variant=tdirect, no
  fusion gates).
- **L10 — reuse existing scripts**: train_v_tdirect.py, train_i.py,
  train_vi_tdirect.py, evaluate_tdirect.py all reused as-is via
  `--config configs/step_j.yaml`.

## Changes Made

### Code

- `blender/trajectory_library.py`: + `_simple_circle_sampler` /
  `simple_circle_trajectory` (R ∈ [1.5, 2.5], peak |a| ∈ [0.066,
  0.110] m/s²); + `_simple_ellipse_sampler` /
  `simple_ellipse_trajectory` (A ∈ [2.0, 3.0], B ∈ [1.0, 1.5], peak
  |a| ∈ [0.088, 0.132] m/s²); + entries in `FAMILY_SAMPLERS`. Both
  reuse the existing `SIMPLE_DURATION_S=30` / `SIMPLE_N_SAMPLES=30001`
  / `SIMPLE_ALTITUDE_M=4.0` constants.
- `blender/generate_data.py`: + `TRAJ_ID_MAP[301..307]` and
  `TEXTURE_ASSIGNMENT[301..307]` (all on `tex_03.webp`).
- `utils/eval.py`:
  + `plot_2d_topdown(...)` — promoted from per-script duplicate; now
    public, includes ATE-in-title and start-green/end-red markers.
  + `plot_per_axis_error(times, gt, pred, …)` — per-axis (x, y, z)
    error vs time with zero baseline. Diagnoses where drift accumulates.
  + `plot_xy_overlay_methods(gt, [MethodTrace], …)` — top-down XY with
    GT in solid black and one line per method, ATE in legend.
  + `plot_3d_overlay_methods(...)` — 3D analogue.
  + `plot_per_axis_error_methods(...)` — per-axis error, one line per
    method per axis.
  + `MethodTrace` dataclass to bundle `(method, pred_ts, times, ate)`.
  + ATE-suffix support and end-marker added to `plot_3d_overlay`.
- `scripts/evaluate_tdirect.py`: refactored to outer-loop trajectory,
  inner-loop method, so we can render comparison overlays after each
  trajectory's methods complete. Now writes 5 single-method PNGs per
  (method, traj) and 3 comparison PNGs per traj.
- `scripts/_train_common.py`: + `generate_training_plots(...)` — runs
  the trained model on val + minitest trajectories and writes 4 plots
  per trajectory (3D, top-down, xyz, xyz_error). Wrapped in try/except
  so a matplotlib failure on the cluster does not kill the train job
  after a successful run.
- `scripts/train_v_tdirect.py`, `scripts/train_i.py`,
  `scripts/train_vi_tdirect.py`: each calls
  `generate_training_plots(..., method=..., tag="final")` once at
  the end of `main()`, after `writer.close()`. No other changes.
- `tests/test_simple_samplers.py`: NEW. Smoke test for the 7 sampler
  invocations matching `TRAJ_ID_MAP[301..307]` — verifies shape,
  altitude, validate_trajectory, and footprint fit. Runs in <1 s.

### Config

- `configs/step_j.yaml`: NEW. Copy of `configs/step_h.yaml` with
  `train_ids=[301,302,303,304]`, `val_traj_id=305`,
  `minitest_traj_id=305`, `test_ids=[305,306,307,300]`,
  `stage2_lr_mult=0.0`, `paths.outputs_dir=outputs/step_j`.

### Pipeline

- V trained from scratch on simple shapes (NOT reused from Step H).
- I trained from scratch on simple shapes (NOT reused from Step G).
- VI trained from scratch using the new V and I as warm-starts;
  backbones FROZEN across both stages.

### SLURM submission graph

```
step_j_render        (no deps; 7 trajs × ~10 min ≈ 70 min, 4 h cap)
step_j_train_v       (after step_j_render)
step_j_train_i       (after step_j_render — parallel with train_v)
step_j_train_vi      (after train_v AND train_i)
step_j_eval          (after step_j_train_vi)
```

Wallclock estimate: ~4 h render + ~6–10 h V/I parallel + ~10 h VI +
~30 min eval = ~21 h end-to-end.

## Data

- Reused: `data/raw_g/traj_0000..0003/`, `traj_0300/` from Steps G/H/I
  (no re-render).
- Added: `data/raw_g/traj_0301/` … `traj_0307/` rendered by
  `slurm/step_j_render.sh` from the new sampler families on
  `tex_03.webp`. 30 s × 1000 Hz IMU → 3001 frames at 640 × 480.

## Results (TBD after eval)

| Method | traj 305 (val/figure-8) | traj 306 (circle) | traj 307 (ellipse) | traj 300 (figure-8 bonus) | Avg |
|---|---|---|---|---|---|
| V  | TBD | TBD | TBD | TBD | TBD |
| I  | TBD | TBD | TBD | TBD | TBD |
| VI | TBD | TBD | TBD | TBD | TBD |

## Comparison vs Steps F / G / H / I

| Step | Train data        | V_test_avg | I_test_avg | VI_test_avg |
|------|-------------------|------------|------------|-------------|
| F    | sinusoid 100 s    | 11.81 m    | 3.46 m     | 6.94 m      |
| G    | sinusoid 100 s    | 5.29 m     | 1.78 m     | 2.35 m      |
| H    | sinusoid 100 s    | 3.86 m     | 1.78 m\*   | 3.36 m      |
| I    | reused H+G        | 3.86 m\*   | 1.78 m\*   | TBD         |
| J    | simple 30 s       | TBD        | TBD        | TBD         |

\* symlinked from earlier step.

## Decision Tree

After eval lands, three numbers matter: `V_test_avg`, `I_test_avg`,
`VI_test_avg`. Apply in order:

- **`V_test_avg < 1.5 m AND VI_test_avg < I_test_avg`** →
  **HEADLINE SUCCESS**. Simple-shape training fixes V quality AND fusion
  works. Adopt Step J as the primary pipeline; report this run.
- **`V_test_avg ∈ [1.5, 3.0) m AND VI_test_avg ≈ I_test_avg`** →
  **PARTIAL SUCCESS**. V improves with simple training; VI is at least
  neutral. Reportable as "fusion is now non-regressing".
- **`V_test_avg > 3 m`** → **FUNDAMENTAL FAILURE**. V doesn't converge
  even on simple data. Bug is fundamental — loss design, architecture,
  or photometric pipeline. Stop adding fusion variants and instead
  investigate V's per-frame loss curves and gradient flow.

Cross-checks before applying the tree (built into
`slurm/step_j_eval.sh`):

- All `i_*` rows non-NaN, single-digit metres (no catastrophic IMU
  divergence).
- `i_305` ≤ `i_300` (validation traj is similar shape, similar
  expected error).
- Per-axis `xyz_error` plots for `vi_305` show the gate isn't drifting
  monotonically on any axis (would indicate frozen-V bias).
