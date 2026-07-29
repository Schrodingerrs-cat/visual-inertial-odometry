# Phase 2 — Final Submission

Final code for the Phase 2 deep visual–inertial odometry system: three
networks (V, I, VI) trained on synthetic Blender drone trajectories and
evaluated end-to-end. This folder is a strict subset of `Code/Phase 2/`
containing only the files needed to reproduce **training** and
**evaluation** of the V / I / VI models from a pre-rendered dataset.

Data generation (Blender, trajectory library, IMU simulation) and SLURM
job scripts are intentionally excluded — they live in
`Code/Phase 2/blender/` and `Code/Phase 2/slurm/` if you need them.

## File map

### Training entry points

| File | Purpose |
|---|---|
| `scripts/train_v.py` | Train **NetworkV** (vision-only, t-direct) — VGG-8 + 6D rotation head + metric translation head. |
| `scripts/train_i.py` | Train **NetworkI** (IMU-only) — bidirectional 2-layer LSTM, 6D rotation + 3D translation heads. |
| `scripts/train_vi.py` | Train **NetworkVI** (fused V+I, t-direct) — wraps V and I, adds a tanh-bounded residual refiner. Two-stage training (Stage 1: refiner only, Stage 2: full network with `stage2_lr_mult` controlling backbone LR). |
| `scripts/_train_common.py` | Shared training utilities: `TextLogger` (writes `train.log`), validation loaders, validation-ATE-on-trajectory helper, periodic-save helper. |

### Evaluation entry points

| File | Purpose |
|---|---|
| `scripts/evaluate.py` | Per-split evaluator. Loads V/I/VI checkpoints, dead-reckons predictions, runs SE(3)-aligned ATE + 2-second segment drift, writes per-trajectory plots into the config's `paths.plots_dir` and a `results.csv`. |
| `scripts/full_results_eval.py` | **Exhaustive evaluator (the deliverable script).** Runs both Step K and Step S on every trajectory in their config (train + val + test), produces every required plot per modality (V, I, VI) and modality-comparison overlays, train/val loss curves parsed from `train.log`, error-vs-distance and error-vs-speed scatters, and aggregate ATE-vs-speed and RPE-vs-speed plots. Output root: `full_results/step_<k|s>/`. |

### Models

| File | What it defines |
|---|---|
| `models/network_v_tdirect.py` | `NetworkV_tdirect` — VGG-8 backbone, 6D-continuous rotation head, direct metric translation head. |
| `models/network_i.py` | `NetworkI` — bidirectional 2-layer LSTM (hidden=128 per direction), all-layers concat → 256-d feature, separate rotation (6D) and translation (3D) heads. |
| `models/network_vi_tdirect.py` | `NetworkVI_tdirect` — wraps V and I, runs both, applies the residual refiner: `R = δR · R_v, t = t_v + δt`. |
| `models/refiner.py` | `_ResidualRefiner` — small MLP that produces (δR, δt) from concat(vfeat, ifeat). Tanh-bounded to ±0.1 rad / ±0.2 m. |
| `models/backbones.py` | `VGG8Backbone` (used by V) and `IMUEncoder` (used by I). |
| `models/losses.py` | Primitive losses: `geodesic_rotation_angle`, `hovering_weight`, `IMUPoseLoss`, `IMUPoseLoss_hover`, `GeodesicRotationLoss`. |
| `models/losses_tdirect.py` | `TDirectPoseLoss` (V branch) and `VILoss_tdirect` (VI fused) — built on the primitives in `losses.py`. |

### Utilities

| File | What it provides |
|---|---|
| `utils/dataloader.py` | `FramePairDataset`, `build_train_loader`, `build_eval_loader`. Loads consecutive frame pairs + the 10 IMU samples between them + GT relative pose. Optional photometric augmentation. |
| `utils/training.py` | `set_seed`, `get_device`, `load_config`, `save_checkpoint`, `load_checkpoint`, `cosine_warmup_lr`. |
| `utils/eval.py` | TUM trajectory I/O, `run_evo_ape` (with in-process Umeyama fallback), `segment_drift`, single-method and multi-method plot helpers, `dead_reckon_from_gt_start`, `gt_trajectory_frames`, `matrix_to_rpy`. |
| `utils/geometry.py` | Rotation conversions (Rodrigues / quaternion / 6D-continuous ↔ rotmat), SE(3) compose / invert / `dead_reckon`, `relative_pose`. |
| `utils/stratified_sampler.py` | Motion-magnitude weighted sampler used by the training dataloader. |

### Configs

| File | Step |
|---|---|
| `configs/step_k.yaml` | Approach 5 — speed-diverse training set (4 shapes × 2 speeds, 0.1–0.5 m/s). First step where VI < V on average. |
| `configs/step_s.yaml` | Approach 6 — corrected synthetic IMU pipeline + bidirectional NetworkI + speed retune (0.5–1.5 m/s). The "best" pipeline; sub-20 cm ATE on traj 412. |

## How to run

All commands assume working directory `Code/Phase2_main/` and that the
EuRoC-style synthetic dataset lives at `data/raw_g/traj_{0000..0413}/`
(rendered by `Code/Phase 2/blender/generate_data.py`; not included in
this folder).

### Train

```bash
# Vision-only (V)
python3 scripts/train_v.py --config configs/step_s.yaml

# IMU-only (I)
python3 scripts/train_i.py --config configs/step_s.yaml

# Fused (VI) — pre-trained V and I are warm-loaded then frozen in Stage 1
python3 scripts/train_vi.py --config configs/step_s.yaml \
    --pretrained_v outputs/step_s/v/best.pt \
    --pretrained_i outputs/step_s/i/best.pt
```

Each script writes checkpoints (`epoch_NNN.pt`, `last.pt`, `best.pt`),
a TensorBoard log, and a tab-separated `train.log` into
`<paths.outputs_dir>/<v|i|vi>/`.

### Evaluate

Per-split evaluation (writes plots + `results.csv` into the paths from
the config):

```bash
python3 scripts/evaluate.py --config configs/step_s.yaml \
    --checkpoints_dir outputs/step_s --split test \
    --methods v i vi --skip_missing
```

**Exhaustive evaluation across both K and S (the deliverable script):**

```bash
# Default — runs both Step K and Step S on every traj in each config:
python3 scripts/full_results_eval.py

# Or one step at a time:
python3 scripts/full_results_eval.py --steps s
python3 scripts/full_results_eval.py --steps k --output_root /scratch/full_results
```

Output structure (per step):
```
full_results/step_s/
  losses/                                  # parsed from train.log
    v_loss.png  i_loss.png  vi_loss.png  combined_loss.png
    v_val_ate.png  ... combined_val_ate.png
  train/<shape>_<speed>mps/<modality>/     # per (split, traj, model)
    gt_vs_pred_3d.png  top_xy.png
    position_xyz.png  position_error.png
    rpy.png  rpy_error.png
    err_vs_distance.png  err_vs_speed.png
    metrics.json  gt.tum  pred.tum
  train/<shape>_<speed>mps/comparison/     # V/I/VI overlay per traj
    xy_overlay.png  3d_overlay.png
    per_axis_error.png  rpy_error_overlay.png
  val/...   test/...                       # same structure
  aggregate/
    ate_per_modality.png   ate_vs_speed.png
    rpe_trans_vs_speed.png  rpe_rot_vs_speed.png
  summary.csv  summary.json
```

Every plot title, filename, and legend uses human-readable
`<shape>_<speed>mps` labels (e.g. `circle_0.5mps`, `figure8_1.0mps`)
— never opaque `traj_<id>` strings.

## Dependencies

Pinned in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

Key packages: `torch` (CUDA 12.8 wheels for Blackwell GPUs),
`torchvision`, `numpy`, `scipy`, `opencv-python-headless`, `matplotlib`,
`tensorboard`, `evo` (optional — falls back to in-process Umeyama ATE).
