# Phase 2 — Deep and Un-Deep Visual-Inertial Odometry (WPI RBE/CS549)

> Folder on disk is `Code/Phase 2/` (with a space). All commands below assume
> you `cd "Code/Phase 2"` first so paths are relative.

## 1. Overview

Three learned frame-to-frame VIO networks for a down-facing drone over a
textured plane:

| Network | Inputs                               | Output               |
|---------|--------------------------------------|----------------------|
| **V**   | stacked RGB frame pair `(6, H, W)`   | relative pose `(R, t)` |
| **I**   | 10 IMU samples `(10, 6)`             | relative pose `(R, t)` |
| **VI**  | frame pair + 10 IMU samples          | relative pose `(R, t)` |

Four architectural fixes over group10 (Pawar & Rane 2024, CS549 project):

1. **Decomposed-homography parameterization** on the vision branch.
2. **Mixed multi-trajectory training** (one shuffled dataset).
3. **Analytic scale fusion** via `d_imu = ||t_imu|| / ||td_v||`.
4. **Motion-magnitude weighted sampling** (soft rebalancing instead of strict
   round-robin).

Each fix has a matching Tier-1 failure-mode mitigation (see §10 below).

## 2. Data protocol

- **8 trajectories total**: 4 training + 3 test + 1 minitest validation.
- Training IDs 0..3: random sum-of-sinusoids.
- Test IDs 100, 101, 102: flat_spiral, figure_8, clover.
- Validation ID 200: minitest (short clover, reduced amplitude, seen texture).
- Drone-dynamics-coupled attitude (thrust-vector alignment). Yaw is a free
  sinusoid.
- **Photograph textures**: one `.webp` per trajectory drawn from a 7-image pool
  (see §3). Plane sizing uses a **resolution-aware_v1 policy**:
  `plane_longer_edge_m = max(image_longer_px / density_target, min_longer_edge_m)`
  with defaults `density_target = 60 px/m`, `min_longer_edge_m = 20 m`.
  High-res textures get larger planes (and stay sharp at 6 m altitude);
  low-res textures hit the 20 m confinement floor so the trajectory always fits.
- **100 s at 1000 Hz** → 100001 poses; every 10th frame rendered → 10001 PNGs.
  Minitest is 2 s → 2001 poses, 201 frames.
- Fixed K = `[[320, 0, 160], [0, 320, 120], [0, 0, 1]]`.

## 3. Texture setup (run once before data generation)

Place your **7 image files** in `Data/textures_raw/`. Formats accepted:
`.jpg`, `.jpeg`, `.png`, `.webp`. Sizes and aspect ratios may vary. Under the
resolution-aware_v1 policy, high-res images (≥ 1200 px) get density-sized
planes (~20–90 m); lower-res images (~600 px) get a 20 m plane at reduced
effective density (texture will look blurry but renders). Hard reject only
below 10 m implied plane.

Preprocess:
```bash
sbatch slurm/prepare_textures.sh
# or directly:
python scripts/prepare_textures.py
```

This creates `Data/textures_processed/` (normalized `.webp` at quality 95),
records originals in `manifest.json`, and splits into
`Data/textures_train/` (5 images: tex_00..tex_04) and
`Data/textures_test/` (2 images: tex_05..tex_06).

Preview all 7 on a centered plane:
```bash
sbatch slurm/preview_textures.sh
```
Review `outputs/sanity/texture_preview_all.png`. Every panel must show a
richly-textured surface with no uniform regions.

Optional full scene sanity:
```bash
sbatch slurm/preview_scene.sh
```
Writes `outputs/sanity/scene_preview.png` (2×2 camera-pose grid) and
`outputs/sanity/plane_geometry.png` (top-down schematic).

### Texture assignment policy

| Trajectory | Pool  | File            | Purpose                                 |
|------------|-------|-----------------|-----------------------------------------|
| 0          | train | `tex_00.webp`   | training                                |
| 1          | train | `tex_01.webp`   | training                                |
| 2          | train | `tex_02.webp`   | training                                |
| 3          | train | `tex_03.webp`   | training                                |
| 100        | test  | `tex_05.webp`   | held-out shape × **held-out texture**   |
| 101        | test  | `tex_06.webp`   | held-out shape × **held-out texture**   |
| 102        | train | `tex_00.webp`   | **held-out shape × seen texture**       |
| 200        | train | `tex_04.webp`   | minitest validation                     |

Trajectory 102 intentionally reuses a training texture so we can isolate pure
shape generalization from shape + texture generalization.

## 4. What "generalization" means

Test-time evaluation measures two types of generalization:

- **Trajectory shape** — flat_spiral / figure_8 / clover are never seen
  during training (only sum-of-sinusoids is).
- **Texture appearance** — test trajectories 100, 101 use the 2 held-out
  textures; test trajectory 102 uses a training texture.

Photometric augmentation during training (ColorJitter + Gaussian noise +
stochastic grayscale, applied identically to both frames of a pair) further
decouples the learned representation from absolute texture appearance.

## 5. Environment setup

```bash
bash setup_env.sh            # creates .venv/ and installs requirements
bash setup_blender.sh        # downloads Blender 4.0 LTS to blender_app/
source .venv/bin/activate
```

## 6. SLURM pipeline

```bash
bash slurm/submit_all.sh     # DAG: texprep → gen → sanity → train → eval → studies
```

Data generation is sharded:
- `slurm/gen_data_shard0.sh` → IDs 0, 1, 100, 101, 200
- `slurm/gen_data_shard1.sh` → IDs 2, 3, 102

Manual stages:
```bash
sbatch slurm/prepare_textures.sh
sbatch slurm/gen_data_shard0.sh
sbatch slurm/gen_data_shard1.sh
sbatch slurm/run_sanity.sh
sbatch slurm/train_v.sh
sbatch slurm/train_i.sh
sbatch slurm/train_vi.sh
sbatch slurm/evaluate.sh
sbatch slurm/motion_magnitude.sh
sbatch slurm/ablation.sh
TARGET=vi SEEDS="1 2 3" sbatch slurm/train_multi_seed.sh     # optional
```

## 7. Data generation (local)

```bash
bash scripts/generate_all_data.sh
```

Per-trajectory outputs under `data/raw/traj_{id:04d}/`:

```
frames/00000.png …                 rendered frames (320×240)
poses_full.npy                     (N, 7)
imu.npy                            (N, 6)
attitudes_full.npy                 (N, 3)
accel_world.npy                    (N, 3)
metadata.json                      intrinsics, plane_dims, texture_path, ...
```

## 8. Sanity checks (8)

```bash
python scripts/run_sanity_checks.py --data_dir data/raw --config configs/default.yaml
```

1. GT pose chaining (machine precision)
2. IMU double-integration (sub-metre drift)
3. Homography warping consistency
4. Motion-magnitude histogram (legacy, cross-trajectory)
5. Drone-dynamics thrust alignment (1e-4)
6. **Motion distribution** per training trajectory (Tier-1 B.5)
7. **Footprint confinement** — plane rectangle + camera footprint containment
8. **Texture assignment** — metadata.texture_path exists and matches policy

DO NOT PROCEED to training if any check fails.

## 9. Training

```bash
python scripts/train_v.py  --config configs/default.yaml
python scripts/train_i.py  --config configs/default.yaml
python scripts/train_vi.py --config configs/default.yaml \
       --pretrained_v outputs/v/best.pt \
       --pretrained_i outputs/i/best.pt
```

Each script logs per epoch to `outputs/<net>/train.log` (tab-separated):
```
epoch, stage, train_loss, val_loss, val_ATE_on_traj_0, val_ATE_on_minitest, lr
```

Best checkpoint is selected by `val_ATE_on_minitest` (OOD validation: seen
texture, novel shape). In-distribution `val_ATE_on_traj_0` is logged for
diagnostics but not used for selection.

Multi-seed training:
```bash
python scripts/train_multi_seed.py --target vi --seeds 1 2 3 --config configs/default.yaml
# writes outputs/vi_multiseed_summary.csv with mean ± std of val_ATE_on_minitest
```

## 10. Tier-1 failure-mode mitigations

1. **Smooth hovering mask** in `DecomposedHomographyLoss` / `VILoss`: the L_td
   and L_n terms are weighted by `sigmoid((|Δt| - center) / width)` so the
   gradient signal fades in / out continuously across the hover boundary.
   L_rot is never weighted (rotation always observable).
   — `models/losses.py::hovering_weight`
2. **Weighted motion-magnitude sampler**:
   `WeightedRandomSampler` with `weight = (1 / bin_count) ** alpha`.
   `alpha=0` uniform, `alpha=0.5` soft (default), `alpha=1` strict.
   — `utils/stratified_sampler.py::build_weighted_motion_sampler`
3. **Inference-time vision-direction fallback**:
   at eval, when `||td_v|| < fallback_threshold`, the IMU-only pose
   `(R_i, Δt_i)` is substituted for `(R, t)`.
   — `models/network_vi.py::NetworkVI.forward` + `scripts/evaluate.py`
4. **Multi-seed training** with summary CSV and mean ± std on
   `val_ATE_on_minitest`.
   — `scripts/train_multi_seed.py` + `slurm/train_multi_seed.sh`
5. **Dual validation metric** — `val_ATE_on_traj_0` (in-distribution) and
   `val_ATE_on_minitest` (OOD shape); best-checkpoint by the OOD metric.
   — `scripts/_train_common.py::compute_val_ate_on_trajectory`
6. **Enhanced sanity checks** — three new checks: motion distribution,
   footprint confinement, texture assignment.
   — `utils/sanity_check.py`
7. **Photometric augmentation + held-out textures** for texture-invariance.
   — `utils/dataloader.py::PhotometricAugment`

## 11. Evaluation

```bash
python scripts/evaluate.py --config configs/default.yaml \
       --checkpoints_dir outputs/ --split test
```

Writes per method:
- `outputs/trajectories/{method}_{traj}.tum`
- `outputs/plots/{method}_{traj}_{3d,xyz,rpy,topdown}.png`
- `outputs/results.csv` — long-form: `method, traj_id, metric_name, value`.
  Metrics: `ate_aligned`, `ate_endpoint`, `ate_endpoint_evo`, `seg2s_median`,
  `seg2s_p95`, `fallback_fraction` (new — share of frames that fell back to
  IMU pose).

## 12. Ablations

- **Motion-magnitude study** (`scripts/motion_magnitude_study.py`):
  `alpha ∈ {0.0, 0.5, 1.0}` plus `small_only`.
- **VI architecture ablation** (`scripts/ablation_study.py`): 4 VI variants
  (`vi_concat`, `vi_no_decomp`, `vi_concat_decompose`, `vi_full`).
- **Capacity ablation** (`scripts/capacity_ablation.py`): VI-full at small /
  medium / large backbone.

## 13. Outputs directory structure

```
outputs/
├── v/ i/ vi/                       per-network best.pt, last.pt, tb/, train.log
├── vi_concat/ vi_no_decomp/        ablation variant checkpoints
├── vi_concat_decompose/ vi_full/
├── vi_cap_{small,medium,large}/    capacity ablation checkpoints
├── v_mm_alpha_0_0/ …               motion-magnitude study subdirs
├── {v,i,vi}_seed{S}/               multi-seed training subdirs
├── {v,i,vi}_multiseed_summary.csv  multi-seed summary
├── sanity/                         8-check plots + summary.json
├── trajectories/                   {method}_{traj}.tum
├── plots/                          3D / XYZ / RPY / topdown overlays
├── studies/                        ablation_*.csv/png, motion_magnitude*.png, capacity*
└── results.csv                     long-form master results
```

## 14. File-by-file reference

### Root
- `README_phase2.md`, `requirements.txt`, `setup_env.sh`, `setup_blender.sh`.

### blender/
- `generate_data.py` — Blender headless renderer; `get_texture_for_trajectory`,
  `compute_plane_dims`, `make_photo_material`.
- `trajectory_library.py` — `sinusoid_trajectory`, `flat_spiral_trajectory`,
  `figure_8_trajectory`, `clover_trajectory`, `minitest_trajectory`,
  `confine_to_plane`, `attitude_from_acceleration`.
- `imu_simulation.py`.

### configs/
- `configs/default.yaml` — all knobs.

### models/
- `backbones.py`, `network_v.py`, `network_i.py`, `network_vi.py`, `losses.py`.

### utils/
- `geometry.py`, `dataloader.py` (with `PhotometricAugment`),
  `stratified_sampler.py` (`build_weighted_motion_sampler`,
  `build_small_only_sampler`),
  `sanity_check.py` (8 checks), `eval.py`, `training.py`.

### scripts/
- `_train_common.py`, `prepare_textures.py`, `preview_textures.py`,
  `preview_scene.py`, `run_sanity_checks.py`, `train_v.py`, `train_i.py`,
  `train_vi.py`, `train_multi_seed.py`, `evaluate.py`,
  `motion_magnitude_study.py`, `ablation_study.py`, `capacity_ablation.py`,
  `generate_all_data.sh`.

### tests/
- `test_geometry.py`, `test_dataloader.py`, `test_losses.py`,
  `test_trajectories.py`, `test_texture_assignment.py`, `test_texture_plane.py`,
  `test_confinement.py`, `test_photometric_augment.py`, `test_sampling.py`,
  `test_vi_fallback.py`.

Run all tests:
```bash
pytest tests/
```
