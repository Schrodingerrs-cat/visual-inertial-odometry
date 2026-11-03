# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Student implementation of Stereo MSCKF (S-MSCKF) visual-inertial odometry for WPI RBE/CS549. Python port of the authors' C++ reference. The starter `Code/Phase 1/msckf.py` shipped with 7 stubbed functions; all are now implemented. Everything else (feature tracking, triangulation, pruning, viewer) is provided.

The assignment targets the EuRoC MAV `MH_01_easy` sequence. Ground truth is from Vicon.

## Repo layout

- `Code/Phase 1/` — Python S-MSCKF implementation. The thing to edit.
- `Code/Phase 1/Misc/` — archived / unused files (`msckf_an.py`, `viewer.py`). Not imported at runtime.
- `msckf_vio-master/` — authors' original C++ implementation (KumarRobotics/msckf_vio). **Primary cross-reference.** When in doubt about an equation's implementation detail, check `msckf_vio-master/src/msckf_vio.cpp`.
- `1712.00036v3.pdf` — Sun et al. 2017, S-MSCKF paper.
- `ICRA07-MSCKF.pdf` — Mourikis & Roumeliotis 2007, seminal MSCKF paper.
- `Data/machine_hall/` — EuRoC bags/zips. All five Machine Hall sequences (`MH_01_easy` … `MH_05_difficult`) are extracted.
- `Results/Phase1/` — generated trajectory CSVs, trajectory PNGs, per-sequence error-plot directories, and `summary.json`.
- `Reference/Phase1/`, `Reference/Phase2/` — reference material per phase.

## Commands

All paths below assume working directory `Code/Phase 1/`.

Run the filter on an unpacked EuRoC ASL-format directory:
```
python3 vio.py --path /path/to/EuRoC_MAV_dataset/MH_01_easy          # headless
python3 vio.py --view --path /path/to/EuRoC_MAV_dataset/MH_01_easy   # with viewer (falls back to viewer_mpl)
```
`--view` prefers `uoip/pangolin` but falls back to the matplotlib-free `viewer_mpl.py` if pangolin isn't installed. Headless mode is the fast path for development.

Batch-run all five sequences and regenerate `Results/Phase1/summary.json`:
```
cd "Code/Phase 1" && python3 run_all.py
```

Unit tests for the quaternion/rotation utilities (must run from `Code/Phase 1/test/`, not via `python -m unittest` from the repo root — the `test/` package name collides):
```
cd "Code/Phase 1/test" && python3 utils_test.py
python3 feature_initialization_test.py
```

Python 3.6+. Deps: `numpy`, `scipy`, `cv2`. `pangolin` optional.

## Architecture

### Runtime wiring (`vio.py`)

Three producer/consumer threads over `queue.Queue`:
- `process_imu` → pulls raw IMU msgs, forwards them to `ImageProcessor.imu_callback` (used for stereo prediction) *and* `MSCKF.imu_callback` (buffers for batch processing).
- `process_img` → pulls stereo frames, calls `ImageProcessor.stareo_callback`, produces `feature_msg` (timestamp + list of `FeatureMeasurement(id, u0, v0, u1, v1)`) and pushes to `feature_queue`.
- `process_feature` → consumes `feature_msg`, calls `MSCKF.feature_callback`. This is the filter's main driver.

`dataset.set_starttime(offset=40.)` — sequences start from a static period so gravity/bias init has clean data.

### Filter driver (`MSCKF.feature_callback`)

On every frame, in order: `batch_imu_processing` → `state_augmentation` → `add_feature_observations` → `remove_lost_features` → `prune_cam_state_buffer` → `publish` → `online_reset`. `imu_callback` only buffers; nothing propagates until `feature_callback` fires. This means the filter is driven by image arrival, and IMU messages are consumed in bulk up to each image's timestamp.

### State layout

`StateServer` owns:
- `imu_state`: **21-dim** error state ordered `[δθ, δb_g, δv, δb_a, δp, δθ_extrinsic, δp_extrinsic]` (each block 3-dim). Note this extends the 15-dim MSCKF state with IMU↔cam0 extrinsics.
- `cam_states`: **ordered dict** keyed by IMU state id. Insertion order is load-bearing — the covariance block for the i-th camera lives at rows/cols `21 + 6i : 21 + 6(i+1)`. Don't reorder.
- `state_cov`: `(21 + 6N) × (21 + 6N)` where N is current camera count.

`map_server`: dict of `Feature` objects, keyed by feature id. Triangulation happens lazily in `remove_lost_features` / `prune_cam_state_buffer`.

### Conventions (critical)

- **JPL quaternion** `[x, y, z, w]`, scalar last. `utils.to_rotation(q)` follows Eq. (78) of the Trawny/Roumeliotis "Indirect Kalman Filter" tutorial.
- `imu_state.orientation` represents rotation **world → IMU**: `to_rotation(q) @ v_world = v_imu`.
- `cam_state.orientation` represents rotation **world → cam0**.
- `IMUState.gravity` is the world-frame gravity vector, initialized to `[0, 0, -g]` after calibration. A stationary accel reads `R(q) @ (-gravity)`.
- Quaternion error update composes as `q_new = small_angle_quaternion(δθ) ⊗ q_old`.
- `from_two_vectors(v0, v1)` returns a JPL quaternion satisfying `to_rotation(q) @ v0 = v1`.
- `T_cam0_imu` translation = origin of cam0 expressed in IMU frame (what the papers call `I p_C`).

### Observability-constrained EKF

`process_model` modifies Φ after state propagation using `orientation_null`, `velocity_null`, `position_null` (previous-step linearization points), following Hesch et al.'s OC-EKF recipe — this is non-obvious and easy to break. The nulls are updated to the post-propagation state at the end of `process_model`. Similarly, `measurement_jacobian` rebases `H_x` onto the null space of a specific 6-vector `u` constructed from the camera null state and gravity.

### Measurement update pipeline

1. `feature_jacobian` stacks per-camera `measurement_jacobian` outputs and projects onto the left null-space of `H_f` via SVD (this is the MSCKF "marginalize feature" trick).
2. `gating_test` is a Mahalanobis chi² rejection at 0.95 confidence.
3. `measurement_update` thin-QRs a tall `H`, computes `K`, updates IMU/cam states with small-angle quaternion composition, and updates covariance with the simple `(I − KH) P` form (not Joseph — matches authors' C++).

## Cross-referencing the papers

Use this map when verifying an implementation detail:

| Concept | S-MSCKF (1712.00036v3) | MSCKF (ICRA07) |
|---|---|---|
| IMU state / error state | §III-A, p. 2 | Eq. (1), (2) |
| Process model F, G | Appendix A, p. 7 | p. 3 |
| Quaternion kinematics Ω | Eq. (1), p. 3 | Eq. (6) |
| State augmentation J | Appendix B, p. 7 | Eq. (14)-(16) |
| Stereo measurement model | Eq. (4), (7), (8), p. 3, 7 | Eq. (18)-(19) (mono) |
| Null-space projection | Eq. (6), p. 4 | Eq. (23)-(24) |
| QR-reduced update | §III-B | Eq. (28)-(30) |
| OC-EKF | §III-C, p. 4 (refs Hesch et al.) | N/A |

Known discrepancy: S-MSCKF Appendix B states "J_I given above corrects the typo in Eq. (16) of [1]" and writes `-C^T ⌊p_C×⌋`. The authors' C++ uses the original MSCKF form `skew(R^T @ t)` with the "corrected" form commented out (`msckf_vio.cpp:726`). The Python port follows the C++.

## When modifying `Code/Phase 1/msckf.py`

- The ordering inside `process_model` is load-bearing: build F/G/Φ **before** calling `predict_new_state` (it mutates `imu_state`), then recompute Φ's blocks using post-propagation state for the OC correction.
- `batch_imu_processing` must increment `IMUState.next_id` **after** assigning the current id (C++ uses `next_id++` post-increment).
- `state_augmentation` relies on dict insertion order to align covariance columns; don't re-sort `cam_states`.
- If the filter diverges, first sanity check: gravity direction, quaternion convention at `initialize_gravity_and_bias`, and whether `orientation_null`/`velocity_null`/`position_null` are being updated at the end of `process_model`.
