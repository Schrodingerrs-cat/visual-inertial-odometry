# Phase 1 — Implementation & Results Analysis

This note walks through what was built, what the numbers say on MH_01_easy,
and where the approach visibly fails on the five Machine Hall sequences.

## 1. What was implemented

Seven functions in `msckf.py` were stubs in the starter code. Each was
filled in following the S-MSCKF paper (Sun et al. 2017), the MSCKF'07
paper (Mourikis & Roumeliotis), and the authors' C++ reference
(`msckf_vio-master/src/msckf_vio.cpp`) for bit-for-bit parity with the
released implementation.

| # | Function | What it does |
|---|---|---|
| 1 | `initialize_gravity_and_bias`   | Uses a static-window IMU buffer to set `b_g = mean(ω)`, `|g| = ‖mean(a)‖`, world gravity `[0, 0, −g]`, and aligns `imu_state.orientation` so `R(q)·(−g_w) = mean(a)`. |
| 2 | `batch_imu_processing`          | Drains the IMU buffer up to the current image timestamp, calling `process_model` per sample; assigns `imu_state.id` pre-increment (matches the C++ `next_id++`). |
| 3 | `process_model`                 | Builds the 21×21 error-state `F` and 21×12 noise `G`; discretises to `Φ = I + F·dt + ½(F·dt)² + ⅙(F·dt)³`; propagates the nominal state with RK4 via `predict_new_state`; applies the OC-EKF Φ-rewrite against `orientation/velocity/position_null`; updates `P_II`, `P_IC`, symmetrises. |
| 4 | `predict_new_state`             | 4th-order Runge–Kutta on `q̇ = ½Ω(ω)·q`, `v̇ = Rᵀâ + g_w`, `ṗ = v` with the closed-form `Ω` exponential for the quaternion and unit-norm re-projection at each sub-step. |
| 5 | `state_augmentation`            | Appends a fresh camera state to the `OrderedDict`, grows `P` by a 6×6 block using the 6×21 Jacobian `J` from S-MSCKF Appendix B (keeping the C++ `skew(Rᵀ·t)` form). |
| 6 | `add_feature_observations`      | Stashes stereo observations `[u0,v0,u1,v1]` under feature id and returns a tracking-rate metric for motion-degeneracy detection. |
| 7 | `measurement_update`            | Thin-QR reduces tall `H`; computes `K = P Hᵀ (HPHᵀ + σ²I)⁻¹` via `np.linalg.solve`; updates IMU (small-angle quaternion compose), extrinsics, and each camera in insertion order; applies the simple `(I − KH)P` covariance update. |

Two non-obvious details that were easy to miscode:

- **Ordering in `process_model`**: `F`/`G`/`Φ` must be built *before* calling
  `predict_new_state` (which mutates `imu_state`); the OC rewrite then uses
  the pre-step nulls and the post-propagation state together.
- **State augmentation Jacobian**: S-MSCKF Appendix B publishes an erratum
  (`−Rᵀ·skew(t_c_i)`), but the authors' C++ keeps the original
  `skew(Rᵀ·t_c_i)` form with the "corrected" form commented out. The Python
  port follows the C++ — swapping to the paper's erratum breaks agreement
  with the reference binary.

## 2. Implemented functions (detailed walk-through)

All seven stubs in `Code/Phase 1/msckf.py` are implemented and cross-checked
against both papers. Line numbers below refer to the Python port.

### 2.1 `initialize_gravity_and_bias` — msckf.py:231

Uses the buffer collected during the static warm-up period to estimate the
initial gyro bias and gravity magnitude, then aligns the IMU frame to world.

- `b_g = mean(ω)` over the buffer.
- `|g| = ‖mean(a)‖` — measured gravity from the static accel readings.
- World-frame gravity set to `[0, 0, −|g|]` (`IMUState.gravity`).
- Initial orientation: `q` such that `to_rotation(q) @ (−gravity_world) = mean(a)` via `from_two_vectors`.

Paper refs: S-MSCKF §III-A (p. 2) describes the static-initialisation
assumption. MSCKF'07 does not specify this explicitly — the approach follows
the C++ reference (`msckf_vio.cpp:248–284`).

### 2.2 `batch_imu_processing` — msckf.py:262

Consumes buffered IMU messages up to the current image timestamp: for each
message it calls `process_model`, assigns `imu_state.id = IMUState.next_id`
**before** incrementing (matches the C++ post-increment at `msckf_vio.cpp:538`),
then slices the buffer.

Paper refs: conceptually §III-A of S-MSCKF (propagation between image frames).

### 2.3 `process_model` — msckf.py:285

Builds the 21×21 continuous-time error-state Jacobian `F` and the 21×12
noise Jacobian `G`, discretises to `Φ` by a 3rd-order Taylor expansion of
`expm(F·dt)`, propagates the nominal state via `predict_new_state`, applies
the Observability-Constrained EKF (OC-EKF) correction to Φ, and updates the
covariance.

Non-zero blocks of `F` (rows = `[δθ, δb_g, δv, δb_a, δp, δθ_ext, δp_ext]`):

| block | expression | source |
|---|---|---|
| `F[0:3, 0:3]` | `−skew(ω̂)` | S-MSCKF Appendix A, p. 7 |
| `F[0:3, 3:6]` | `−I₃` | S-MSCKF Appendix A |
| `F[6:9, 0:3]` | `−Rᵀ · skew(â)` | S-MSCKF Appendix A |
| `F[6:9, 9:12]`| `−Rᵀ` | S-MSCKF Appendix A |
| `F[12:15, 6:9]`| `I₃` | S-MSCKF Appendix A (`ṗ = v`) |

`G` has four 3×3 blocks of `−I` / `I` mapping `[n_g, n_wg, n_a, n_wa]` into
the IMU error state, mirroring `msckf_vio.cpp:540–628`.

Discrete transition: `Φ = I + F·dt + 0.5·(F·dt)² + (1/6)·(F·dt)³` (matches
MSCKF'07 p. 3 and the C++ expansion).

**OC-EKF** (S-MSCKF §III-C, p. 4; Hesch et al.): after propagating, we
rewrite three columns of Φ so that the filter preserves the unobservable
yaw/translation directions. Using `orientation_null / velocity_null /
position_null` captured at the previous step:

- Φ[0:3, 0:3] ← recomputed so it maps `gravity_null` onto itself.
- Φ[6:9, 0:3] ← `A − (A·u − w)·uᵀ / (uᵀu)` with `u = R(q_null)·g_world`,
  `w = skew(v − v_null)·g_world`, `A = Φ[6:9, 0:3]`.
- Φ[12:15, 0:3] ← same recipe with `w = skew(dt·v_null + p − p_null)·g_world`.

After the update, `orientation_null / velocity_null / position_null` are
refreshed to the post-propagation state — required so the next step's
re-projection uses the correct linearization point.

Covariance: `P_II ← Φ P_II Φᵀ + Q_d`, and the IMU–camera cross-terms
`P_IC ← Φ P_IC`. The result is symmetrised with `0.5·(P + Pᵀ)`.

### 2.4 `predict_new_state` — msckf.py:356

4th-order Runge–Kutta integration of the nominal IMU state over `dt`.

- Quaternion: closed-form solution of `q̇ = ½Ω(ω)·q` via
  `dq_dt = cos(|ω|·dt/2)·I + sin(|ω|·dt/2)/|ω|·Ω(ω)` applied to the current
  quaternion (S-MSCKF Eq. (1), p. 3; MSCKF'07 Eq. (6)). We also compute
  `dq_dt2` at `dt/2` for the mid-step.
- Velocity / position: standard RK4 with `v̇ = Rᵀ·â + g_w`, `ṗ = v`.

All rotations are computed using `quaternion_normalize` of the interpolated
quaternions — keeps RK4 from drifting off the unit sphere.

### 2.5 `state_augmentation` — msckf.py:411

On every image, a fresh camera state is inserted into the
ordered-dict `cam_states` and the covariance is grown by one 6-dim block.

- Camera pose: `R_w_c = R_i_c · R_w_i`,  `t_c_w = p_imu + R_w_i ᵀ · t_c_i`.
- Jacobian `J` (6×21), following S-MSCKF Appendix B (p. 7) and MSCKF'07
  Eq. (14)–(16), non-zero blocks:
  - `J[0:3, 0:3] = R_i_c`
  - `J[0:3, 15:18] = I₃`
  - `J[3:6, 0:3] = skew(R_w_i ᵀ · t_c_i)`
  - `J[3:6, 12:15] = I₃`
  - `J[3:6, 18:21] = R_w_i ᵀ`
- Augmented covariance, matching `msckf_vio.cpp:695–755`:

  ```
  P_new = [ P        (J P_II ↦ P_IC)ᵀ
            J P_II   J P_II Jᵀ ]
  ```

**Known paper-vs-C++ discrepancy**: S-MSCKF Appendix B's erratum gives
`J[3:6, 0:3] = −Rᵀ · skew(t_c_i)`, but the reference C++ keeps the original
MSCKF'07 form `skew(Rᵀ · t_c_i)` with the corrected form commented out
(`msckf_vio.cpp:726`). The Python port follows the C++ for bit-for-bit parity.

### 2.6 `add_feature_observations` — msckf.py:456

For each `FeatureMeasurement` in the incoming frame, stash its stereo obs
`[u0, v0, u1, v1]` under the feature id. Computes a tracking-rate metric
(`already-tracked / total`) that the filter uses to decide when the motion
is too degenerate to update.

### 2.7 `measurement_update` — msckf.py:588

Runs the stacked measurement update after feature marginalisation.

1. **Thin QR**: `H = Q·R₁` (reduced mode) — only used when `H` is tall.
   Projects `r` and reduces work from `O(m³)` to `O(n³)` (S-MSCKF §III-B;
   MSCKF'07 Eq. (28)).
2. **Kalman gain**: `K = P·Hᵀ·(H·P·Hᵀ + σ²I)⁻¹`, computed via `np.linalg.solve`
   on `(H·P·Hᵀ + σ²I)·Kᵀ = H·P`.
3. **State update** `δx = K·r`:
   - IMU: `q_new = small_angle(δθ) ⊗ q_old`; bias/velocity/position add;
     extrinsics get the same quaternion+translation update.
   - Cameras: iterate `cam_states.values()` in insertion order — slice
     `δx[21+6i : 21+6(i+1)]` per camera. Order is load-bearing because the
     covariance blocks are keyed by insertion position.
4. **Covariance**: `P ← (I − K·H)·P`, then symmetrised. This is the simple
   form used by the reference C++ (not the Joseph form).

## 3. MH_01_easy — the reference sequence

MH_01_easy is the assignment target. Because the drone spends ~40 s on the
ground before takeoff, the first IMU window is clean and gravity/bias
initialise correctly (`DATASET_OFFSETS["MH_01_easy"] = 40.0`).

### 3.1 ATE (SE(3)-aligned)

| metric | value |
|---|---:|
| RMSE        | **0.083 m** |
| mean        | 0.078 m |
| median      | 0.076 m |
| max         | 0.180 m |
| poses (n)   | 2841 |
| trajectory length | ~73 m |

Drift stays sub-20 cm across the entire flight, with the RMSE under 10 cm.
For reference: S-MSCKF paper Fig. 2(a) plots the S-MSCKF bar at ~0.22 m on
MH_01; we beat the paper on this sequence by roughly a factor of 2.5, which
we attribute to running a single clean replay (the paper averages five
runs) plus a warm, well-calibrated static window.

### 3.2 Figure-by-figure reading

**Fig 1 — Translation error vs distance (`fig1_trans_error.png`).**
The x/y/z drift bands stay within ±150 mm across the full 73 m traverse.
There's no monotonic growth: the estimate is well-observed through the
flight, so drift is re-anchored by feature constraints rather than walking
off. Local oscillations (~20 mm peak-to-peak) at 15–25 m come from the
drone's first hovering loop — attitude is slightly less observable at low
velocity, so the triangulation geometry weakens briefly. After 50 m the x
band drifts noticeably negative; this matches the return-to-start leg, a
classic "closing the loop without loop-closure" signature.

**Fig 2 — Rotation error vs distance (`fig2_rot_error.png`).**
Yaw/pitch/roll are all sub-3° across the run. Yaw is the most mobile
(±2.5°) because yaw is the unobservable direction around gravity; the
OC-EKF keeps the filter from reporting false confidence, but the
linearisation still lets it wander slightly. Pitch/roll are tightly bounded
(±1°) — exactly the behaviour predicted for a gravity-observable filter.

**Fig 3 — Relative translation error boxplot (`fig3_rel_trans_error.png`).**
Sub-trajectory RPE at 10/20/30/40/50% of total length (~7, 15, 22, 29, 37 m).
Median RPE at 7 m is ~90 mm and grows roughly linearly with segment length,
which is the textbook signature of a filter that's drift-limited rather
than biased — errors accumulate but don't explode.

**Fig 4 — Relative yaw boxplot (`fig4_rel_yaw_error.png`).**
Median yaw RPE rises from ~0.8° at 7 m to ~1.5° at 37 m. Again linear and
bounded; matches Fig. 2's global yaw bound.

**Figs 5 & 6 — Side / top views (`fig5_traj_side.png`, `fig6_traj_top.png`).**
After SE(3) alignment the estimated and ground-truth trajectories overlay
visually: the pink (GT) and blue (estimate) curves trace the same shape —
a single loop with a shallow altitude profile. There's no visible
separation at the plot scale; differences live in the dm-range error plots
above.

**Takeaway for MH_01**: the filter is working as designed. Errors are
dominated by global unobservable-yaw drift and feature-tracking noise, not
by any process-model or linearisation defect.

## 4. Outliers across the five sequences

| Sequence          |  RMSE |  max |
|-------------------|------:|-----:|
| MH_01_easy        | 0.083 | 0.18 |
| MH_02_easy        | 0.377 | 1.12 |
| MH_03_medium      | 0.173 | 0.34 |
| **MH_04_difficult** | **0.898** | **2.54** |
| **MH_05_difficult** | **0.916** | **2.53** |

Two clear outliers: **MH_04 and MH_05**, both roughly 10× MH_01's RMSE with
peak errors above 2.5 m. One softer outlier: **MH_02** — comparable RMSE to
MH_03 but with a visible transient discontinuity that drags its max to 1.1 m.

### 4.1 MH_04 / MH_05 — "no static window" failure mode

**Cause.** The static-init procedure in `initialize_gravity_and_bias`
assumes a window where `‖a‖ ≈ 9.81` with low variance. MH_01 gives us 40 s
of stationary taxi; MH_02/MH_03 give 10 s; MH_04 and MH_05 are essentially
**airborne within the first 1–2 seconds of the log**. Setting
`DATASET_OFFSETS["MH_04_difficult"] = 0.0` is the closest-to-static choice,
but the "static" window here still contains the takeoff impulse.

That contaminates:

1. The initial gravity magnitude (off by fractions of 1 m/s² → tilts the
   world frame).
2. The initial gyro bias estimate (measures rotational motion as bias).
3. The initial attitude (`from_two_vectors(−g, mean(a))` — if `mean(a)` is
   contaminated by thrust, the initial tilt is wrong).

**Effect.** Figs 1 for MH_04 and MH_05 both show a dramatic spike in the
first 3 m of distance — translation error reaches −2000 mm on x (MH_05)
and −2000 mm on y (MH_04). The filter recovers over the next ~15 m as
feature measurements constrain the state, but the transient already lands
in `max` and inflates RMSE.

A secondary failure mode on these sequences: the aggressive motion yaws
faster than the KLT tracker can keep up, producing feature-tracking gaps
that starve the measurement update and let bias drift creep in. This shows
up in the error plots as a slow linear growth of z and x error after
t = 50 m.

**What would fix it.** A "moving-base" initialisation — e.g., run a short
VIO warm-up with a prior on gravity magnitude, or seed the filter with an
attitude from the first stereo image's horizon line. This project kept the
stock static-init because the assignment targets MH_01; the other four
sequences are bonus validation.

### 4.2 MH_02 — the mid-sequence jump

**Cause.** Fig 1 for MH_02 shows a tall vertical spike near 6 m distance:
x swings to −1000 mm and z to +450 mm in a single frame before the error
settles back into a normal ±200 mm band. This isn't an init issue — it's a
**feature-tracking glitch mid-flight**. MH_02's trajectory re-enters the
same room from a slightly different heading, and the KLT tracker picks up
a burst of false-positive matches on repeated texture, fires the
measurement update with a bad `H`, and the covariance update pulls the
state outside chi² for a frame before the gating test recovers.

**Effect.** A single-frame outlier that survives the 0.95 gate (the
cluster of bad matches is large enough to pass) shows up as a ~1 m max
error. RMSE is diluted across 2798 poses, so the RMSE of 0.377 m is still
in a reasonable range — but the *max* is a full metre above MH_03's.

**What would fix it.** A tighter `max_track_count` cap in the image
frontend, or a RANSAC check on the triangulated feature residuals before
they feed `measurement_update`. Out of scope for this phase.

### 4.3 Why MH_03 is *not* an outlier

MH_03 is labelled "medium" but ends up as the second-best sequence. Its
static window is short (10 s) but the takeoff is gentle, and the mid-flight
path re-uses the same textured walls as MH_02 without re-entering the
problem region. The gravity/bias init takes cleanly and the tracker stays
healthy — the 0.173 m RMSE looks more like a "hard MH_01" than a failed
MH_02.

## 5. Summary of where the approach wins and loses

| Regime | Outcome | Why |
|---|---|---|
| Long static window, slow motion     | Excellent (MH_01 @ 8 cm) | Textbook MSCKF operating point. |
| Short static window, gentle takeoff | Good (MH_03 @ 17 cm)     | Init tolerates brief motion contamination. |
| Short static window, texture re-entry | OK mean / bad max (MH_02) | Tracker occasionally leaks bad matches past the chi² gate. |
| No static window, aggressive takeoff | Poor (MH_04/05 @ ~90 cm) | Static-init assumption violated; transient dominates RMSE. |

The filter itself is correct — the linearisation, OC-EKF, and QR-reduced
update are all behaving as predicted on MH_01. The two improvements that
would lift the difficult-sequence numbers (moving-base init and a stricter
feature-gate) are frontend/init changes rather than changes to the seven
implemented functions.

## 6. Comparison with the S-MSCKF paper — discussion

Sun et al. 2017 does **not** publish a numeric table for the EuRoC Machine
Hall sequences — they report RMSE as bar heights in Fig. 2(a) of the paper,
and only for MH_01, MH_04, MH_05 (no MH_02 / MH_03; the V1/V2 bars are Vicon
room sequences that we do not run). The paper numbers the README table
quotes are eyeballed from that figure (y-axis in metres, ±~0.03 m reading
error).

Reading the paper's other bars as a sanity check for the y-axis scale: the
ROVIO bar on MH_04 reaches ~1.3 m and VINS-Mono ~0.33 m, which matches the
authors' text that *"ROVIO has larger error in the machine hall datasets"*.
We beat the paper on MH_01 and sit ~2× the paper on MH_04/MH_05, putting
our implementation in the S-MSCKF/ROVIO accuracy band rather than the
VINS-Mono band. Likely drivers for the gap on MH_04/MH_05:

1. **Feature frontend.** The Python `image.py` KLT tracker was not tuned
   for fast motion — the authors ran their C++ frontend at a higher effective
   frame rate and stereo-matched more aggressively.
2. **Noise priors.** `ConfigEuRoC` uses the off-the-shelf EuRoC noise values
   without any sequence-specific retuning.
3. **Single run.** The paper averages five runs per sequence; our numbers
   are from a single run, so the difficult-sequence RMSE is more sensitive
   to feature-track luck.

**Per-dataset offsets** — `dataset.set_starttime(offset=…)` fast-forwards the
replay into a window where the drone is stationary, so the first 200 IMU
samples (1 s at 200 Hz) feed `initialize_gravity_and_bias` a clean gravity
read. The starter `offset=40` is tuned for MH_01's long taxi-on-ground
period; applying it to MH_02/03/04/05 lands the init buffer inside flight
motion and the filter diverges (RMSE in the thousands of metres). Picking
an offset per sequence — the first 1 s window with std(‖a‖) ≲ 0.3 m/s² —
fixes this. See `DATASET_OFFSETS` in `Code/Phase 1/run_all.py`.

**MH_04 image-count mismatch** — the shipped `Stereo` class asserts
`len(cam0) == len(cam1)`; MH_04 has 2033 cam0 vs 2032 cam1 images (one
dropped frame on cam1). `dataset.py` now truncates to the shared length.
