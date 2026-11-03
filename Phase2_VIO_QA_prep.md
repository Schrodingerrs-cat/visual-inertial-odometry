# Phase 2 VIO — Q&A preparation

**Read this before the presentation.** Organized by topic so you can scan to whatever's asked. Every fact traces to a CSV row, the report, or the actual code.

---

## 1. Metrics — what we report, what they mean

### ATE — Absolute Trajectory Error

The **single number** we quote for "how well did it work". Computed as follows:

1. Predict relative pose $(R_k, t_k)$ for every consecutive frame pair $k = 0, 1, \ldots, N-1$.
2. **Dead-reckon** to absolute pose: starting from GT pose at frame 0, accumulate $T_k = T_{k-1} \cdot \text{exp}((R_k, t_k))$.
3. **Align** the predicted absolute trajectory to the GT trajectory (see "alignment" below).
4. ATE = $\sqrt{\frac{1}{N}\sum_k \lVert p_k^{\text{pred}} - p_k^{\text{gt}} \rVert^2}$ — RMSE of position errors over the trajectory, in metres.

We report two ATE variants from `evaluate_tdirect.py`:
- **`ate_aligned`** — Umeyama SE(3) alignment (rotation + translation, NO scale). This is what we cite in the table.
- **`ate_endpoint`** — distance between predicted final position and GT final position after the same alignment. Cumulative drift signal.

### `seg2s_median` and `seg2s_p95` — per-frame quality

For each 2-second sliding window, compute the predicted-vs-GT trajectory drift over that window. Report median and 95th percentile across all windows. **This is the per-frame quality signal** — cleaner than ATE because alignment can hide small per-frame errors.

### Why aligned ATE can mislead

`ate_aligned` uses SE(3) Umeyama alignment — rotation + translation but **no scale**. If the prediction is a tiny version of the right shape (predicted motion magnitudes are 5% of GT but the path is correct), SE(3) alignment can't compensate for the scale, but it can rotate/translate optimally. This makes ATE look reasonable while the prediction is fundamentally wrong.

Step K cross-family showed this: V_avg went 10.85 m → 3.61 m looking at SE(3) alignment, but Sim(3) alignment (with scale) was identical — the predictions were just slightly less collapsed in magnitude. **Lesson:** when ATE drops dramatically across a distribution shift, also check Sim(3) and per-step magnitudes before celebrating.

---

## 2. Alignment — Umeyama SE(3)

The Umeyama algorithm finds the optimal rigid transformation $(R^*, t^*)$ minimizing
$$\sum_k \lVert R^* \cdot p_k^{\text{pred}} + t^* - p_k^{\text{gt}} \rVert^2$$

Closed-form solution via SVD of the cross-covariance matrix. **No scale** in SE(3) version; **with scale** in Sim(3). We use SE(3) by default because metric scale should come from the network, not the alignment.

We invoke this through `evo_ape` (the EVO trajectory-evaluation toolkit) inside `utils/eval.py:run_evo_ape`. Output is what `ate_aligned` reports.

---

## 3. Coordinate frames — gotcha-rich

| Frame | Convention | Used in |
|---|---|---|
| **World (ENU)** | x = east, y = north, z = up. Gravity = (0, 0, −9.81). | Ground truth poses, render scene, trajectory shapes |
| **Body (FLU)** | x = forward (along velocity), y = left, z = up (along thrust). | IMU readings, attitude representation |
| **Camera** | OpenGL convention — x = right, y = up, z = back (camera looks along −z). Blender default. | Rendering only |

Stationary IMU reading in our convention: accel = (0, 0, +9.81), gyro = (0, 0, 0) — accel measures *specific force*, which is "force per unit mass that's NOT gravity"; standing still means the rotors push UP against gravity, so accel reads +g in body z.

---

## 4. The IMU pipeline (Step S corrected version)

### What's an IMU?

Inertial Measurement Unit — 6-channel sensor: 3-axis accelerometer + 3-axis gyroscope.

- **Accelerometer (acc_x, acc_y, acc_z)** — measures specific force in m/s². Includes gravity.
- **Gyroscope (gyro_x, gyro_y, gyro_z)** — measures angular velocity in rad/s.

### How we synthesize IMU from a trajectory

Given position $\mathbf{p}(t)$ along a trajectory:

1. **Velocity and acceleration**: analytic for figure-8 / circle / ellipse (closed-form derivatives); 2nd-order central differences for diamond/zigzag/triangle (Gaussian-smoothed piecewise paths).
2. **Coordinated-turn attitude**: body-x along velocity direction, body-z along thrust direction (acc_world − g_world), body-y = body-z × body-x. So as the drone orbits, its body frame yaws naturally.
3. **Specific force in body frame**: $f_{\text{body}} = R_{wb}^\top \cdot (a_{\text{world}} - g_{\text{world}})$.
4. **Body angular velocity from R-chain**: $\Omega_{\text{body}} = R_{wb}^\top \dot{R}_{wb}$, extract angular velocity from the skew-symmetric matrix. Single finite-difference stage on a continuous rotation matrix.
5. **Add Gaussian noise**: σ_acc = 0.02 m/s², σ_gyro = 0.002 rad/s.

### Sample rates and the "10 ms window"

- **Image stream**: 100 Hz (a frame every 10 ms)
- **IMU stream**: 1000 Hz (a sample every 1 ms)
- **Ratio 1:10**: each image-pair interval contains 10 IMU samples → that's the "10 ms IMU window" the I network sees per prediction.

### Why this matters

10 ms is the limiting design choice for I. Per-window translation signal at peak training speed (1.5 m/s): 1.5 m/s × 10 ms = 15 mm. Gravity-integration noise in the same window: 0.5·g·dt² ≈ 0.5 mm. SNR is 30:1 at best, often closer to 1:1 on small-motion samples. Modern learned-IMU papers (TLIO, RoNIN, AirIO) use 200–1000 ms windows for this reason.

---

## 5. Network architectures — exact specs

### V — `NetworkV_tdirect` (`models/network_v_tdirect.py`)

- **Input**: two stacked RGB frames concatenated on the channel dim → tensor of shape (B, 6, 480, 640)
- **Backbone**: VGG-8 — eight conv blocks (Conv-BN-ReLU), max-pool every 2 blocks → 4 spatial resolution stages
- **Channels**: 32, 32, 64, 64, 128, 128, 256, 256
- **Projection**: 1×1 conv to 512 channels, then `AdaptiveAvgPool2d(1)` → 512-d feature
- **Heads**: two MLPs (Linear → ReLU → Linear) — one outputs 6-D rotation, the other outputs 3-D metric translation
- **Output**: $(R \in \text{SO}(3), t \in \mathbb{R}^3)$, plus the 512-d feature
- **Total params**: ~1.5 M

### I — `NetworkI` (`models/network_i.py`, Step S rewrite)

- **Input**: IMU window — tensor of shape (B, 10, 6)
- **Encoder**: bidirectional 2-layer LSTM, hidden = 128 per direction
- **All-layers concat**: outputs from both directions, both layers → 2 × 2 × 128 = 512-d concatenated feature
- **Projection**: Linear(512 → 256) + ReLU + Dropout(0.1)
- **Heads**: two MLPs — 6-D rotation and 3-D translation
- **Output**: same shape contract as V
- **Total params**: ~600 K
- **Step S change vs Step K**: was unidirectional, single hidden state, axis-angle rotation, single MLP head. Now bidirectional, all-layers concat, 6-D continuous rotation, two heads, dropout 0.1.

### VI — `NetworkVI_tdirect` (`models/network_vi_tdirect.py`)

- **Input**: image pair + IMU window
- **Branches**: runs V and I on their respective inputs
- **Residual refiner**: small MLP that takes `[v_feat (512), i_feat (256)]` → produces `(δR ∈ SO(3) bounded ±0.1 rad, δt ∈ ℝ³ bounded ±0.2 m)`
- **Final pose**: $R = \delta R \cdot R_v$, $t = t_v + \delta t$
- **Why residual**: V is the high-confidence anchor; I provides corrections. Bounded δ prevents the refiner from over-correcting at OOD operating points (we hope — see Limitations).

### 6-D continuous rotation representation

Why not quaternions or axis-angle?
- **Quaternions**: have a sign ambiguity (q and −q represent the same rotation), and the unit-norm constraint isn't naturally enforced by gradients.
- **Axis-angle**: has a discontinuity at zero rotation — the gradient explodes near identity, exactly where most VIO predictions live.
- **6-D continuous (Zhou et al. 2019)**: predict the first two columns of the rotation matrix (6 numbers); apply Gram-Schmidt to recover the full orthogonal matrix. **Continuous everywhere**, no singularities, smooth gradients. Standard practice for deep pose regression now.

---

## 6. Losses (full version of slide 4)

### Primitives

**Geodesic loss on SO(3)** — angle of the residual rotation
$$L_{rot} = \arccos\left(\frac{\text{tr}(R_{pred}^\top R_{gt}) - 1}{2}\right)$$
Returns radians. Scale-invariant, mathematically the shortest path on the SO(3) manifold from pred to gt.

**L1 / Mean Absolute Error (MAE)** — translation
$$L_t = \lVert t_{pred} - t_{gt} \rVert_1 = \sum_{j=1}^{3} |t^{(j)}_{pred} - t^{(j)}_{gt}|$$
Robust to outliers compared to L2.

**Hover weight** — sigmoid sample reweighting
$$w_{hover}(\lVert t_{gt} \rVert) = \sigma\left(\frac{\lVert t_{gt} \rVert - 0.01}{0.005}\right)$$
$w \approx 0$ when the GT motion is below ~1 cm (near-static frame), $w \approx 1$ when above. **Purpose**: down-weight near-static frames so "predict zero" doesn't win.

### Step S losses

**V (TDirectPoseLoss)** — used by V branch
$$L_V = w_R \cdot L_{rot}(R_v, R_{gt}) + w_T \cdot \frac{\sum_b w_{hover}^{(b)} L_t(t_v^{(b)}, t_{gt}^{(b)})}{\sum_b w_{hover}^{(b)}}$$
Step S weights: $w_R = 10$, $w_T = 5$.

**I (IMUPoseLoss)** — used by I branch
$$L_I = w_R^I \cdot L_{rot}(R_i, R_{gt}) + w_T^I \cdot \mathbb{E}_b[L_t(t_i^{(b)}, t_{gt}^{(b)})]$$
Step S weights: $w_R^I = 5$, $w_T^I = 2$. **No hover weighting** — `imu_use_hover: false` in `step_s.yaml`. This is responsible for I's residual zero-bias risk.

**VI (VILoss_tdirect)** — used for fusion
$$L_{VI} = L_V + L_I + w_R \cdot L_{rot}(R, R_{gt}) + w_{trans} \cdot \mathbb{E}_b[L_t(t^{(b)}, t_{gt}^{(b)})]$$
The first two terms are auxiliary supervision on the branches; the last two supervise the residual-refined fused output. Step S: $w_R = 10$, $w_{trans} = 5$ for the fused terms.

---

## 7. Training schedule

| Network | Epochs | Optimizer | LR | Batch size | Notes |
|---|---|---|---|---|---|
| V | 30 | AdamW | 1e-4 | 8 | Cosine warmup, 2 warmup epochs |
| I | 20 | AdamW | 1e-4 | 8 | Same |
| VI | 30 (10 stage1 + 20 stage2) | AdamW | 1e-4 | 8 | Stage 2 LR multiplier on V/I = 0.0 |

**`stage2_lr_mult: 0.0`** — in stage 2 of VI training, V and I are unfrozen but their LRs are scaled by 0, so they effectively stay frozen. Only the residual refiner moves. This was lesson L1 from earlier steps where joint fine-tuning destabilized convergence.

**Sampling**: weighted by frame-pair translation magnitude, 5 bins, alpha=0.5 — over-samples large-motion pairs to keep gradients meaningful (counters the same near-static frame imbalance the hover weight addresses on the loss side).

**Gradient clipping**: max-norm 1.0.

---

## 8. Evaluation pipeline — how the trajectory is stitched

`scripts/evaluate_tdirect.py` per trajectory:

1. Load checkpoint, set model to eval.
2. Iterate frame-pairs in order: predict $(R_k, t_k)$ for each.
3. **Dead-reckon**: starting from GT pose at frame 0, $T_k = T_{k-1} \cdot \text{exp}((R_k, t_k))$. Predicted absolute pose at every frame.
4. Write `gt_<tid>.tum` (GT poses) and `<method>_<tid>.tum` (predicted poses) in TUM format.
5. Run `evo_ape` for SE(3) Umeyama alignment + ATE computation.
6. Compute `seg2s_median`, `seg2s_p95` over 2-second sliding windows.
7. Compute `ate_endpoint` — predicted vs GT final position after alignment.
8. Append rows to `outputs/<step>/results.csv`: `(method, traj_id, metric_name, value)`.
9. Render plots: 3D, top-down XY, per-axis (XYZ, RPY).

---

## 9. The six approaches — what each tested

| # | Step | Hypothesis | What changed | Outcome |
|---|---|---|---|---|
| 1 | F | Baseline VIO works on synthetic | Mixed-texture train/test, 100 s sinusoid trajectories, decomposed (R, t/d, n) head | Failed at 11.81 m mean — texture confound |
| 2 | G | Fixing the texture removes the appearance confound | Same data, locked to single texture (`tex_03`) | V dropped to 5.29 m — confound was 55 % of failure |
| 3 | H | The decomposed head is itself a bug | Switched V's head from (R, t/d, n) to direct (R, t) prediction (t-direct) | Another 27 % off V — 3.86 m |
| 4 | J | Trajectory complexity is the bottleneck | 30 s simple shapes (figure-8/circle/ellipse), 4 train trajs, slow speed | V = 0.089 m within-family — 43× drop |
| 5 | K | Speed diversity unlocks generalization | 4 shape families × 2 speeds = 8 train trajs, varied speeds | V = 0.184 m, **VI = 0.173 m** (first VI < V) |
| 6 | S | I-branch fix + faster speeds | Corrected IMU pipeline, bidirectional I, speeds 0.5–1.5 m/s | V regressed to 1.16, I monotonically descending |

(Step I — anchored fusion with soft gates — is intentionally omitted from this throughline; the gates collapsed because the photometric residual was uninformative when V was still poor.)

---

## 10. Anticipated questions

### "What does ATE mean specifically?"

Absolute Trajectory Error — RMSE of the predicted absolute position trajectory vs ground truth, after Umeyama SE(3) alignment, in metres. Measures total accumulated drift after dead-reckoning. ATE-aligned of 0.18 m means: average position error along the trajectory is 18 cm after optimal rotation+translation alignment.

### "Why did V get worse from Approach 5 to 6?"

Speed range went from 0.3-0.5 m/s (Approach 5) to 0.5-1.5 m/s (Approach 6). The VGG-8 receptive field can match features within about 3-5 pixels per layer; at 1.2 m/s, inter-frame pixel motion is around 8-10 pixels, exceeding the kernel reach. The CNN has no explicit displacement search. V's architecture is the same; the test distribution moved out of its competence band.

### "Why does I plateau around 1 metre?"

The 10 ms IMU window puts the per-window translation signal (around 5 mm at peak training speed) within an order of magnitude of gravity-integration noise (0.5 mm). Signal-to-noise ratio is 6:1 best case, 1:1 on small-motion samples. No matter how clever the encoder, the input information is bounded. TLIO/AirIO use 200-1000 ms windows for the same task.

### "Why does VI sometimes do worse than V?"

The residual refiner can only nudge V's prediction (bounded by ±0.1 rad rotation, ±0.2 m translation per design). When V is wrong at OOD operating points, the refiner doesn't fall back to I — it adds noise on top of V's wrong prediction. Step S 411: V=0.76 m, VI=2.29 m — refiner made things worse by 200%. A confidence-gated soft fusion would help here.

### "Why use SE(3) alignment instead of Sim(3)?"

We want metric trajectory accuracy. SE(3) doesn't allow the alignment to rescale the prediction — if the network learns trajectories at 5% of true magnitude but the right shape, SE(3) ATE will still be large. Sim(3) would hide that as a scale issue. Using SE(3) keeps the alignment honest about magnitude. The trade-off: SE(3) ATE is harder to interpret across scale-collapsing failures (Step K cross-family).

### "What's the difference between aligned ATE and per-frame quality?"

ATE is a single trajectory-level number after SE(3) alignment — it says "how close are these two trajectories overall after optimally aligning them." `seg2s_median` is a per-frame quality signal — for each 2-second window, how much does the prediction drift from GT. The two can disagree: a trajectory with low aligned ATE can still have bad per-frame jitter, and vice versa.

### "Why are you training on synthetic data?"

Real-world VIO data with ground-truth pose is rare and expensive (motion capture, RTK GNSS, Vicon). Synthetic Blender renders give us perfect ground truth at zero acquisition cost, plus controlled experiments — change one variable at a time (texture, speed, shape) without confounds. Trade-off: synthetic doesn't transfer trivially to real video. We acknowledge this as a limitation, not a deployment claim.

### "Why not classical SLAM?"

We did — Phase 1 is a Stereo MSCKF implementation on EuRoC data. Phase 2 is the learning-based half: can a CNN+LSTM learn the equivalent representation end-to-end? Phase 1 is the reference; Phase 2 is the comparison.

### "What's the role of IMU here when the motion is so simple?"

IMU should provide:
- **Metric scale** (vision is fundamentally scale-ambiguous from monocular images; IMU has metric units built in)
- **Rotation observability** (camera rotation can be confused with translation in monocular vision; gyroscope measures it directly)
- **High-frequency motion** (image stream is 100 Hz; IMU is 1000 Hz, so it captures motion between frames)

In our experiments I underperforms because the 10 ms window is too short for the IMU to integrate meaningfully — but VI training still benefits from the IMU-provided scale signal at moderate speeds.

### "What's a 'residual refiner'?"

A small MLP that takes both branch features and predicts a small correction to V's prediction: $\delta R$ bounded to ±0.1 rad, $\delta t$ bounded to ±0.2 m via tanh. The fused prediction is $\delta R \cdot R_v$ and $t_v + \delta t$. Idea: V is a strong prior, IMU provides correction. The bounding is critical — without it, the refiner could swing arbitrarily.

### "Why 6-D continuous rotation instead of quaternions?"

Quaternions have a sign ambiguity ($q$ and $-q$ are the same rotation) and a unit-norm constraint that gradient descent doesn't enforce naturally. Axis-angle has a discontinuity at zero rotation — gradients explode near identity, exactly where most VIO predictions live. The 6-D continuous representation (Zhou et al. 2019) predicts the first two columns of $R$ as a 6-vector and applies Gram-Schmidt to recover an orthonormal matrix. Continuous everywhere, no singularities, smooth gradients. Industry standard for deep pose regression now.

### "What's the inference latency?"

Single-pair forward through all three networks on an RTX Pro 6000 — order of milliseconds. Eval is mostly data-loading-bound, not compute-bound.

### "What's training time?"

Per-trajectory render: ~5 minutes on Blender EEVEE. Full Step S render of 13 trajectories: ~65 min. Train V: ~1 hour for 30 epochs. Train I: ~30 min for 20 epochs. Train VI: ~2 hours for 30 epochs (two stages). Eval: a few minutes per trajectory.

### "What if the IMU rate were higher / window were longer?"

The fix we'd recommend. Going from 10 ms to 100-200 ms windows (TLIO-style) should improve I's SNR by 10-20×. The architectural change is: extend the dataloader's IMU slice; route the longer context only to standalone I (so VI's I-branch stays compatible with existing checkpoints), or accept that VI also retrains with the longer context. Documented in `report.md` §6.5.

### "What's a hover weight?"

A sigmoid-gated sample weight applied to the L1 translation loss. It's small (≈0) when the ground-truth translation magnitude is below ~1 cm and large (≈1) when above. Purpose: in mixed batches with both small-motion and large-motion samples, "predict zero" pays low loss on small-motion samples and dominates the batch mean. Hover weight discounts those samples so the network learns motion in the regime that matters.

### "Why is I's loss not hover-weighted?"

Step K's loss config didn't enable it. It exists in the codebase (`IMUPoseLoss_hover` in `models/losses.py`) and is the diagnosed Phase 1 fix for the I zero-bias collapse. Step S kept `imu_use_hover: false` to preserve the architectural-vs-loss attribution: we wanted to know if the bidirectional-I architecture alone moves the needle, before stacking hover-loss on top.

### "How is gravity handled in the IMU input?"

We DO NOT pre-subtract gravity from the accelerometer. The raw 6-channel input to I is `[ax, ay, az, gx, gy, gz]` in body frame, with gravity preserved. Reasoning: Liu et al. (AirIO 2023) showed that pre-subtracting gravity discards orientation information the network can use. The network learns to handle gravity as part of the signal.

### "What's the difference between accel and specific force?"

Specific force is what an accelerometer actually measures: the non-gravitational acceleration in the body frame. So a stationary IMU reads (0, 0, +g) — the rotors push UP against gravity, sensor reads UP (+g). It's not "the actual acceleration" — it's "the force per unit mass needed to deviate from free fall."

### "Why 8 training trajectories — isn't that small?"

By deep-learning standards, yes — vision foundation models train on millions of images. Our justification: each trajectory contains 30 seconds × 100 Hz = 3000 frame pairs, so the actual training set is ~24 000 pairs across 8 trajectories. Still small, but the network is also small (1.5 M params) and the task is bounded (relative pose between consecutive frames is a constrained regression problem). The bigger constraint isn't volume — it's **diversity** (single texture, single altitude, narrow speed range).

### "What's the test set methodology?"

For each step:
- **Train trajectories** drive training optimization
- **Validation trajectory** drives `best.pt` selection (lowest val_ATE_on_minitest)
- **Minitest trajectory** provides a second validation signal during training
- **Test trajectories** are held-out, never seen during training; final ATE is reported on these

For Step S:
- Train: 400-407 (8 trajs, 4 shape families × 2 speeds)
- Val: 408 (held-out ellipse @ 1.0 m/s)
- Test: 410-413 (held-out ellipse at 0.3 / 0.5 / 0.8 / 1.2 m/s)

### "What is OOD?"

Out-of-distribution. Test sample sits outside training distribution along some axis (speed, texture, altitude, shape). Network has no learned response, typically fails non-monotonically. Step S traj 410 (ellipse @ 0.3 m/s) is OOD because 0.3 m/s is slower than every training trajectory; all three networks fail there.

### "What are your top 3 limitations?"

1. **CNN receptive field caps motion magnitude V can match** — V degrades at higher speeds because feature correspondence breaks beyond the kernel reach.
2. **10 ms IMU window is the noise floor for I** — per-window SNR is bounded; bidirectional encoder helps but can't extract information that isn't in the input.
3. **Residual refiner has no fail-safe** — fusion degrades V at OOD operating points instead of falling back to I.

### "What would you do next?"

Two options that address actual failures:
1. **Replace V's CNN with a cost-volume / RAFT-style encoder** — explicit displacement search handles larger motions. Address the speed-induced V degradation.
2. **Replace residual refiner with confidence-gated fusion** — use photometric warp residual as a per-sample V-confidence signal; gate `α = sigmoid(MLP)` to weight V vs I per sample. When V is wrong, fusion falls back to I. Address the VI > V regression at OOD operating points.

Or: extend the IMU window to 100-200 ms and retrain I — addresses the architectural ceiling, but is a bigger lift.

---

## 11. Things to NOT say

- "Step K cross-family was a real improvement" — it wasn't, it was an SE(3) alignment artifact.
- "VI always beats V" — only at Approach 5, and only on 4 of 5 test trajs.
- "Step S unlocks the IMU branch" — partially. I improved at one operating point (0.8 m/s, 0.43 m) but is *worse on average* than Step K.
- "We use real-world data" — we don't. All synthetic Blender.
- "Network is end-to-end trained" — V and I are trained separately first; VI training freezes them and only updates the refiner.

---

## 12. One-line summary you can repeat

> **"Synthetic-data deep-learning VIO. Three networks — vision CNN, IMU LSTM, fused refiner. Six diagnostic steps; the headline is that fusion finally beat vision-only at Approach 5 once we added speed diversity. Approach 6 fixed the IMU branch architecture and pipeline, and is the first run where all three branches are simultaneously learning."**
