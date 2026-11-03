# Phase 2 VIO — Master reference

**Read this. If you remember everything in here, you cannot be tripped up.**

Companion docs (already saved):
- `Phase2_VIO_speech.md` — 3-minute spoken script, slide-cued
- `Phase2_VIO_concepts.md` — intuitive deep-dive on losses, IMU window, fusion
- `Phase2_VIO_QA_prep.md` — Q&A reference

This document is the consolidated everything-in-one.

---

## 0 · The 30-second pitch

> "Project 4 — deep visual-inertial odometry on synthetic Blender drone footage. Three networks: a vision-only CNN, an IMU-only LSTM, and a fused VI with a residual refiner. Six diagnostic experiments. The headline is that fusion finally beat vision-only at Approach 5 (VI=0.173 m vs V=0.184 m within-family). Approach 6 fixed the IMU pipeline and architecture and is the first run where all three branches are simultaneously learning."

---

## 1 · The system in one breath

| | What | Inputs | Outputs |
|---|---|---|---|
| **V** | VGG-8 CNN, 512-d feature | Two stacked RGB frames, (B, 6, 480, 640) | R (3×3), t (3) — metric |
| **I** | Bidirectional 2-layer LSTM, 256-d feature | IMU window, (B, 10, 6) — accel + gyro, body frame | R (3×3), t (3) — metric |
| **VI** | Residual refiner over (V_feat, I_feat) | Image pair + IMU window | R = δR · R_v, t = t_v + δt; δR/δt bounded ±0.1 rad / ±0.2 m |

**Loss for everything**: geodesic SO(3) for rotation + L1/MAE for translation, with sigmoid hover-weighting on V. VI loss adds auxiliary supervision on each branch + supervision on the fused output. All trained with AdamW, lr=1e-4, batch 8, cosine warmup.

---

## 2 · The story (six approaches)

| # | Step | Setup | V | I | VI | What changed |
|---|---|---|---:|---:|---:|---|
| **1** | F | Different texture per traj · 100 s sinusoid · decomposed head | **11.81** | 3.46 | 6.94 | baseline (broken) |
| **2** | G | **Fixed** texture (`tex_03`) · same trajs · decomposed head | **5.29** | 1.78 | 2.35 | texture confound removed → V −55% |
| **3** | H | Fixed tex · **t-direct** prediction (no t/d decomposition) | **3.86** | 1.78 | 3.36 | parameterization fixed → V −27% |
| **4** | J | **Simple shapes** 30 s · single speed · 4 train trajs | **0.089** | 1.51 | 0.131 | training-data complexity → V drops 43× |
| **5** | K | Simple shapes · **varied** (slow) speed · 4 families × 2 speeds | **0.184** | 1.24 | **0.173** | speed diversity → **VI < V for the first time** |
| **6** | S | Step K + **corrected IMU pipeline** + **bidirectional NetworkI** + higher speeds (0.5–1.5 m/s) | 1.16 | 1.36 | 1.63 | I-fix verified per-traj; mean dragged up by OOD-slow |

**Step I (anchored fusion with soft gates) is intentionally omitted** — the gates collapsed because V was still poor at that point. Skip if asked.

### Step 6 (Step S) per-trajectory breakdown — the part the mean hides

| traj | speed | V | I | VI | notes |
|---|---|---:|---:|---:|---|
| 410 | **0.3 m/s** OOD slow | 3.00 | 2.20 | 3.31 | all three fail — slower than every train traj |
| 411 | 0.5 m/s | 0.76 | 1.60 | 2.29 | in-range; refiner over-corrects |
| 412 | 0.8 m/s | 0.27 | **0.43** | **0.16** | **only cell where VI < V; best I in the project** |
| 413 | 1.2 m/s | 0.62 | 1.21 | 0.75 | train-ceiling speed |

---

## 3 · The losses (formulas + plain-English meaning)

### Primitives

**Geodesic SO(3) (rotation error)**
$$L_{rot} = \arccos\!\left(\frac{\mathrm{tr}(R_{pred}^\top R_{gt}) - 1}{2}\right)$$
The angle (in radians) of the residual rotation. The shortest rotation needed to map prediction to ground truth. Coordinate-free, no gimbal lock.

**L1 / MAE (translation error)**
$$L_t = \lVert t_{pred} - t_{gt} \rVert_1 = |t_x| + |t_y| + |t_z|$$
Sum of per-axis absolute errors. Robust to outliers vs L2/MSE; constant-magnitude gradients keep all samples relevant.

**Hover weight (sigmoid sample reweighting)**
$$w_{hover}(\lVert t_{gt} \rVert) = \sigma\!\left(\frac{\lVert t_{gt}\rVert - 0.01}{0.005}\right)$$
≈ 0 when GT motion is below ~1 cm; ≈ 1 above. **Stops the network from "winning" by predicting zero on near-static frames.**

### Per-network losses

**V — TDirectPoseLoss**
$$L_V = 10 \cdot L_{rot}(R_v, R_{gt}) + 5 \cdot \overline{w_{hover} \cdot L_t(t_v, t_{gt})}$$

**I — IMUPoseLoss** (no hover — `imu_use_hover: false`)
$$L_I = 5 \cdot L_{rot}(R_i, R_{gt}) + 2 \cdot \overline{L_t(t_i, t_{gt})}$$

**VI — VILoss_tdirect** (auxiliary + fused supervision)
$$L_{VI} = L_V + L_I + 10 \cdot L_{rot}(R, R_{gt}) + 5 \cdot \overline{L_t(t, t_{gt})}$$
where $(R, t) = (\delta R \cdot R_v, t_v + \delta t)$ are the residual-refiner outputs.

---

## 4 · The data

- **Source**: synthetic Blender renders, EEVEE, 30 s trajectories at 100 Hz image / 1000 Hz IMU
- **Scene**: single textured ground plane (`tex_03.webp`), constant 4 m altitude
- **IMU model**: Gaussian noise σ_acc=0.02 m/s², σ_gyro=0.002 rad/s, gravity preserved (NOT pre-subtracted)
- **Step S generation pipeline (corrected)**:
  1. Analytic vel/acc from samplers (figure-8/circle/ellipse) or 2nd-order central diff (diamond/zigzag)
  2. **Coordinated-turn attitude**: body-x along velocity, body-z along thrust direction → yaw tracks orbital motion
  3. **Specific force in body** = R^T · (a_world − g_world)
  4. **Body angular velocity** = R^T · Ṙ (single finite-diff stage on a continuous rotation matrix)
  5. Add Gaussian noise

**Why this matters**: the previous (pre-Step S) pipeline did three stacked finite-differences (pos → vel → acc → roll/pitch → ω), producing 2.34 rad/s gyro spikes at trajectory boundaries on cosine-start families. The corrected pipeline reduced those spikes from 2.35 → 0.42 rad/s and gave the gyro real orbital signal instead of noise.

### Train / val / test splits (Step S)

- **Train**: 400-407 (8 trajs, 4 shape families × 2 speeds: circle 0.5/1.2, fig8 0.5/1.0, diamond 0.8/1.05, zigzag 0.8/1.0)
- **Val**: 408 (held-out ellipse @ 1.0 m/s — drives best.pt selection)
- **Test**: 410-413 (held-out ellipse at 0.3 / 0.5 / 0.8 / 1.2 m/s — speed sweep)

Ellipse is intentionally held out as the test family — never seen at training.

---

## 5 · The metrics

### ATE — Absolute Trajectory Error (the headline number)

1. Predict (R, t) for every consecutive frame pair
2. **Dead-reckon** to absolute pose: T_k = T_{k−1} · exp((R_k, t_k))
3. **Umeyama SE(3) align** the predicted absolute trajectory to GT (rotation + translation, **no scale**)
4. **ATE = RMSE** of position errors over the trajectory, in metres

We report `ate_aligned` (this) and `ate_endpoint` (final-position drift after alignment).

### `seg2s_median` / `seg2s_p95`

For each 2-second sliding window: predicted-vs-GT trajectory drift over that window. Median and 95th percentile across all windows. **The cleanest per-frame quality signal** — alignment can't hide local errors.

### SE(3) vs Sim(3) — the silent-failure mode

SE(3) alignment doesn't allow rescaling — if the prediction is "shape correct but magnitude collapsed to 5%", SE(3) ATE will still be large. Sim(3) would compensate. We use SE(3) by default to keep alignment honest about magnitude. Step K cross-family showed the cost: an apparent 3× ATE drop turned out to be entirely an SE(3) alignment artifact (Sim(3) numbers identical). **Always sanity-check with Sim(3) when ATE drops dramatically across a distribution shift.**

---

## 6 · Architecture details (memorize these)

### V — `NetworkV_tdirect`

- VGG-8: 8 conv blocks, channels (32, 32, 64, 64, 128, 128, 256, 256), max-pool every 2 blocks, BatchNorm + ReLU
- Projection: 1×1 conv → 512 channels, AdaptiveAvgPool2d(1) → 512-d feature
- Two MLP heads: rotation (Linear→ReLU→Linear→6) using **6-D continuous representation** (Zhou et al. 2019), translation (Linear→ReLU→Linear→3) metric
- ~1.5 M parameters total

### I — `NetworkI` (Step S)

- Bidirectional 2-layer LSTM, hidden 128 per direction
- All-layers concat: 2 layers × 2 directions × 128 = 512-d
- Projection: Linear(512→256) + ReLU + Dropout(0.1)
- Same head structure as V (6-D rotation + 3-D translation)
- ~600 K parameters

**Why bidirectional + multi-layer concat?** The previous Step K NetworkI was unidirectional (top-layer hidden state only) and used axis-angle rotation. We saw `models/imu_model.py` had an alternative architecture that "trained well" — bidirectional, all-layers concat, 6-D rotation, separate heads, dropout 0.1. That architecture was lifted into NetworkI for Step S. Public API unchanged — `NetworkI(hidden_dim=128)` still works.

### VI — `NetworkVI_tdirect`

- Wraps V and I; runs both forward
- Residual refiner: small MLP on `concat(v_feat 512, i_feat 256)` → outputs δR (3-D axis-angle) and δt (3-D)
- Bounding via tanh: |δR| ≤ 0.1 rad, |δt| ≤ 0.2 m
- Final pose: R = δR · R_v, t = t_v + δt

**Why bounded?** Without bounding, the refiner could output arbitrary corrections, effectively predicting the entire pose itself and making V irrelevant. Bounding forces it to be a *correction* mechanism, not an *override* mechanism. **Trade-off**: when V is more than 0.1 rad / 0.2 m wrong, VI is also wrong by more than that — no fail-safe. This is what causes Step S VI > V at OOD operating points.

### Two-stage VI training

- **Stage 1** (epochs 0-9): V and I frozen; refiner trains alone.
- **Stage 2** (epochs 10-29): V and I unfrozen, but `stage2_lr_mult: 0.0` → their effective LR is zero. So they're "unfrozen but not actually updating". This is the conservative choice from lesson L1; joint fine-tuning destabilized convergence in earlier steps.

---

## 7 · The 10 ms IMU window — what it actually means

- Image stream is 100 Hz → 10 ms between frames
- IMU stream is 1000 Hz → 10 IMU samples per image-pair interval
- That's the "10 ms window" the I network sees per prediction

**Per-window translation signal** (at 1.5 m/s peak training speed): 1.5 × 0.01 = 15 mm
**Gravity-integration noise** in the same window: 0.5 · g · dt² ≈ 0.5 mm
**SNR**: 30:1 best case, ~1:1 on small-motion samples

**Why it's the I branch's hard ceiling**: even with a perfect bidirectional encoder, you can't extract information that isn't in the input. Modern learned-IMU papers (TLIO, RoNIN, AirIO) use 200–1000 ms windows for this exact reason — 20-100× more integration baseline.

**Why we don't extend it**: would require a dataloader change to reach back in time before frame A; would invalidate VI checkpoints (which currently consume the legacy 10-sample slice). Documented as Phase 2 fix in `report.md` §6.5; deferred to after Step S.

**Why I works at all on traj 412 (0.43 m, best in project)**: at 0.8 m/s, motion magnitude is well-matched to training distribution; bidirectional LSTM extracts the maximum information available; rotation is much easier than translation (gyro directly measures angular velocity). The improvement is operating-point-specific, not universal — at OOD-slow (410) I still collapses.

---

## 8 · Top 5 limitations

| # | Limitation | Direct evidence |
|---|---|---|
| **1** | **CNN receptive-field cap on inter-frame motion** — VGG8 has no explicit displacement search; fails when pixel motion exceeds the kernel reach (~3-5 px) | V went 0.184 → 1.16 m moving from Step K (0.3-0.5 m/s) to Step S (0.5-1.5 m/s). At 1.2 m/s, pixel motion is 8-10 px. |
| **2** | **10 ms IMU window puts I at the gravity-integration noise floor** | I_avg never below ~1 m across F → S; per-window SNR is bounded by physics, not architecture |
| **3** | **Residual refiner has no fail-safe** — bounded δ means VI degrades when V is more than 0.1 rad / 0.2 m wrong | Step S 411: V=0.76, VI=2.29 — refiner makes V worse by 200% |
| **4** | **Single-frame-pair prediction, no temporal recurrence** — errors integrate linearly with trajectory length | Step K cross-family: 100 s sinusoids → 4-200× scale collapse, hidden by SE(3) alignment |
| **5** | **Synthetic-only: 1 texture, 1 altitude, ideal IMU** — zero variance on three deployment-relevant axes | Step F: changing the texture alone broke V (11.81 m on mixed textures) |

---

## 9 · Q&A — most likely questions

### "What is ATE?"
Absolute Trajectory Error. Dead-reckon per-frame predictions into a global trajectory, optimally align rotation+translation (no scale, Umeyama SE(3)) to GT, then RMSE of position errors. Units: metres. ATE-aligned of 0.18 m means: average position error along the trajectory is 18 cm after optimal alignment.

### "What is OOD?"
Out-of-distribution. A test sample sits outside what training saw on some axis (speed, texture, altitude, shape). Step S traj 410 (ellipse @ 0.3 m/s) is OOD because 0.3 m/s is slower than every train trajectory. Networks have no learned response for OOD regions, so they fail unpredictably.

### "Why did V get worse from Approach 5 to 6?"
Speed range went from 0.3-0.5 m/s to 0.5-1.5 m/s. VGG-8's receptive field can match features within ~3-5 pixels per layer; at 1.2 m/s, inter-frame pixel motion is 8-10 px, exceeding the kernel reach. The CNN has no explicit displacement search. V's architecture didn't change; the test distribution moved out of its competence band.

### "Why does I plateau around 1 m?"
The 10 ms IMU window puts the per-window translation signal (~5 mm at peak training speed) within an order of magnitude of gravity-integration noise (~0.5 mm). SNR is at the floor. Bidirectional LSTM in Step S helped at one operating point (0.8 m/s, I=0.43 m) but the architectural ceiling is the input window length, not the encoder capacity.

### "Why does VI sometimes do worse than V?"
The residual refiner can only correct V by ±0.1 rad / ±0.2 m — bounded by design to prevent override. When V is wrong by more than that (OOD speeds), the refiner can't fix it; the I-derived correction adds noise on top of V's already-wrong prediction. Step S VI > V on 3 of 4 test trajs because of this. Confidence-gated fusion (with a fall-back to I when V is unreliable) would help.

### "Why use SE(3) alignment instead of Sim(3)?"
SE(3) doesn't allow rescaling — if predictions are scale-collapsed (5% of GT magnitude but right shape), SE(3) ATE will still be large. Sim(3) would compensate and hide the issue. We use SE(3) to keep alignment honest about magnitude. Trade-off: ATE numbers are harder to interpret across distribution shifts where scale collapses (Step K cross-family: SE(3) showed 3× improvement, Sim(3) showed identity → improvement was an alignment artifact).

### "Why use 6-D continuous rotation instead of quaternions?"
Quaternions have a sign ambiguity (q and -q are same rotation) and the unit-norm constraint isn't naturally enforced by gradients. Axis-angle has a discontinuity at zero rotation — gradients explode near identity, exactly where most VIO predictions live. The 6-D continuous representation (Zhou et al. 2019) predicts the first two columns of R; Gram-Schmidt recovers the orthonormal matrix. Continuous everywhere, no singularities.

### "What is a hover weight?"
A sigmoid-gated sample weight on the L1 translation loss. Centred at 1 cm: w ≈ 0 when ground-truth motion is below ~1 cm, w ≈ 1 above. **Purpose**: in mixed batches with both small-motion and large-motion frames, "predict zero" pays low loss on small-motion frames and dominates the batch mean. Without hover weight, the network learns to predict zero everywhere — the I branch's failure mode. With hover weight, near-static frames are removed from the loss budget so the optimizer can't game it.

### "Why is I's loss not hover-weighted in Step S?"
`imu_use_hover: false` in `step_s.yaml`. This was deliberate: we wanted to attribute Step S's I-branch improvement to the architecture upgrade alone, before stacking the loss-side fix on top. `IMUPoseLoss_hover` exists in the codebase (the diagnosed Phase-1 fix) and is the obvious next experiment.

### "How is gravity handled in the IMU input?"
We do **not** pre-subtract gravity. The raw 6-channel input is [ax, ay, az, gx, gy, gz] in body frame, with gravity preserved. Reasoning: AirIO (Liu et al. 2023) showed pre-subtracting gravity discards orientation information the network can use. The network learns to handle gravity as part of the signal.

### "What's the difference between accelerometer and 'specific force'?"
Specific force is what an accelerometer actually measures: non-gravitational acceleration in the body frame. A stationary IMU reads (0, 0, +g) — the rotors push UP against gravity, sensor reads UP at +g. It's not "the actual acceleration" — it's "the force per unit mass needed to deviate from free fall."

### "Why 8 training trajectories — isn't that small?"
Each trajectory is 30 s × 100 Hz = 3000 frame-pairs, so the actual training set is ~24 000 pairs. The bigger constraint isn't volume — it's **diversity**: single texture, single altitude, narrow speed range. Adding more trajectories of the same kind wouldn't help; widening the distribution would.

### "Why not classical SLAM?"
We did. Phase 1 implements Stereo MSCKF on EuRoC data. Phase 2 is the learning-based half: can a CNN+LSTM learn the equivalent end-to-end? Phase 1 is the reference baseline; Phase 2 is the comparison.

### "Why train on synthetic at all?"
Real-world VIO data with GT pose is rare and expensive (motion capture, RTK GNSS, Vicon). Synthetic Blender renders give us perfect GT at zero acquisition cost, plus controlled experiments — change one variable at a time without confounds. Trade-off: synthetic doesn't transfer trivially to real video. Acknowledged as a limitation, not a deployment claim.

### "What's next?"
Two approaches, in priority order:
1. **Curriculum learning + confidence-gated fusion**: warm-start V from Step K's checkpoint, gradually expand training to Step S speeds. Replace residual refiner with photometric-confidence-gated soft fusion. Both pieces are existing patterns in the codebase. ~3-5 days of work.
2. **RAFT/cost-volume vision encoder**: explicit displacement search handles larger motions. Addresses the speed-induced V degradation at the architectural level. Bigger lift (~2 weeks) but addresses the root cause.

The IMU-window extension (10 ms → 100-200 ms) is the obvious next architectural fix on the I side — would lift the SNR ceiling.

---

## 10 · Things to NOT say

- ❌ "Step K cross-family was a real improvement" — it was an SE(3) alignment artifact. Sim(3) numbers identical.
- ❌ "VI always beats V" — only at Approach 5, only on 4 of 5 test trajs. Step S VI > V on 3 of 4 trajs.
- ❌ "Step S unlocks the IMU branch" — partially. Best per-traj I value in the project (0.43 m at traj 412), but I_avg = 1.36 m, *worse* than Step K's 1.24 m on average.
- ❌ "We use real-world data" — we don't. All synthetic Blender.
- ❌ "Network is end-to-end trained" — V and I are pre-trained separately first; VI training freezes them and only updates the refiner.
- ❌ "RAFT distillation worked" — it didn't, in Step K. Forward-backward consistency rejected 99.6% of frame-pairs because flow magnitudes were too small. (At Step S speeds, RAFT might actually engage now — but we haven't retested.)

---

## 11 · The one-line summary you can fall back on

> **"Synthetic-data deep VIO. Three networks — vision CNN, IMU LSTM, fused refiner. Six diagnostic experiments. Headline: fusion finally beat vision-only at Approach 5 once we added speed diversity. Approach 6 fixed the IMU branch architecture and is the first run where all three branches are simultaneously learning."**

---

## 12 · Slide-by-slide cheat (your deck `Phase2_VIO_Deck.pdf`)

| # | Slide | One-line beat |
|---|---|---|
| 1 | Title | "Project 4 — deep VIO on synthetic Blender drone footage" |
| 2 | Three networks | "V CNN, I bidir LSTM, VI residual refiner — shared evaluation" |
| 3 | Architecture diagram | "Two inputs, two branches, fusion module produces δR·R_v + δt; total loss = L_V + L_I + L_fused" |
| 4 | Loss function | "Geodesic SO(3) for rotation, L1 for translation, sigmoid hover weight on V" |
| 5 | Dataset | "Single texture, four shape families, 8 train trajs, 30 s × 100 Hz × 1000 Hz IMU" |
| 6 | Train loss | "V converges fast; I plateaus at noise floor; VI shows two-stage transition at epoch 10" |
| 7 | Results overview | "V drops 3 orders of magnitude; I flat ~1-2 m; VI tracks V then beats it at Approach 5" |
| 8 | Cumulative ATE | "−55%, −27%, ×43, **VI<V at Approach 5**, regression at Approach 6 due to harder test set + speed shift" |
| 9 | Train trajectory (figure-8) | "V/VI overlay GT (5/9 cm); I drifts at 1.14 m — zero-bias collapse mode" |
| 10 | Approach 5 (ellipse) | "**VI=0.07 < V=0.08 < I=1.06** — the punchline of the project" |

---

## 13 · Backup definitions (in case you blank)

- **VIO**: Visual-Inertial Odometry. Estimating drone/robot motion from camera + IMU.
- **ATE**: Absolute Trajectory Error. Trajectory-level RMSE after optimal alignment.
- **Dead reckoning**: integrating per-frame predictions to get a global trajectory.
- **OOD**: Out-of-distribution. Test sample outside training distribution on some axis.
- **SE(3)**: rigid 3D transformations (rotation + translation, no scale). 6 DoF.
- **Sim(3)**: rigid 3D + isotropic scale. 7 DoF.
- **SO(3)**: 3D rotations only. 3 DoF.
- **IMU**: Inertial Measurement Unit. 6-channel sensor (3-axis accel + 3-axis gyro).
- **Specific force**: what an accelerometer measures. Non-gravitational acceleration in body frame. Stationary reading: (0, 0, +g).
- **Geodesic distance**: shortest path on a curved manifold. For SO(3) it's the rotation angle.
- **Umeyama**: closed-form SVD-based algorithm for optimal SE(3) or Sim(3) alignment between two point sets.
- **Hover weight**: sigmoid sample weight that down-weights near-static frames in the L1 translation loss.
- **t-direct**: predicting metric translation directly, instead of decomposing as (translation/depth) × altitude.
- **Residual fusion**: VI architecture pattern where V predicts the pose, refiner predicts a small correction, final = V + correction.
- **6-D continuous rotation**: predicting the first two columns of a rotation matrix, applying Gram-Schmidt. No singularities.
- **AdamW**: optimizer (Adam with decoupled weight decay).
- **Cosine warmup**: LR schedule that ramps up linearly for a few epochs, then decays as a cosine curve to zero.

---

## 14 · The 60-second walk-in summary

If someone catches you in the hallway right before the talk and says "what's the project?":

> "Deep visual-inertial odometry on synthetic drone trajectories. Three branches — vision CNN, inertial LSTM, fused VI with a residual refiner. We ran six diagnostic experiments. The first three fixed confounds in the data and the parameterization. The fourth showed within-family generalization works. The fifth was the unlock — adding speed diversity to training, and for the first time fusion beat vision-only. The sixth fixed the IMU pipeline and architecture; results are mixed because we also pushed speed up, which broke V in the high range, but I is finally learning. Top limitations: VGG8's receptive field caps motion magnitude V can match; the 10 ms IMU window caps how much signal I can extract; the bounded residual refiner has no fail-safe when V is wrong. Open work: cost-volume V encoder for higher speeds, longer IMU windows, confidence-gated fusion."

---

**You have everything you need. Walk in confident.**
