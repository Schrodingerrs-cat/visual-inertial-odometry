# Phase 2 VIO — How it actually works, conceptually

This document is the "I want to *understand* it, not just recite numbers" version. Three parts, each written so you could explain it to someone who hasn't read the code:

1. **How the loss functions are actually used** — what each term tells the network
2. **The 10 ms IMU window — what that means and why it matters**
3. **The fusion network — what it's really doing and where it breaks**

---

# Part 1 — How the losses are actually used

## The big picture: a loss is the network's only feedback

The network never sees ground-truth poses directly. It sees the loss value — a single scalar that says "your predictions were this wrong on this batch." Backpropagation then computes gradients of that scalar with respect to every weight, and the optimizer nudges weights to reduce it. **The shape of the loss function determines what the network learns to optimize.**

So when we design a loss, we're really answering: "what behavior do I want to make easy for the optimizer, and what behavior do I want to make expensive?"

## Why two terms (rotation + translation), not one

We could have written one giant loss: "the Frobenius norm of the difference between the predicted 6-DoF pose and ground-truth 6-DoF pose." But rotation and translation **live in different units and different geometries**:

- Translation lives in $\mathbb{R}^3$ — a flat Euclidean space, units of metres, additive.
- Rotation lives on SO(3) — a curved manifold, no natural units, multiplicative.

If you mash them into one number you have to pick an arbitrary scaling between metres and radians, and the gradient direction will be dominated by whichever has bigger numerical magnitude. Worse, naive matrix-difference rotation losses don't actually measure how *wrong* a rotation is — they measure how different two matrices look in coordinates.

So we use **two separate terms with explicit weights**, each measuring its own quantity in its own units:
- `w_rot * L_rot` — angular error in radians, weight 10
- `w_t * L_t` — translation error in metres, weight 5

The weights (10 and 5) are how we tell the optimizer "rotation matters more per unit of error than translation does." A 0.1 rad error (~5.7°) contributes the same to total loss as a 0.2 m translation error.

## What L_rot (geodesic SO(3)) actually does

Imagine you're rotating a Rubik's cube and want to undo the rotation. The geodesic angle is **how far you'd have to rotate it back to undo it**, regardless of which axis. Mathematically: take $R_{pred}^\top R_{gt}$ — that's the rotation you'd apply to your prediction to get to ground truth. Its trace tells you the rotation angle via the formula $\arccos((tr - 1)/2)$.

Why this and not, say, "MSE between Euler angles"?
- Euler angles have **gimbal lock** (singularities) and depend on the rotation order convention.
- A small Euler-angle change can correspond to a huge actual rotation near the singularity, and vice versa.
- Geodesic distance is **coordinate-free** — it measures the actual rotation, not its parameterization.

What the network sees: when the prediction is close to ground truth, the gradient of L_rot pushes it closer along the shortest path on the SO(3) manifold. When far, the gradient is bounded (arccos's slope flattens near 1, by design).

## What L_t (L1 / MAE) actually does

Per-axis absolute error, summed across x, y, z. Why L1 instead of L2 (MSE)?

- **L2 is dominated by outliers**: a sample with 1 m translation error contributes 100× more loss than a sample with 0.1 m error. The gradient on the bad sample dominates, and the network learns to fix the worst sample at the expense of the average.
- **L1 has constant gradient magnitude** (sign of the error): every sample pulls equally hard toward zero error. The network learns to fix the *most common* error, not the worst one.

In our setting most frame-pairs have small translation magnitudes (millimetres at hover, centimetres at slow speeds). L2 would basically ignore them — they're "small" in a squared-error sense. L1 keeps them in the loss budget.

## What the hover weight is really doing (and why it's a conceptual hack)

Here's the failure mode it prevents:

- Most consecutive frames in a slow-motion training set have very small ground-truth translation (~1 mm).
- A network that predicts "zero translation always" would have a per-frame L1 loss of ~0.001 — tiny.
- The handful of frames with real motion (~50 mm) would contribute much more to the loss, but on a batch of 8, the average is dominated by the small-motion frames if there are more of them.
- Optimizer concludes: "the cheapest way to reduce loss is to predict zero on small-motion frames and accept some error on big-motion frames."
- **Result: the network learns to predict near-zero for everything**. This is the I-branch failure mode we documented.

The hover weight fixes this by **discounting near-static frames in the loss**. The sigmoid is centred at 1 cm: frames with $\lVert t_{gt} \rVert < 1$ cm get weight ~0; frames with $\lVert t_{gt} \rVert > 1$ cm get weight ~1.

So now the network can no longer "win" by predicting zero on small-motion frames — those frames contribute nothing to the loss either way. The optimizer is forced to learn the actual motion, because that's the only thing the loss measures anymore.

**Conceptually**: the hover weight tells the optimizer "this sample is uninformative; don't bias the gradient toward it." It's a form of importance sampling at the loss level instead of the dataloader level.

We use it on V (and would use it on I if we'd flipped the config). We don't use it on the fused VI term because the fused output is supervised against the same ground truth, but we trust V's regularization to carry through the refiner.

## How the VI loss orchestrates fusion training

The VI loss has four terms:

```
L_VI = L_V_branch  +  L_I_branch  +  w_R * L_rot(fused)  +  w_trans * L_t(fused)
       └──────────────auxiliary──────────────┘   └────────fused supervision────────┘
```

Each term plays a specific role:

- **L_V_branch** — supervises V's intermediate predictions $(R_v, t_v)$. Without it, V's heads would have no gradient during VI training, and could drift to whatever produces a good fused output (which might not be a sensible standalone V prediction).

- **L_I_branch** — same role for I.

- **w_R · L_rot(fused)** — supervises the residual-refined fused rotation. This is the gradient that actually trains the refiner's δR output.

- **w_trans · L_t(fused)** — same for the refiner's δt.

The auxiliary terms are called "deep supervision" in the literature — they keep intermediate sub-networks anchored to their individual targets, even when a downstream module could, in principle, recover from any drift. Without them, the refiner could "compensate" for V drifting in a weird direction, and you'd lose your standalone V baseline.

In practice, in Step S, V and I are frozen during VI training (`stage2_lr_mult: 0.0`), so the auxiliary terms only update V's and I's *head MLPs*, not the backbones. The backbones inherit their weights from the standalone V and I training runs that came before VI training.

## The training step in slow motion

Here's what happens for a single batch during V training:

1. **Sample a batch** of 8 frame-pair indices, weighted to over-represent large-motion pairs (counters the same imbalance the hover weight handles on the loss side).
2. **Forward**: feed images through VGG8 → feature → heads → predict $(R_v, t_v)$ for all 8 samples.
3. **Compute loss**:
   - Per-sample geodesic rotation angle → mean → multiply by 10.
   - Per-sample L1 translation error, multiplied by per-sample hover weight, normalised by sum of weights → multiply by 5.
   - Sum the two → scalar.
4. **Backward**: PyTorch autograd computes $\partial L / \partial \theta$ for every parameter.
5. **Optimizer step**: AdamW updates each weight by `−lr · adaptive_estimate(gradient)`.
6. Repeat for the next batch.

Multiply by ~3000 batches/epoch × 30 epochs and you have one training run.

## What happens if you remove a term

| Remove | Symptom |
|---|---|
| L_rot from V | V learns translation, but rotation drifts to identity (or some constant). Trajectory follows the right path but with wrong heading. |
| L_t from V | V learns rotation but predicts zero translation. Trajectory shape is right, but stuck at origin. |
| Hover weighting on V | "Predict zero" wins. V outputs near-zero translation everywhere, ATE blows up. (We've seen this in I — its loss has no hover.) |
| L_V_branch from VI loss | V's heads drift during VI training because they're no longer pinned to their individual target. |
| Fused supervision (last two terms) from VI loss | Refiner has no gradient — it could output anything, including identity (no correction). |

---

# Part 2 — The 10 ms IMU window, explained intuitively

## What 10 ms feels like

Ten milliseconds is **less than the time for a single frame of a 100 Hz video**. It's about a tenth of a typical eye blink. If your phone's IMU is sampling at 100 Hz (most are), it's collecting *one sample* in 10 ms. We're sampling at 1000 Hz, so we get *ten samples* in that window — but the total time elapsed is still 10 ms.

In 10 ms at 1 m/s, the drone has moved **1 centimetre**. The IMU is being asked to detect that 1 cm of motion from 10 acceleration readings.

## What signal is actually in the window

Think about what the accelerometer measures during 10 ms of slow flight:

- **Gravity** — about 9.81 m/s² in the body z-axis, present in every sample.
- **Real linear acceleration** — at 0.5 m/s peak speed, accelerating from rest takes maybe 0.5 s, so within a 10 ms window the acceleration changes by maybe 10 mm/s² of change. That's 0.1% of the gravity signal.
- **Sensor noise** — Gaussian, σ = 0.02 m/s² per axis. So in 10 samples you expect noise of around 0.02 / √10 ≈ 0.006 m/s² after averaging.

The signal you actually want to extract (translation over the 10 ms window) is on the order of millimetres. The dominant signal is gravity (constant). The next dominant is noise. The thing the network is trying to detect is way down at the noise floor.

## The integration arithmetic

To get translation from acceleration over 10 ms, the network has to **double-integrate** acceleration:

- Position = ∫∫ acceleration dt dt
- Over 10 ms with ~5 mm of true displacement: that's 5 mm of useful signal
- Over 10 ms of gravity-noise integration: σ_acc · dt² / 2 ≈ 0.02 · (0.01)² / 2 = 1 micrometre per sample
- Over 10 samples: noise scales as √10, so about 3 micrometres

OK so noise alone is small. But here's the catch: **errors in the gravity estimate dominate the noise**. If the network's belief about the gravity direction is off by 1°, the residual specific-force error is 9.81 · sin(1°) ≈ 17 cm/s². Integrated over 10 ms that's 8 mm — 60% of the actual signal. Pose-aware integration is hard at short windows.

This is why classical inertial navigation systems use **multi-second windows with bias-aware estimators (Kalman filters)**. They're not trying to extract translation from a 10 ms window — they accumulate over hundreds of milliseconds and use the longer baseline to separate signal from noise.

## Why we have a 10 ms window in the first place

Two reasons:
1. **The image rate is 100 Hz** — one frame every 10 ms. Each frame-pair prediction has 10 ms of IMU samples between the two frames. The 1:10 ratio is dictated by the dataset.
2. **The model architecture chose to predict per-frame-pair**. We could have widened the window to 100 IMU samples (100 ms) by reaching backward in time before frame A — but that's an architectural choice we didn't make.

So the 10 ms window is not a fundamental limit — it's a design choice. The fix is documented in `report.md` §6.5: extend the IMU input to 100–200 ms by reaching back in time, route the longer context to the I branch only.

## Why our network can do anything at all with 10 ms

Even though the signal is tiny, the network learns useful things from the IMU window:

- **Rotation estimation** is much easier than translation. Gyroscope directly measures angular velocity in rad/s. Over 10 ms at moderate angular rates (~0.1 rad/s), you accumulate ~1 mrad of rotation. That's small, but the gyro noise is also small (σ = 0.002 rad/s, integrated to ~6 µrad over 10 samples). Rotation SNR is way better than translation SNR.
- **Acceleration patterns** carry signature information. A circular trajectory has a steady centripetal acceleration; a turn has a transient. The network can learn to recognize these patterns even if it can't precisely integrate them to translation.
- **The "scale" signal**: even if you can't estimate translation magnitude precisely, you can estimate *whether* the drone is moving fast or slow, which calibrates the V branch's predictions.

## The intuitive analogy

Imagine you're handed 10 still photos of a car, taken at 1 ms intervals. The car has moved maybe 1 mm between adjacent photos. Your job: tell me where the car is going to be in 10 minutes.

That's the I branch's task. The information you have is technically there — the motion blur, the relative pixel offsets — but it's so close to the resolution limit that you're guessing more than measuring. To do well, you'd want a second-long video, not a 10 ms snapshot. **That's the architectural ceiling.**

## What changes if we extend to 200 ms

Same arithmetic, scaled up:
- Translation signal: 1 m/s × 200 ms = **20 cm of motion** — easy to detect
- Noise integration: still small (scales as √window)
- Rotation: 200 ms × 0.1 rad/s = **20 mrad** — well above any noise floor

The trade-off is **latency** — every prediction now uses 200 ms of IMU history, so the I branch can't react to changes faster than that. For drone applications at modest speeds, that's fine. For aggressive maneuvers, it's not.

---

# Part 3 — The fusion network, deeper

## Three classical ways to fuse vision and IMU

In the deep VIO literature, there are roughly three philosophies:

### Early fusion — concat and learn
Concatenate the V and I features into one big vector, feed into a single MLP that outputs the final pose. Simple, lets the network learn whatever combination it wants. Risk: the network can't learn what it can't supervise — if V is dominant in training, fusion may just ignore I.

### Late fusion — combine predictions
Each branch produces its own pose; combine them with a weighted average or a switching rule. Conceptually clean (each branch is an interpretable estimator). Limited by how cleverly you combine the predictions.

### Residual fusion — V is the anchor, I provides corrections
What we use. V produces a pose; a refiner predicts a small δ-correction; the final pose is V's prediction adjusted by δ. Hybrid: lets the network use both feature-level information AND prediction-level structure.

## Why we picked residual fusion

V is the strong branch (works at 0.18 m on within-family). I is the weak branch (1.2 m). If you used early fusion, the network might just learn "ignore I, copy V" — and you'd be back to a vision-only system with extra parameters. If you used pure late fusion (e.g. weighted average of poses), you'd be moving V's prediction in I's direction, which makes things worse.

Residual fusion lets you say: "use V's prediction as the starting point; only nudge it where I has useful information." The δR and δt are bounded (±0.1 rad, ±0.2 m) so the refiner can't override V — only adjust it.

## What the refiner actually computes

```
v_feat = V_branch(image_pair)        # 512-d
i_feat = I_branch(imu_window)        # 256-d
combined = concat(v_feat, i_feat)    # 768-d

h = ReLU(Linear(combined, 64))       # hidden layer
output = Linear(h, 6)                # δaa (3-D axis-angle), δt (3-D)

δR = rodrigues(tanh(output[:3]) * 0.1)  # bounded ±0.1 rad
δt = tanh(output[3:]) * 0.2             # bounded ±0.2 m

R_final = δR @ R_v
t_final = t_v + δt
```

Two MLP layers, total ~50K parameters. Tiny compared to V and I.

## What the refiner is actually learning

It's learning a function from `(v_feat, i_feat) → (δR, δt)` such that applying that correction to V's prediction makes it closer to ground truth.

In practice, what it learns depends on what V and I give it:
- If V is reliable and I gives complementary scale info: refiner learns to amplify motion magnitude when I says "yes there was real motion" — improves V.
- If V is wrong in a systematic way (e.g. always under-predicts at some operating point): refiner learns the bias correction — improves V.
- **If V is wrong in an unpredictable way and I doesn't have useful info**: refiner has no signal, learns to add noise on top of V's wrong predictions — **makes V worse**. This is the Step S failure mode at OOD speeds.

## Why bound δR and δt?

If we let the refiner output arbitrary corrections, it could in theory predict the entire pose itself: $\delta R = R_v^{-1} R_{true}$, $\delta t = t_{true} - t_v$. Then V becomes irrelevant — the refiner is doing all the work.

Bounding δR to ±0.1 rad and δt to ±0.2 m says: "you may correct V by at most 6° rotation and 20 cm translation per frame-pair. If V is more than that wrong, you can't fix it." This forces the refiner to be a *correction* mechanism, not an *override* mechanism.

The bound also means: if V is wrong by more than the bound, **VI is also wrong by more than the bound** — there's no fail-safe. This is the design weakness Step S exposed.

## Why VI > V on 3 of 4 Step S trajectories

At Step S OOD-slow (traj 410, ellipse @ 0.3 m/s, slower than every train traj):
- V is wrong because it never saw 0.3 m/s.
- I is wrong because the IMU window has weak signal at low speeds (lower acceleration → noisier).
- The refiner sees two confused features and outputs noise as a correction.
- The noise is bounded to ±0.1 rad / ±0.2 m, but it's still noise — adding it to V's already-wrong prediction makes the result worse on average.

This isn't unique to OOD-slow — even at 411 (ellipse @ 0.5 m/s, in-distribution), VI=2.29 vs V=0.76. The refiner is amplifying an unhelpful pattern that happens to correlate with V's failure mode at that speed.

## What a better fusion would look like

Two known-better designs the literature suggests:

### Confidence-gated fusion (the Step I attempt, with a twist)
- Compute a per-sample V-confidence signal — e.g. the photometric warp residual: warp `img_a` to `img_b` under V's predicted pose, measure pixel-level mismatch. Low residual = V is right.
- Compute a per-sample I-confidence signal — e.g. gyro-integration consistency.
- Per-sample fusion weight $\alpha$ from a small MLP over the confidences.
- $t_{fused} = \alpha \cdot t_v + (1-\alpha) \cdot t_i$, $R_{fused} = \text{SLERP}(R_i, R_v, \alpha)$.

The key difference from residual fusion: when V is wrong (low confidence), the gate falls back to I. When I is wrong, it falls back to V. There's a fail-safe in both directions.

We tried this in Step I but the gates collapsed — V was so bad at the time that the photometric residual was uninformative everywhere. Now V is good enough at trained speeds that the photometric signal would actually discriminate good vs bad V predictions.

### Kalman-filter-style fusion (more classical)
- Maintain a state estimate $\hat{x}$ with covariance $P$.
- V provides a measurement $z_v$ with covariance $R_v$ (low covariance = V is confident).
- I provides a prediction step (use IMU integration to propagate $\hat{x}$ forward).
- Apply Kalman update: $\hat{x}_{new} = \hat{x} + K(z_v - H\hat{x})$ where $K$ depends on covariances.
- **Naturally weighted by uncertainty**: when V is uncertain, K is small and the prediction relies on the IMU; when V is certain, K is large and the prediction trusts V.

This is what classical VIO (S-MSCKF, our Phase 1 work) does. The challenge in deep VIO is making the network produce calibrated covariances, not just point estimates.

## The two-stage VI training, conceptually

Stage 1 (epochs 0-9): V and I are loaded from their separately-trained checkpoints and **frozen**. Only the refiner trains.

Why? Because if you trained everything jointly from scratch, the refiner would have no useful gradient — V and I are random at initialisation, so their features are noise, so the refiner's input is noise. By pre-training V and I to convergence first, the refiner sees meaningful features from epoch 0 of stage 1.

Stage 2 (epochs 10-29): V and I are unfrozen, but the LR multiplier on V/I is set to 0 (`stage2_lr_mult: 0.0`). So they're "unfrozen" but their weights don't actually update — the LR is 0.

Why the elaborate dance instead of just keeping them frozen? Three reasons:
1. **Optimizer state**: AdamW maintains per-parameter momentum and variance buffers. Unfreezing them clears those buffers; setting LR=0 doesn't. For exact reproducibility of the schedule, it matters.
2. **Future flexibility**: if we wanted to fine-tune V/I in stage 2, we'd just set `stage2_lr_mult > 0`. The architecture supports it; we just chose not to enable it after Step J showed joint fine-tuning destabilized convergence.
3. **Gradient flow**: even with LR=0, the auxiliary V and I losses still produce gradients (which become zero updates). This validates that the gradient computation is consistent across stages.

In practice: stage 2 is the same as stage 1 with extra epochs. The refiner gets more training time after V and I are validated as stable. Step S's lesson L1 was that this gives more reliable convergence than truly fine-tuning V and I jointly.

## Why fusion is hard, conceptually

Fusion fundamentally has to answer: "given two estimators with different error characteristics, how do I produce a third estimator that's better than either alone?"

Statistical answer: **the optimal fusion is inverse-variance weighted averaging**. Each estimator should contribute proportional to its precision (inverse of variance). When one estimator is uncertain, lean on the other.

But neural networks don't naturally produce uncertainties — they produce point estimates. So you need either:
- A second head that outputs variance (Bayesian deep learning approaches)
- A surrogate uncertainty signal (photometric residual, gyro consistency)
- A learned gate (anchored fusion)
- Or a structural assumption like "V is always our prior and I is always corrective" (residual fusion — what we use)

Each has trade-offs. The residual approach is simplest but has no fail-safe. The Bayesian approach is principled but adds complexity. The gated approach was tried in Step I and failed because V wasn't good enough yet.

The "right" answer in modern deep VIO research (TartanVO, DROID-SLAM) is closer to: produce uncertainty estimates per-sample, then fuse via inverse-variance weighting. This is the next architectural step, but it requires retraining V and I to also output variances, not just means.

---

# Part 4 — Putting it all together: one frame-pair prediction, end to end

To make all of this concrete, here's what happens for a single inference step in Step S:

1. **Inputs**: two consecutive RGB frames (480×640 each) at time $t_a$ and $t_b = t_a + 10$ ms; 10 IMU samples in between.

2. **V forward pass**:
   - Concatenate frames → tensor (1, 6, 480, 640)
   - VGG8 backbone: 4 conv stages with max-pooling → (1, 256, 30, 40)
   - Project + adaptive pool → (1, 512) v_feat
   - Rotation head: Linear(512, 128) → ReLU → Linear(128, 6) → 6-D continuous → R_v (3×3)
   - Translation head: Linear(512, 128) → ReLU → Linear(128, 3) → t_v (3,) — metres

3. **I forward pass**:
   - IMU window → tensor (1, 10, 6)
   - Bidirectional 2-layer LSTM, hidden 128 per direction → (1, 10, 256) per timestep
   - All-layers concat → (1, 512), project to (1, 256) i_feat
   - Same head structure → R_i (3×3), t_i (3,)

4. **VI fusion**:
   - Concat (v_feat, i_feat) → (1, 768)
   - Refiner MLP → (δR_axis_angle, δt) bounded
   - R_final = δR @ R_v
   - t_final = t_v + δt

5. **Dead reckoning**: chain (R_final, t_final) onto the running absolute pose estimate to get the global pose at time $t_b$.

6. **Repeat** for the next frame pair.

For a 30 second trajectory at 100 Hz, that's 3000 inference steps. Each one is independent — there's no temporal smoothing, no Kalman filter, no global optimization. Pure dead reckoning, errors accumulate.

That's both the strength (simple, fast, end-to-end differentiable) and the weakness (no global trajectory consistency, so cross-trajectory generalization fails) of the current pipeline.

---

## Three sentences to remember from this whole document

1. **The hover weight is what stops the network from "winning" by predicting zero on small-motion frames** — without it, the optimizer's cheapest path to low loss is to not move.

2. **The 10 ms IMU window has fundamentally limited per-window signal** — the bidirectional LSTM in Step S squeezes out more information than the previous unidirectional version, but you can't extract information that isn't in the input.

3. **Residual fusion uses V as a strong prior and I as a small corrector — bounded ±0.1 rad / ±0.2 m** — which works when V is mostly right but fails at OOD operating points where V is wrong by more than the correction can fix.
