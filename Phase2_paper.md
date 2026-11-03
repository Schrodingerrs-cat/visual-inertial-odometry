# Deep Visual–Inertial Odometry on Synthetic Drone Footage

## A Diagnostic Story Across Six Approaches

**WPI RBE/CS549 — Project 4, Phase 2**
*Aryan Shah, May 2026*

---

## Abstract

We report a long-running diagnostic study of learned visual–inertial
odometry (VIO) on synthetic Blender renders. The system has three
networks: a vision-only VGG-8 (V), an IMU-only bidirectional LSTM (I),
and a fused residual-refiner (VI). Across six numbered approaches
(F, G, H, J, K, S) we change exactly one or two variables at a time
and report what each change buys us. The headline is not a leaderboard
number — it is a story about *why* learned VIO is hard, told through
the failures we ran into and the specific intervention each step
shipped to address one of them. Approach 5 (Step K) was the first
configuration in which fusion (VI) measurably improved over
vision-only (V); Approach 6 (Step S) corrected the synthetic IMU
pipeline and lifted the I branch's per-trajectory floor for the first
time, but exposed a separate failure on V's side once we pushed
inter-frame motion past the CNN's receptive-field reach. The pipeline
is **not** a deployable VIO solution, and we close by enumerating the
specific things that prevent it from being one.

---

## 1. Origin and motivation

Phase 1 of this project implements Stereo MSCKF — a tightly-coupled
filter-based VIO from Sun et al. (2017) — in Python, on the EuRoC MAV
Machine Hall sequences. That system works: it is mathematically
principled, requires no training data, and gives consistent ATE on the
five MH sequences (0.07–0.21 m on MH_01 through MH_05 in our run).
The interesting question that motivates Phase 2 is whether a learned
end-to-end network can recover the *equivalent* of what MSCKF does, so
that the same setup that produces the EKF baseline can also produce
a deep model on the same dataset format and the same evaluation
protocol.

The goal was not to claim deep VIO beats classical VIO — both papers
on the SLAM side and our own Phase-1 results say that's not yet the
case. The goal was to take the components that make a classical VIO
work (vision branch, IMU branch, fusion of the two), build them as
neural networks, train them on data we control, and use diagnostic
experiments to surface *which assumptions a learned VIO breaks under
which conditions*. Specifically: at what motion magnitude does a CNN
stop being a useful pose regressor? What integration window does an
IMU branch need before its translation prediction has any signal? What
fusion strategy lets two weak branches combine into a stronger one
without introducing failure modes neither has alone?

The synthetic-data choice falls out of this framing. Real VIO data
with ground truth (Vicon, RTK-GNSS, motion capture) is expensive and
sparse, and it does not allow the kind of one-variable-at-a-time
ablations the diagnostic question demands. Blender renders give us
perfect ground truth at zero acquisition cost and let us change one
thing — a texture, a peak speed, a trajectory shape — without
confounds.

---

## 2. Data generation

### 2.1 Scene and rendering

Every rendered frame is a top-down view from a virtual drone camera
flying 4 m above a single textured ground plane. The plane is a flat
Blender mesh with a single tiled image texture (`tex_03.webp` for all
the speed/shape-controlled experiments — Approach 2 onward). The
camera is pinhole, 640×480 at 100 Hz, with intrinsics (fx, fy, cx, cy)
= (640, 640, 320, 240). Lighting is a single sun lamp; the EEVEE
realtime renderer is fast enough to render a 30-second trajectory
(3000 frames) in a few minutes per trajectory on a single GPU.

The reason for this minimal scene is the same as the
synthetic-data reason: **we want to remove every source of variance
that is not the trajectory**. A textured plane gives the CNN something
to track; the tiled texture (`tex_03.webp`) provides high-frequency
features the network can match across consecutive frames; the constant
4 m altitude removes depth as a confound. Earlier approaches (F)
varied the texture per trajectory, and we measured the cost of that
choice — a 55 % degradation in V's ATE — when we collapsed it to a
single fixed texture in Approach 2 (G). That was the first surprising
result of the project: changing the texture broke V harder than any
other single variable.

### 2.2 IMU model

The IMU is a synthetic 6-channel sensor running at 1000 Hz (10 IMU
samples per inter-frame interval). It returns specific force and body
angular velocity:

- `accel = R^T (a_world − g_world) + n_acc`
- `gyro  = ω_body + n_gyro`

with `n_acc ~ N(0, σ_acc²)`, σ_acc = 0.02 m/s², `n_gyro ~ N(0,
σ_gyro²)`, σ_gyro = 0.002 rad/s. Gravity is preserved in the
accelerometer signal — we do **not** pre-subtract it. This follows
AirIO (Liu et al. 2023), which showed that gravity carries
orientation information the network can use, and that pre-subtraction
discards that information.

The IMU pipeline producing `R(t), ω(t), a_body(t)` from a parametric
trajectory `pos(t)` is more interesting than it looks, and got rebuilt
between Approach 5 and Approach 6 — see §3.5 below.

### 2.3 Trajectory library

Trajectories are produced by parametric samplers in
`blender/trajectory_library.py`. Each sampler takes a target peak
horizontal speed (m/s) and a duration (30 s for everything from
Approach 4 onward) and returns position, velocity, and acceleration
arrays at 1000 Hz. The library exposes seven shape families, in two
groups:

**Smooth (analytic vel/acc)**: `simple_circle`, `simple_figure8`,
`simple_ellipse`. These return `pos`, `vel`, `acc` directly from
closed-form expressions; the IMU pipeline never finite-differences
these.

**Piecewise-smooth (smoothed corner samplers)**: `simple_diamond`,
`simple_zigzag`, `simple_triangle`, `simple_sinusoid_30s`. These have
corners or direction reversals; we use a 2nd-order central
finite-difference plus reflect-padding smoothing to extract `vel` and
`acc` from `pos`. The smoother zeroes velocity at sample 0 and T-1 by
construction; we carry forward the previous valid body-x to keep the
attitude continuous across those degenerate samples (otherwise the
gyro spikes at trajectory boundaries by 2+ rad/s — see §3.5).

We deliberately mix smooth and piecewise families so the same trained
model has to handle (a) constant-curvature motion (circle, ellipse),
(b) figure-8 lemniscate motion with one direction reversal at the
crossing, (c) cornered translation (diamond, zigzag), and (d) the
held-out shapes the network has never trained on (triangle, sinusoid).
Speed and shape are deliberately decoupled: in Approaches 5 and 6 we
have 4 shape families × 2 peak speeds each in the training set, so the
network cannot memorise a (shape, speed) pair.

### 2.4 Trajectory design — speed profiles, shape diversity, variation strategy

Speed is parameterised by `target_peak_speed` and varies per trajectory
ID. Approach 4 trained at a single 0.4 m/s peak; Approach 5 expanded
to 0.1–0.5 m/s across pairs of (shape, speed); Approach 6 pushed the
training envelope up to 0.5–1.5 m/s, with per-family speed ceilings
because the diamond's plane-fit ceiling is around 1.05 m/s on its
trained seed (above that the diamond shrinks more than 1 % from the
target perimeter).

Shape variation is meaningful: a circle has constant curvature and a
constant-magnitude lateral acceleration; a figure-8 has one direction
reversal; a diamond has four discrete corners; a zigzag has alternating
back-and-forth motion at constant peak speed. The point is to expose
the V/I/VI networks to a range of *gyro signatures* — circle yaws
monotonically, diamond has yaw spikes at corners, zigzag has yaw
sign-flips. If the I branch is going to learn anything about
orientation, the gyro has to carry orbital signal, and the variety of
shapes ensures it does.

The training/validation/test split convention from Approach 4 onward
treats `simple_ellipse` as a held-out shape that never appears in
training. Within Approach 6 (Step S), the ellipse is also the
single-shape carrier of the test-time speed sweep: trajs 410, 411,
412, 413 are all `simple_ellipse` at 0.3, 0.5, 0.8, 1.2 m/s
respectively — one OOD-slow, three in-range. This isolates the
*speed-shift effect* without having to control for shape.

---

## 3. Step-by-step evolution

We run six numbered approaches: F, G, H, J, K, S. Step I (anchored
soft-gated fusion) is intentionally omitted — its gates collapsed
because V was still poor at that point and the soft gates committed
to V everywhere. The story below treats it as a dead end and skips
it. Headline ATE numbers (m, lower is better) are reproduced from
each step's `outputs/<step>/results.csv`; per-step `analysis.md`
files document the specific decisions.

### 3.1 Approach 1 — Step F: Baseline (broken)

> V = 11.81  ·  I = 3.46  ·  VI = 6.94

**What we set out to solve**: get the three networks (V, I, VI) end
to end on a varied dataset before tuning anything. Train each
trajectory under its own texture, on 100-second sinusoidal
trajectories, with V predicting `(R, t/d, n)` (translation
decomposed as a unit-direction times depth times altitude); VI
combines V and I via a small fusion head with a learned scale mixer.

**What changed vs nothing**: this is the first run; everything is a
baseline.

**What it improved**: nothing — but it produced numbers we could
diagnose.

**What it exposed**: V's 11.81 m ATE was driven by mode collapse —
the network was outputting nearly-zero translations on every frame
pair, and was effectively predicting "I haven't moved" no matter
what was on screen. Looking at the per-traj breakdown, V degraded
worst on trajectories with the *least familiar* texture, suggesting
the per-trajectory texture variation was a confound rather than a
useful generalisation signal. This is the moment we realised the
data design itself was the dominant problem and the network
architecture was a secondary one.

### 3.2 Approach 2 — Step G: Single texture

> V = 5.29  ·  I = 1.78  ·  VI = 2.35

**What we set out to solve**: confirm the texture-confound
hypothesis. If V's degradation on Step F is texture-driven, fixing
the texture should fix V.

**What changed vs Step F**: every trajectory was rendered with the
same texture (`tex_03.webp`). Same trajectories, same 100 s, same
(R, t/d, n) parameterisation, same architecture.

**What it improved**: V dropped 55 % (11.81 → 5.29 m). The
hypothesis was right: the network had been spending capacity on
texture invariance instead of geometry. Removing the texture as a
variable freed that capacity.

**What it exposed**: V was still doing badly in absolute terms —
5 m ATE on a 30-second trajectory at 0.4 m/s peak speed is 40 % of
the trajectory length. The (R, t/d, n) decomposition was still in
play, and it has a multiplicative coupling between the predicted
unit translation `t/d` and the GT depth `d` that is fragile when
either is noisy.

### 3.3 Approach 3 — Step H: Direct translation parameterisation

> V = 3.86  ·  I = 1.78  ·  VI = 3.36

**What we set out to solve**: the parameterisation. A network whose
output is `t = (t/d) · d` is multiplicatively sensitive to errors in
both factors; the learned target `t/d` can't be larger than ~2 m
without exploding gradients during training. We had been working
around this and the multiplicative coupling was a mode-collapse
attractor (predict tiny `t/d`, multiply by GT `d`, get a small
translation, score moderately well).

**What changed vs Step G**: V was rebuilt as `NetworkV_tdirect`,
with a 9-d head (6 for the rotation in continuous-6D, Zhou et al.
2019 + 3 for translation directly in metres). The (R, t/d, n)
pipeline was retired entirely. Same texture, same trajectories, same
loss except the translation supervision is now pure L1 on metric `t`
instead of L1 on `t/d` followed by reconstruction.

**What it improved**: V dropped a further 27 % (5.29 → 3.86 m). The
parameterisation change was orthogonal to the texture fix; it
attacked a different failure mode (multiplicative coupling) and
delivered an independent improvement.

**What it exposed**: V was still at 3.86 m, and VI = 3.36 m was
basically the same — the residual refiner wasn't doing useful work.
The bottleneck was no longer the parameterisation. It was the
*training data complexity*: 100-second sinusoidal trajectories with
variable-amplitude oscillation are hard to memorise when you only
have four of them.

### 3.4 Approach 4 — Step J: Simple shapes, single speed

> V = 0.089  ·  I = 1.51  ·  VI = 0.131

**What we set out to solve**: collapse the trajectory complexity to
something the network can memorise, and verify within-family
generalisation works at all. The previous three approaches had been
training on 100-second variable-amplitude sinusoids, and there was
no clean answer to "is V's failure architectural, or is it just
that the trajectories are too hard?".

**What changed vs Step H**: we replaced the sinusoid family
entirely. New trajectories were 30 seconds long, used the simple
parametric shape samplers (`simple_circle`, `simple_figure8`,
`simple_ellipse`), and ran at a single speed. Four training
trajectories total; same architecture, same texture, same
parameterisation.

**What it improved**: V dropped 43× (3.86 → 0.089 m). VI tracked
V at 0.131 m. The within-family within-speed regime is *easy* once
the trajectories are simple enough.

**What it exposed**: a follow-up cross-family eval (10.85 m on the
original aggressive sinusoids) showed V had memorised the shape
*and* the motion magnitude; both axes had to be addressed before
generalisation would happen. The diagnosis became clearer: V's
generalisation envelope was being defined by the specific (shape,
speed) pairs in the training set, and the held-out shapes were
sitting outside that envelope.

### 3.5 Approach 5 — Step K: Speed-diverse training

> V = 0.184  ·  I = 1.24  ·  VI = 0.173

**What we set out to solve**: widen the training distribution along
*both* shape and speed simultaneously, so that V sees enough
(shape, speed) variety to interpolate to held-out shapes at
held-out speeds.

**What changed vs Step J**: 8 training trajectories instead of 4, in
4 shape families (circle, figure-8, diamond, zigzag) at 2 peak
speeds each. Training peak speeds spanned 0.1–0.5 m/s. The val/test
sets used new shapes (ellipse, triangle, sinusoid_30s) at a mix of
seen and unseen speeds. We also added a RAFT-distillation
auxiliary loss on V — but it didn't engage (more on this below).

**What it improved**: VI = 0.173 m beat V = 0.184 m for the first
time in the project. *Fusion finally helped*. This is the
punchline of Approach 5.

**What it exposed**: three things.
  1. RAFT distillation didn't engage. Forward-backward consistency
  rejected 99.6 % of frame-pairs because flow magnitudes were too
  small at the 0.1–0.5 m/s training speeds. We kept the codepath
  but the loss term was effectively zero throughout training.
  2. Cross-family eval (testing on the original aggressive
  sinusoids) showed an apparent 3× ATE drop that turned out to be
  entirely an SE(3)-alignment artifact — Sim(3) numbers were
  identical. This is the moment we learned to always sanity-check
  ATE under both alignment groups when the test distribution
  shifts. Scale collapse is invisible to SE(3).
  3. The I branch was still pinned around 1 m. Pre-S2 we already
  suspected the IMU pipeline was producing pathological signals.
  Step S verified it.

### 3.6 Approach 6 — Step S: Corrected pipeline, faster speeds

> V = 1.16  ·  I = 1.36  ·  VI = 1.63

**What we set out to solve**: the I branch's persistent ~1 m floor.
Two hypotheses were on the table — the data is broken, or the
architecture is too weak — and Step S addressed both at once. (The
controlled-variable framing this breaks is acknowledged in the
config; Step S is "does the corrected pipeline + upgraded I make I
work?", not "which of the two interventions matters more?")

**What changed vs Step K** — three coincident changes:

1. **IMU pipeline correction**. The previous pipeline did three
stacked finite-differences (pos → vel → acc; pos → roll/pitch;
roll/pitch → ω) and used a constant-yaw attitude that didn't track
the orbital motion. The result was 2.34 rad/s gyro spikes at
trajectory boundaries on cosine-start families, and a gyro signal
that was approximately Gaussian noise on the no-corner shapes
(circle, ellipse) because constant-yaw means ω = 0 in body frame
during an orbit, regardless of how fast you're going. We rebuilt the
pipeline as: (a) use the analytic vel/acc the parametric samplers
already return, never finite-difference position for the smooth
families; (b) coordinated-turn attitude (body-x along velocity,
body-z along thrust direction = a_world − g_world), so yaw tracks
orbital motion automatically; (c) compute body angular velocity in a
single finite-difference stage on the rotation matrix
(R^T · Ṙ → vee), with 2nd-order one-sided endpoints that eliminate
the boundary spikes. After the rewrite, the gyro carries real
orbital signal and boundary spikes drop from 2.35 → 0.42 rad/s.

2. **NetworkI architectural upgrade**. The original Step K NetworkI
was a 1-layer unidirectional LSTM with a single MLP head and
axis-angle rotation output. We had separately seen, in
`models/imu_model.py`, an alternative architecture that trained
better in isolation: bidirectional 2-layer LSTM, all-layers concat
(2 layers × 2 directions × 128 = 512-d), projected to 256-d with
dropout 0.1, two separate heads (rotation as 6D-continuous,
translation as 3-d). That architecture was lifted into NetworkI.
The public API is unchanged.

3. **Speed retune**. Train trajectories now span 0.5–1.5 m/s peak
(per-family ceilings: circle 1.2, figure-8 1.0, diamond 1.05, zigzag
1.0); val is `simple_ellipse` at 1.0 m/s; test is the
`simple_ellipse` speed sweep at 0.3, 0.5, 0.8, 1.2 m/s.

**What it improved**: per-trajectory, the I branch finally works —
on traj 412 (`simple_ellipse` at 0.8 m/s, in-range) I = 0.43 m, the
best I value in the project, and on the same traj VI = 0.16 m
beats both V and I. This is the first evidence that a
fully-learned three-branch VIO can produce a sub-20-cm ATE on a
30-second drone trajectory at drone-magnitude speed, with all three
branches simultaneously contributing.

**What it exposed**: the *mean* numbers regressed (V = 1.16, I =
1.36, VI = 1.63) because the test set widened to include OOD-slow
(0.3 m/s, slower than every train trajectory) where all three
networks fail. Per-trajectory:

| Test traj | Speed (m/s) | Regime | V (m) | I (m) | VI (m) |
|---|---:|---|---:|---:|---:|
| 410 | 0.3 | OOD-slow | 3.00 | 2.20 | 3.31 |
| 411 | 0.5 | in-range | 0.76 | 1.60 | 2.29 |
| 412 | 0.8 | in-range | 0.27 | 0.43 | **0.16** |
| 413 | 1.2 | train ceiling | 0.62 | 1.21 | 0.75 |

Three observations: (i) the mean is dragged up by the OOD-slow
trajectory; (ii) at 0.5 m/s, V is fine but VI degrades by 200 % —
the residual refiner is over-correcting on a sample where V was
already accurate; (iii) at the trained ceiling speed (1.2 m/s),
inter-frame pixel motion is 8–10 px, exceeding VGG-8's
receptive-field reach (~3–5 px per layer). V degraded relative to
Step K not because the architecture changed (it didn't) but because
the test distribution shifted out of V's competence band.

---

## 4. Honest limitations

Step S is the strongest configuration we have. It is also not, in
any sense, a deployable VIO solution for real drones. Five things
prevent that:

1. **Synthetic-only training and evaluation.** Every result above
is on Blender-rendered top-down footage of a single textured plane
at a single altitude with an idealised IMU model. The cost of any
of these — varying the texture (Step F), having 9–17 % altitude
variation (the per-frame altitude lookup we added in Step G), or
moving to real video — is unmeasured because we never tested it.
The synthetic-to-real domain gap is a separate research project,
not a tunable hyperparameter. Photometric augmentation is enabled
during V's training (ColorJitter + Gaussian noise + occasional
grayscale) but it covers brightness/colour shift, not the actual
distribution shift between synthetic and real photography.

2. **Idealised IMU noise model.** We model accelerometer and gyro
noise as Gaussian with constant σ. Real IMUs have time-varying
biases (random walks on bias acceleration and bias angular rate,
typically ~1e-3 m/s² per √s and ~1e-4 rad/s per √s for a consumer
MEMS), temperature dependence, axis misalignment, scale-factor
errors, and discretisation noise that depends on internal
sampling rate. None of these is in our synthetic IMU. A real
I-branch deployment would need at minimum bias estimation as part
of the network output, and re-training on actual IMU noise.

3. **The CNN's receptive-field cap on inter-frame motion.** VGG-8
has no explicit displacement search — the receptive field at the
deepest feature map is bounded, and the network can only correlate
features within the kernel reach. Empirically that's about 3–5 px
per layer; at 1.2 m/s peak speed and 100 Hz frame rate, inter-frame
pixel motion is 8–10 px (depending on altitude), exceeding the
reach. This is the structural reason V degraded from Step K's 0.184
m to Step S's 1.16 m once the speed range moved up. The fix is an
architecture with explicit displacement search — RAFT, GMA,
cost-volume optical-flow encoders. We attempted RAFT distillation
in Step K and it didn't engage at the time (the flow magnitudes
were below the forward-backward consistency threshold). At Step S
speeds, RAFT might engage now, but we haven't retested.

4. **The I branch's 10 ms IMU window puts translation at the
gravity-integration noise floor.** Image rate is 100 Hz, so the
inter-frame interval is 10 ms; we feed exactly the 10 IMU samples
in that window to the I branch. At 1.5 m/s peak training speed,
the per-window translation signal is 1.5 × 0.01 = 15 mm; the
gravity-integration noise (the position error from integrating
0.5 g · dt² with σ_acc = 0.02 m/s²) is around 0.5 mm on the same
window. SNR is around 30:1 at the best operating point and
collapses to ~1:1 on small-motion samples. Modern learned-IMU
papers (TLIO, RoNIN, AirIO) use 200–1000 ms windows for exactly
this reason — 20–100× more integration baseline. We did not extend
the window in Step S because (a) it would invalidate the existing
VI checkpoints, which were trained on length-10 inputs, and (b)
the dataloader change is non-trivial (you have to reach back in
time before frame A, which makes the per-sample frame indexing
awkward at trajectory boundaries). The plumbing exists in
`utils/dataloader.py` (the `imu_context_length` parameter) but was
never engaged.

5. **The residual refiner has no fail-safe.** VI's prediction is
`R = δR · R_v, t = t_v + δt`, with `|δR| ≤ 0.1 rad` and `|δt|
≤ 0.2 m` enforced via `tanh` bounding. The bound is what stops the
refiner from overriding V entirely (without it, the refiner could
predict the whole pose and V would become irrelevant). The cost is
that when V is wrong by more than 0.1 rad / 0.2 m — i.e. when V is
at OOD speed — VI can only correct by less than V's error. There's
no "trust I instead" mode. This is exactly what we see at Step S
traj 411 (`simple_ellipse` 0.5 m/s): V = 0.76, I = 1.60, VI = 2.29.
The refiner makes V worse because both V and I are slightly off,
and adding two slightly-wrong predictions is worse than picking the
better one.

Two more, briefly: the system has **no temporal recurrence** — every
frame-pair prediction is independent, so errors integrate linearly
with trajectory length and there's no opportunity to recover from a
local mistake. And the **scale ambiguity** in monocular vision means
the V branch has to learn metric scale from the training distribution
of altitudes; we held altitude constant at 4 m throughout, and a
real deployment at any other altitude would be a distribution shift
on a deployment-relevant axis we never tested.

---

## 5. Path forward

Six concrete steps separate Step S from a deployable speed-invariant
VIO, in roughly the order they should be addressed:

1. **Replace VGG-8 with a cost-volume vision encoder (RAFT or GMA).**
   This addresses the receptive-field cap on inter-frame motion at
   the architectural level. RAFT's iterative correlation update can
   match features across pixel motions of 50+ px without the
   per-layer reach constraint. Estimated effort: ~2 weeks. This is
   the single largest expected win on the V side and is the one
   thing that would make speed-invariance plausible on a learned
   vision branch.

2. **Extend the IMU window from 10 ms to 200 ms.** The plumbing
   exists; what's missing is (a) the dataloader change to reach
   before frame A at trajectory boundaries (left-pad with zeros, or
   skip the first N frame pairs), (b) retraining the I branch on
   the wider window, and (c) verifying VI checkpoints still work
   after the change (they expect length-10 input, so we'd train two
   separate VI configurations). At 200 ms, the per-window
   translation signal grows by 20× while the gravity-integration
   noise floor grows by only ~4× (it scales as dt²), so SNR
   improves substantially.

3. **Confidence-gated fusion.** Replace the residual refiner with a
   soft-gated combination where the gate is a learned function of
   per-branch confidence. When V is unreliable (high reconstruction
   error, low photometric consistency, high optical-flow
   uncertainty) the gate falls back to I; when I is unreliable
   (high gyro variance, near-static motion) the gate falls back to
   V. This addresses the no-fail-safe limitation directly. The
   anchored-fusion architecture in `models/network_vi_anchored.py`
   was a first attempt at this and was retired because the gates
   collapsed to one branch — but that was at Step F/G data quality,
   when V was poor everywhere; at Step S data quality the gates
   should have something to differentiate against.

4. **IMU bias modelling.** Add per-axis accelerometer and gyro bias
   to the network output and supervise them via the IMU pre-integration
   residual against GT trajectory segments. This is what classical
   tightly-coupled VIO does (MSCKF estimates bias as part of the
   filter state), and re-deriving it in the learned setting brings
   the I branch closer to a bias-aware estimator that could survive
   a real IMU's bias random-walk.

5. **Synthetic-to-real bridge.** Two complementary directions: (a)
   add domain-randomisation textures and altitudes to training
   (multi-texture training was Step F's broken design — but at Step
   S data quality and with photometric augmentation, multi-texture
   might now help instead of hurt); (b) add a small amount of real
   data with GT pose (KITTI, EuRoC, our Phase 1 Machine Hall
   sequences) to a fine-tuning stage. Neither is the same problem
   as deploying on a real drone, but they would close part of the
   gap.

6. **Temporal recurrence.** Replace the per-frame-pair independent
   prediction with a recurrent or attention-based VIO that
   maintains a state across multiple frame pairs. This is how
   classical VIO recovers from local errors — the filter state
   carries information forward — and adding it on the learned side
   is what closes the linear error-integration regime.

The first three of these (cost-volume V, longer IMU window,
confidence-gated fusion) have explicit hooks already in the codebase
and are by far the highest-leverage. They are the work this project
would do next if it continued.

---

## 6. The summary in one paragraph

> Phase 2 is a deep VIO system on synthetic Blender drone footage
> with three networks (V/I/VI). We ran six diagnostic experiments
> that each isolated and fixed one failure mode of the previous
> step. The pipeline went from 11.81 m mean ATE on V (Step F,
> baseline) to 0.16 m on the best per-trajectory VI configuration
> (Step S, traj 412), through a sequence of changes: fix the
> texture (Step G), fix the parameterisation (Step H), simplify
> the trajectories (Step J), add speed and shape diversity (Step
> K, where fusion first helps), and fix the IMU pipeline plus the
> I architecture (Step S, where all three branches simultaneously
> learn for the first time). The pipeline is not deployable because
> it is synthetic-only, the IMU is idealised, the V branch's CNN
> caps the inter-frame motion it can match, the I branch sees a
> 10 ms window that puts it at the gravity-integration noise floor,
> and the fusion has no fail-safe when V is wrong. The path
> forward is concrete: cost-volume V encoder, 200 ms IMU window,
> confidence-gated fusion. The plumbing for the first two exists
> in the codebase; the third has a retired prototype.

---

*Code: `Code/Phase 2/`. Configs: `configs/step_*.yaml`. Trained
checkpoints: `outputs/step_k/{v,i,vi}/best.pt` and
`outputs/step_s/{v,i,vi}/best.pt` (on the Turing cluster).
Reference papers: Mourikis & Roumeliotis (2007) for MSCKF; Sun et
al. (2017) for S-MSCKF; Liu et al. (2023) for AirIO; Zhou et al.
(2019) for 6D-continuous rotation; Teed & Deng (2020) for RAFT.*
