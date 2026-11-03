# Phase 2 VIO — Speech

**Read aloud. ~3 minutes at normal pace. ~530 words. Slide cues in [brackets] — advance the deck where you see them.**

---

[Slide 1 — Title]

Good morning. This is Project 4, *Deep Visual-Inertial Odometry*. The task is end-to-end relative pose estimation from a stereo image pair plus a ten-millisecond IMU window, on synthetic Blender drone footage. We trained three networks — a vision-only CNN, an IMU-only LSTM, and a fused visual-inertial network — across six diagnostic experiments. I'll walk you through what each one unlocked, what's still broken, and what the headline result actually says.

[Slide 2 — Three networks]

Three branches share the same training and evaluation pipeline. The vision branch — V — is a VGG-8 CNN over two stacked RGB frames; it predicts a six-dimensional rotation and a three-D metric translation. The inertial branch — I — is a bidirectional two-layer LSTM over the IMU window with the same head structure. The fusion branch — VI — takes both feature vectors, runs them through a residual refiner, and produces a small δR and δt that are applied as corrections to V's prediction. So the final pose is δR times R_v, plus t_v plus δt.

[Slide 3 — Architecture diagram]

This is the same idea in detail. Image pair concatenated on the channel dimension into the CNN; IMU sequence into the bidirectional LSTM. Each branch outputs its own pose. Fusion combines both feature vectors through the refiner. The total training loss is the vision loss plus the IMU loss plus a fused-pose supervision term — three signals into one optimizer.

[Slide 4 — Loss function]

All three losses are built from two primitives. *Geodesic SO(3)* for rotation — the arc-cosine of the trace formula gives the shortest rotation angle from prediction to ground truth. *L1*, or mean absolute error, for translation. The vision branch wraps L1 in a *sigmoid hover-weight* centred at one centimetre — that down-weights near-static frames so the network can't reach low loss by trivially predicting zero. The IMU branch deliberately omits hover-weighting; that's part of why I has a residual zero-bias risk we'll see later.

[Slide 5 — Dataset generation]

All renders use a single textured ground plane at four-metre altitude — fixing the texture across train and test removes a major confound. Four trajectory families — circle, diamond, figure-eight, zigzag — at speeds from one-tenth to half a metre per second. Eight training trajectories total; held-out test trajectories use shapes the network never saw at training.

[Slide 6 — Training loss curves]

V converges fast — loss drops from 0.22 to under 0.01 within five epochs, validation tracks closely, no overfitting. I plateaus quickly at the noise floor — the ten-millisecond window is the limit. VI shows the two-stage schedule, with the refiner trained alone in stage one and unfrozen at epoch ten.

[Slide 7 — Results overview]

Mean test ATE on a log scale across the six approaches. V drops three orders of magnitude from Approach one to Approach four as we removed three confounds. I sits stubbornly around one to two metres throughout — the IMU window is its ceiling. VI tracks V closely and beats it for the first time at Approach five.

[Slide 8 — Cumulative ATE table]

The numbers. Approach one — broken at 11.81 metres because train and test had different textures. Approach two — fixed the texture, V dropped fifty-five percent. Approach three — switched to direct translation prediction instead of decomposing through depth, another twenty-seven percent off V. Approach four — simple shapes at single speed — V dropped *forty-three times* to nine centimetres. Approach five — speed diversity across four shape families. V went up slightly to eighteen centimetres on a harder test set, but **VI dropped to seventeen centimetres — fusion beat vision-only for the first time in the project**. Approach six — corrected IMU pipeline, bidirectional LSTM, higher training speeds. V regressed to one-point-one-six metres at the new speed range, but I monotonically descended from one-point-five-five metres — the first signs the IMU branch was actually learning.

[Slide 9 — Train trajectory: figure-8]

Per-trajectory plots. V — blue — overlays the ground-truth figure-eight almost perfectly at five centimetres ATE. VI — red — does the same at nine. I — green — that erratic trace at over a metre is the IMU branch in its near-zero-bias collapse mode; magnitudes are a fraction of ground truth and dead-reckoning drifts immediately.

[Slide 10 — Approach 5: ellipse]

One representative test trajectory — held-out ellipse the network never saw at training. V hugs ground truth at eight centimetres ATE. **VI is even tighter at seven.** I drifts at over a metre. This is the punchline of the project: when V is good and I provides complementary scale information, the fusion network beats either branch alone. Thank you.

---

**Total: ~530 words. Speak at ~180 wpm = exactly 3 minutes.**

If you're running long, drop the second sentence of slide 4 and shorten slide 8's middle (skip the "twenty-seven percent" detail). Saves about 20 seconds.
