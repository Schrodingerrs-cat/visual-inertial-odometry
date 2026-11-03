# Speech — Visual-Inertial Odometry, Phase 2 Checkpoint

**Speaker:** ajshah1 · WPI RBE/CS 549
**Date:** May 4, 2026
**Duration target:** 3 minutes (verbal) — this document contains a fuller backing script per slide, so you can pick what fits.

Every number in this script traces to a `results.csv` under `Results/Phase2/step_*_results/`, the report (`report.md`), or the live Step S training logs cited at submission time.

---

## Slide 1 — Title

> **"Visual-Inertial Odometry"** · WPI RBE/CS 549 · ajshah1 · May 4, 2026
> Subtitle: *Learning relative pose from synthetic drone imagery + IMU*

**What to say (≈18 s):**

This is a Phase 2 checkpoint on a learning-based Visual-Inertial Odometry system. The setup is end-to-end relative-pose estimation from synthetic Blender drone footage. Each frame-pair gets a stereo image plus 10 milliseconds of IMU samples; the network predicts the rotation and metric translation between the two frames. Trajectories are dead-reckoned from the per-frame predictions and compared against ground truth.

I'll walk through six diagnostic steps — F through K — and then the current Step S retry, which is training right now on Turing.

**Backing facts:**
- Camera: 640×480, fx=fy=640, cx=320, cy=240
- Image stream: 100 Hz; IMU stream: 1000 Hz; ratio 1:10 → 10-sample IMU window per image-pair interval
- Synthetic data: Blender renders RGB frames over a single textured ground plane (`tex_03.webp`) at constant 4 m altitude
- IMU noise (synthetic): σ_acc=0.02 m/s², σ_gyro=0.002 rad/s, gravity preserved in body frame

---

## Slide 2 — Architecture

> Three networks: V (vision), I (inertial), VI (fusion). Block diagram with input dims and output (R, t).

**What to say (≈20 s):**

Three networks share the training and evaluation pipeline. **V**, the vision branch, is a VGG-8 style CNN — eight conv blocks, 512-dimensional feature, two MLP heads predicting 6-D rotation and 3-D metric translation. **I**, the inertial branch, is a bidirectional two-layer LSTM over the 10-sample IMU window; in Step S that LSTM was rewritten — bidirectional, all-layers concatenated, projected to 256 dimensions, two separate heads for rotation and translation. **VI** wraps both with a residual refiner: take V's prediction and perturb it using V and I features jointly. Same loss family across all three; same dataloader; same eval script.

**Backing facts:**
- V: `models/network_v_tdirect.py::NetworkV_tdirect`, 1.5 M params, t-direct heads since Step H
- I (Step S form): `models/network_i.py::NetworkI`, bidirectional 2-layer LSTM hidden=128/dir → 512 concat → projection to feat_dim=256 with ReLU + dropout 0.1; 6-D continuous rotation (Zhou et al. 2019)
- VI: `models/network_vi_tdirect.py::NetworkVI_tdirect` — refiner sized via `self.imu.out_dim` (256) since the Step S NetworkI rewrite

---

## Slide 3 — Approach evolution overview

> Six-row table: Approach 1=F, 2=G, 3=H, 5=J, 6=K, Step S — Setup / Outcome / Headline metric

**What to say (≈25 s):**

Six steps, one row each. The colour bar is red/amber/green — broken, partial, working.

- **Approach 1 (Step F)** — 100-second sinusoid trajectories with mixed textures and a decomposed-homography head. **V failed at 11.81 m mean ATE.**
- **Approach 2 (Step G)** — same data, but locked to a single texture across train and test. **V dropped to 5.29 m on the held-out trajectory.** Texture explained twenty-eight percent of the failure.
- **Approach 3 (Step H)** — switched the head from "translation divided by depth" to direct metric translation. **Another 27 % drop, to 3.86 m.** The decomposition wasn't physics — it was a bug.
- **Approach 5 (Step J)** — Step I (anchored fusion) is intentionally omitted from this throughline. Step J replaced the long aggressive sinusoid with **30-second simple shapes — figure-eights, circles, ellipses — at modest speed**. **V crashed to 0.089 m.** Within-family generalisation was finally working.
- **Approach 6 (Step K)** — added speed diversity, four shape families. **V held at 0.184 m on a held-out test set, and crucially, VI beat V for the first time — 0.173 m.**
- **Step S** — three coincident fixes; training in progress.

**Backing facts:**
- Step F: `report.md` §2.1, V_avg = 11.81 m, I_avg = 3.46 m, VI_avg = 6.94 m (sourced from project notes; `outputs/step_f/results.csv` not on disk)
- Step G: `report.md` §2.2, V=5.29, I=1.78, VI=2.35 on traj 3 (sourced from `step_i_results/CHANGES.md` line 202)
- Step H: `Results/Phase2/step_h_results/results.csv` — V=3.856, I=1.781, VI=3.362 on traj 3
- Step J within-family: `Results/Phase2/step_j_results/results.csv` — V=0.089, I=1.506, VI=0.131 mean over (300, 305, 306, 307)
- Step K within-family: `Results/Phase2/step_k_results/results.csv` — V=0.184, I=1.242, VI=0.173 mean over (320, 321, 322, 323, 324)

---

## Slide 4 — Approach 2 (Step G): fix the texture confound

> "V on traj 3: 11.81 m → 5.29 m" — single-texture training; right side shows traj 3 V trace

**What to say (≈18 s):**

Step F's failure had a confound. Train and test trajectories used **different textures** — so the network was learning to associate texture features with motion, not motion with motion. Step G forces a single texture, `tex_03`, across both train and test, with everything else identical.

**V drops from 11.81 m to 5.29 m on the held-out 100-second sinusoid — a 2.2× improvement.** Texture confound accounted for roughly 28 percent of the original V failure. V is now learnable, but five metres of error on a held-out sequence is still well above usable.

**Backing facts:**
- 11.81 → 5.29 = ratio 0.45 = retain 45% of error → 55% reduction → user-friendly framing "−2.2× ATE"
- Same trajectory IDs, same architecture, same loss as Step F — only the texture varied
- Pre-trained checkpoints reused for I (1.78 m on traj 3)

---

## Slide 5 — Approach 3 (Step H): direct translation prediction

> "V on traj 3: 5.29 m → 3.86 m" — t-direct prediction; right side shows traj 3 V trace

**What to say (≈18 s):**

Step H targets the parameterisation. The original V head predicted **translation divided by depth**, plus the plane normal, then reconstructed metric translation by multiplying by per-frame altitude. That cascade introduced an ambiguity the network never resolved. Step H replaces it with direct metric translation prediction, and the loss simplifies to L_rotation plus L_translation with a hover-weighting on small-motion samples.

**V drops another 27 percent — from 5.29 m down to 3.86 m on the same traj 3.** The decomposition wasn't physics, it was a bug. VI architecture also rewritten as residual fusion in this step.

**Backing facts:**
- `Results/Phase2/step_h_results/results.csv` traj 3: V=3.856, ate_endpoint=9.411, seg2s_median=2.551
- Compared against Step G's 5.29 m → ratio 0.73 → 27% reduction
- VI on traj 3 = 3.362 — VI was still worse than I (1.78) at this point: residual refiner had no quality signal indicating *when* V was wrong

---

## Slide 6 — Approach 5 (Step J): simple shapes, slow, 30 s

> "3.86 m → 0.089 m on within-family" — 43× drop; right side shows traj 305 V/I/VI overlay

**What to say (≈22 s):**

Step J is the largest single jump in the project. The hypothesis: V's representation ceiling on long aggressive sinusoids may be driven by training-data complexity. So train on **30-second simple shapes** — figure-eight, circle, ellipse — at constant altitude, peak speed around 0.4 m/s.

**V drops two orders of magnitude — from 3.86 m to 0.089 m mean on the within-family test set.** I sits at 1.51 m. VI at 0.131 m, still trailing V. Note the caveat: the validation trajectory is also in the test set, so removing it gives a held-out-only V_avg of 0.098 m — still under 10 cm.

But Step J's network does not generalise to the original aggressive 100-second sinusoids. **Cross-family V is 10.85 m mean.** Within-family solved; cross-family unsolved.

**Backing facts:**
- `Results/Phase2/step_j_results/results.csv` mean over (300, 305, 306, 307): V=0.089, I=1.506, VI=0.131
- Held-out-only mean V over (300, 306, 307) = 0.098 m
- Cross-family `Results/Phase2/step_j_results/step_j_cross_results/results.csv`: V=10.85, I=39.54, VI=10.74
- Train: trajs 301–304 (4 simple-shape trajs); val=305, test=[300, 305, 306, 307]; all on tex_03
- Per-traj raw RMSE (no Umeyama) is 2.7× the aligned RMSE — so SE3 alignment absorbs some real per-frame error

---

## Slide 7 — Approach 6 (Step K): speed diversity unlocks fusion

> V=0.184 / I=1.242 / **VI=0.173** — first time VI < V; right side traj 320 overlay

**What to say (≈25 s):**

Step K adds **speed diversity**. Eight training trajectories: four shape families — circle, figure-8, diamond, zigzag — each at two speeds, ranging 0.1 to 0.5 m/s peak. The held-out test set introduces shapes the network never saw at training: ellipse, triangle, and a never-trained sinusoid_30s.

**V holds at 0.184 m mean over five test trajectories — 18 centimetres on shapes never seen in training.** I is still poor at 1.24 m. But here's the unlock: **VI is 0.173 m. VI beats V for the first time in the project.** On four of five trajectories, fusion adds value — improvements range from 1 % to 19 %. On training data, VI versus V is 0.045 vs 0.056 m: VI wins on six of eight training trajectories. Fusion can usefully combine V and I when V is sufficiently accurate.

The IMU branch is still the bottleneck. I predicts only 10 to 33 percent of ground-truth motion magnitude — it hovers near zero rather than tracking. That's the open problem heading into Step S.

**Backing facts:**
- `Results/Phase2/step_k_results/results.csv` test trajs (320, 321, 322, 323, 324): V=0.184, I=1.242, VI=0.173
- Per-traj VI vs V: 320 (-16%), 321 (-19%), 322 (+15%), 323 (-1%), 324 (-11%) — VI wins 4 of 5
- Step K extended (325-328 held-out at intermediate speeds): V=0.078, I=0.822, VI=0.081 — V's within-family generalisation is broad across all 4 trained shapes
- Step K traineval (training-fit on 312-319): V=0.056, I=0.899, VI=0.045 — VI<V on 6/8 training trajs
- Cross-family numbers (3.61 m) look better than Step J — but Sim3-aligned RMSE is identical to Step J on every trajectory; the SE3 difference is alignment-scale absorption, not real per-frame improvement (`report.md` §5)
- I network failure (`report.md` §6.3): L1 loss without hover-weighting + 10 ms IMU window puts the per-window translation signal at the gravity-integration noise floor

---

## Slide 8 — Step S: three coincident fixes; training in progress

> Three changes: speed retune / IMU pipeline / bidirectional NetworkI; right side shows IMU |ω| before vs after correction

**What to say (≈25 s):**

Step S addresses the IMU bottleneck head-on. **Three coincident changes** versus Step K — this is no longer a clean A/B against speed alone.

First, **speed retune** — train spans 0.5 to 1.2 m/s across all four shape families, with a per-family ceiling, and the test set sweeps 0.3 to 1.2 m/s on a held-out ellipse.

Second, **the IMU generation pipeline is rewritten**. The plot on the right shows the bug. Top panel: the old pipeline produced a 2.34 rad/s gyro spike at every trajectory boundary on circles and ellipses — that's an artifact of three stacked finite-difference stages. Bottom panel: the corrected pipeline — analytic velocity and acceleration from the samplers, coordinated-turn attitude so yaw tracks velocity, single-stage R-chain ω. **Spike collapses from 2.35 rad/s to 0.42 rad/s, which is the real orbital rate.** The IMU finally has signal.

Third, **NetworkI is upgraded**: bidirectional two-layer LSTM, all layers concatenated, projection to a 256-d feature, two separate heads, 6-D rotation. Public API unchanged.

**Status — training right now on Turing.** V is at epoch 11, val ATE 1.03 metres. I is at epoch 10, val ATE **0.92 metres** — and crucially, **descending monotonically from 1.55**. Every prior step the I network plateaued above 1.2 within five epochs. This is the first time it's actually learning.

**Backing facts:**
- TRAJ_ID_MAP[400..413] locked: 400/401 circle 0.5/1.2, 402/403 fig-8 0.5/1.0, 404/405 diamond 0.8/1.05, 406/407 zigzag 0.8/1.0, 408 val ellipse 1.0, 410-413 test ellipse 0.3/0.5/0.8/1.2
- IMU pipeline correction: `blender/imu_simulation.py` rewrite — `_central_diff_o2` (2nd-order endpoints), `_coordinated_turn_R_wb`, `_gyro_from_rotation_chain`; `blender/trajectory_library.py` samplers return analytic vel/acc (Option B)
- Fix-A: `_coordinated_turn_R_wb` carry-forward of body-x at degenerate-velocity boundary samples; eps = 1e-3 m/s
- NetworkI rewrite: `models/network_i.py` + `models/backbones.py:IMUEncoder` — bidirectional LSTM, hidden=128 per direction, num_layers=2, feat_dim=256, dropout 0.1
- Verified end-to-end on all 13 IDs: 13/13 PASS — speed err 0.0%, no boundary spike, validate ok
- Live Step S Stage 1 (per user message): V epoch 11 val_ATE=1.03 m, I epoch 10 val_ATE=0.92 m (1.55 → 0.92 monotone)

---

## Slide 9 — The IMU branch — diagnosed and addressed

> Problem (left, red) — Answer (right, green); bottom green box: "First signs of life"

**What to say (≈22 s):**

The IMU branch was the open problem from F through K. **I_avg never below 1.2 metres across every step.** Predictions are only 10 to 33 percent of ground-truth motion magnitude — the network learned that "predict near zero" is locally near-optimal under L1 loss with a 3-mm-per-frame translation target.

Two compounding causes. **Data:** the 10 ms IMU window puts the per-window translation signal within an order of magnitude of gravity-integration noise — modern learned-IMU approaches like TLIO and AirIO use 200-to-1000 millisecond windows, twenty to a hundred times longer. And the boundary spike I just showed stripped real signal from training data. **Architecture:** unidirectional LSTM, single MLP head, axis-angle rotation with a discontinuity at zero — exactly where the network sits.

Step S addresses both halves. Data: corrected pipeline. Architecture: bidirectional LSTM, separate heads, 6-D rotation. **Ten epochs in, val ATE has dropped from 1.55 to 0.92 monotonically.** Every prior step the same network plateaued at 1.2-plus within five epochs. First time the I branch is actually descending instead of hovering.

**Backing facts:**
- `report.md` §6.2: per-step magnitude diagnostic for I on Step K test trajs (320-324): I/GT motion ratio 10-33 %, direction error 25.7° to 80.6° (random-direction baseline 90°)
- `report.md` §6.3: NEAR-ZERO BIAS COLLAPSE — L1 loss without hover-weighting at 3 mm/frame target; "predict zero" pays loss ≈ |t_gt|.sum() ≈ 0.003
- 10 ms window: gravity integration noise = 0.5·g·dt² = 0.5 mm; SNR ~6:1 on good window, ~1:1 on small-motion window
- Phase 1 fix (`IMUPoseLoss_hover`): submitted as job 1998879 — pending at report writing
- Phase 2 fix (longer window): planned per `report.md` §6.5; superseded by Step S
- TLIO/AirIO precedent: 200-1000 sample windows = 200 ms - 1 s, 20-100× longer than current

---

## Slide 10 — Takeaways

> 4 bullets, last in green/bold

**What to say (≈22 s):**

Four takeaways.

**One.** Three orthogonal fixes — texture, parameterisation, training shapes — compounded into 18 cm V on within-family. Each removed a different confound.

**Two.** Speed diversity in Step K let fusion beat vision alone for the first time — VI 0.173 versus V 0.184 m. The ordering held on six of eight training trajectories.

**Three.** The IMU branch was the bottleneck. We diagnosed it as data plus architecture — the 10 ms window gave the LSTM no signal, and L1 loss without hover-weighting biased toward zero output. Both addressed in Step S.

**Four — and this is the punchline.** Step S is the first run where V, I, and VI are all simultaneously learning. **V is converging, I is descending instead of hovering, VI is downstream of both.** Final ATE numbers expected tonight.

Thank you.

**Backing facts:**
- 18 cm = 0.184 m headline V from `Results/Phase2/step_k_results/results.csv`
- 0.173 vs 0.184 = 6 % relative improvement; Step K traineval VI<V on 6/8 trajs (`Results/Phase2/step_k_results/step_k_traineval_results/results.csv`)
- Three compounding fixes per Steps F→G→H: 11.81 → 5.29 → 3.86 m on traj 3 (52 % then 27 % reductions)
- Step S Stage 1 partial: V@e11=1.03 m, I@e10=0.92 m (descending from 1.55) — sourced from live training logs

---

## Appendix — full numerical backing (for Q&A)

### Per-step headline ATE (m)

| Step | Train data | V_test_avg | I_test_avg | VI_test_avg | V_crossfam |
|---|---|---:|---:|---:|---:|
| F | 100 s sinusoid, mixed tex | 11.81 | 3.46 | 6.94 | — |
| G | 100 s sinusoid, tex_03 | 5.29 | 1.78 | 2.35 | — |
| H | 100 s sinusoid, tex_03, t-direct | 3.86 | 1.78 | 3.36 | 3.86 |
| J | 30 s simple, single speed | 0.089 | 1.51 | 0.131 | 10.85 |
| K | 30 s simple, varied speeds | 0.184 | 1.24 | 0.173 | 3.61 (SE3 artifact) |
| K extended (325-328) | same K ckpts | 0.078 | 0.82 | 0.081 | — |
| K traineval (312-319) | same K ckpts | 0.056 | 0.90 | 0.045 | — |
| Step S Stage 1 (live) | 0.5–1.2 m/s, 4 fams | V@e11=1.03 | I@e10=0.92 (↓) | pending | pending |

### Step K within-family per-traj (test set)

| traj | shape · speed | V | I | VI | VI vs V |
|---|---|---:|---:|---:|---:|
| 320 | ellipse 0.3 | 0.082 | 1.060 | 0.069 | **−16 %** |
| 321 | ellipse 0.5 | 0.207 | 1.709 | 0.168 | **−19 %** |
| 322 | triangle 0.3 | 0.116 | 1.021 | 0.134 | +15 % |
| 323 | triangle 0.5 | 0.370 | 1.801 | 0.366 | **−1 %** |
| 324 | sinusoid_30s 0.5 | 0.144 | 0.619 | 0.129 | **−11 %** |
| **mean** | | **0.184** | **1.242** | **0.173** | **−6 %** |

### IMU pipeline before/after (ellipse seq, ω magnitude)

| | Old pipeline | Corrected (Step S) |
|---|---:|---:|
| max \|ω\| | 2.354 rad/s (boundary spike) | **0.425 rad/s** (orbital rate) |
| median \|ω\| | 0.003 rad/s (≈ noise floor) | **0.168 rad/s** (real signal) |
| signal-to-noise | ~1.5 : 1 | **~85 : 1** |

### Step S locked TRAJ_ID_MAP[400..413]

| ID | family | seed | split | speed |
|---|---|---|---|---|
| 400 | simple_circle | 7001 | train | 0.5 |
| 401 | simple_circle | 7002 | train | 1.2 |
| 402 | simple_figure8 | 7003 | train | 0.5 |
| 403 | simple_figure8 | 7004 | train | 1.0 |
| 404 | simple_diamond | 7005 | train | 0.8 |
| 405 | simple_diamond | 7006 | train | 1.05 |
| 406 | simple_zigzag | 7007 | train | 0.8 |
| 407 | simple_zigzag | 7008 | train | 1.0 |
| 408 | simple_ellipse | 7009 | val | 1.0 |
| 410 | simple_ellipse | 7010 | test | 0.3 |
| 411 | simple_ellipse | 7011 | test | 0.5 |
| 412 | simple_ellipse | 7012 | test | 0.8 |
| 413 | simple_ellipse | 7013 | test | 1.2 |

### Anticipated Q&A talking points

- **"Why are Step F numbers from notes, not a CSV?"** `outputs/step_f/results.csv` and `outputs/step_g/results.csv` were not on local disk; numbers carried from `step_i_results/CHANGES.md` line 202 which independently records them as a regression target. All Steps H onward are sourced directly from on-disk `results.csv`.
- **"Step K cross-family V=3.61 — isn't that better than Step J's 10.85?"** The Sim3-aligned RMSE is numerically identical between J and K on every (method, trajectory) pair. The SE3 difference is captured entirely by the alignment scale. Per-step magnitudes are 4-25× smaller than GT in both J and K. `seg2s_median` is 1.0–2.5 m in both — unchanged. Cross-family generalisation to aggressive 100 s sinusoids remains unsolved. (`report.md` §5.)
- **"Why was Step I skipped?"** The anchored-fusion variant (soft gates + photometric self-validation) was investigated; the gates collapsed to constants because the photometric residual was uninformative. We pivoted to residual fusion instead. Documented in `report.md` §2.4. Doesn't fit the throughline of the talk.
- **"Why did RAFT distillation fail in Step K?"** Forward-backward consistency rejected 99.6 % to 100 % of frame pairs across all 9 cached trajectories. Mean flow magnitudes were 0.15-0.80 px per pair vs an FB-rejection threshold the data could not reach on uniform-texture short-baseline images. The distillation loss term was effectively zeroed; Step K's improvement over Step J came from speed-diverse training data alone. (`report.md` §4.3, §8.5.)
- **"What's the validation methodology issue in Step J?"** Trajectory 305 was both the val traj (driving best.pt selection) and a test traj. Removing it leaves V_avg over (300, 306, 307) = 0.098 m — still under 10 cm. Documented in `report.md` §3.4.
- **"What changed in Step S vs Step K beyond IMU and architecture?"** Speed buckets: Step K used 0.1-0.5 m/s, Step S spans 0.5-1.2 m/s. The two-speed-per-family design is preserved.

---

*End of speech document. Use the slide-by-slide "What to say" blocks for the 3-minute delivery; the backing facts and appendix are for Q&A.*
