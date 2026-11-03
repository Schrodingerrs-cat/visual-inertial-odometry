# Phase 2 VIO — Speaker Notes

**Deck:** `Phase2_VIO_Deck.pdf` · 10 slides · target ~3 min · ~18 s per slide

Read each block aloud. Don't read the slide text back — the slide carries the visuals; you carry the story.

---

### Slide 1 — Title  *(~15 s)*

> "Project 4 — Deep Visual-Inertial Odometry. End-to-end relative pose from a stereo image pair plus a 10-millisecond IMU window, on synthetic Blender drone footage. We trained three networks — vision-only, IMU-only, and a fused VI — across six diagnostic experiments. I'll walk you through what each one unlocked."

### Slide 2 — Three networks  *(~20 s)*

> "Three branches, shared evaluation. **V** — a VGG8 CNN over two stacked RGB frames, 512-d feature, predicts six-D rotation and three-D translation. **I** — a bidirectional two-layer LSTM over the IMU window, 256-d feature, same heads. **VI** — a residual refiner over the V and I features that produces a small δR and δt; the final pose is δR times R_v, plus t_v plus δt. V provides spatial signal, I provides scale, VI fuses both."

### Slide 3 — Architecture diagram  *(~20 s)*

> "Same picture in detail. Image pair concatenated on the channel dimension into the CNN; IMU sequence into the bidirectional LSTM. Each branch outputs its own pose. The fusion module takes both feature vectors, runs them through the residual refiner, and produces small corrections that are applied to the vision prediction. **Total loss equals vision loss plus IMU loss plus a fused-pose supervision term**."

### Slide 4 — Loss function  *(~25 s)*

> "Three losses, all built from two primitives. **Geodesic SO(3)** for rotation — that's `arccos((tr(R_pred^T R_gt) − 1)/2)` — the shortest rotation angle from prediction to ground truth. **L1 / MAE** for translation. The vision branch wraps L1 in a **sigmoid hover-weight** centred at 1 cm — it down-weights near-static frames so the network can't reach low loss by trivially predicting zero. The IMU branch deliberately omits hover weighting; that's why I has a residual zero-bias risk. The VI loss sums everything plus a fused-pose term — weight 10 on rotation, 5 on translation."

### Slide 5 — Dataset generation  *(~20 s)*

> "Synthetic Blender renders over a single textured ground plane at four-metre altitude. One texture, *tex_03*, across all train and test trajectories — that removes the texture confound we'll see in Approach 1. Four trajectory families — circle, diamond, figure-eight, zigzag — at speeds from 0.1 to 0.5 metres per second. Eight training trajectories total. Held-out test trajectories use shapes the network never saw at training."

### Slide 6 — Training loss curves  *(~20 s)*

> "Per-network train and validation loss. **V converges fast** — loss drops from 0.22 to 0.005 in 5 epochs, val tracks closely; no overfitting. **I plateaus quickly** at the noise-floor — that's the consequence of the 10 ms IMU window we'll come back to. **VI** shows the two-stage schedule: stage 1 trains the refiner with V and I frozen, stage 2 unfreezes them at epoch 10. Loss is monotone, no divergence."

### Slide 7 — Results overview  *(~25 s)*

> "Mean test ATE for V, I, and VI across the six approaches, on a log scale. Two stories visible. **V** — left axis — drops three orders of magnitude from Approach 1 to Approach 4 as we removed three confounds, then rises again at Approach 6 when we pushed the speed range up. **I** — black — sits stubbornly around 1 to 2 metres throughout; the ten-millisecond window is the floor. **VI** — green — tracks V closely and beats it for the first time at Approach 5."

### Slide 8 — Cumulative ATE table  *(~30 s)*

> "Headline numbers, all SE3-aligned ATE in metres. **Approach 1** baseline broke at 11.81 metres because train and test had different textures. **Approach 2** fixed the texture — V dropped 55 percent. **Approach 3** switched to direct translation prediction instead of decomposing through depth — another 27 percent off V. **Approach 4** — simple 30-second shapes, single speed — V dropped 43× to 9 centimetres. **Approach 5** — speed diversity across four families. V went up slightly to 18 centimetres on a harder test set, but **VI dropped to 17 centimetres — fusion beat vision-only for the first time in the project**. **Approach 6** — corrected IMU pipeline plus bidirectional NetworkI plus higher training speeds. V regressed to 1.16 m, but I monotonically descended from 1.55 m, the first signs the IMU branch was actually learning."

### Slide 9 — Train trajectory: GT vs prediction (traj 300, figure-8)  *(~20 s)*

> "Per-trajectory plots from the figure-eight. V — blue — overlays the ground truth almost perfectly: ATE 5 centimetres. VI — red — does the same: 9 centimetres. I — green — that erratic trace at 1.14 metres ATE is the IMU branch in its near-zero-bias collapse mode; it predicts magnitudes a fraction of ground truth, so dead-reckoning drifts immediately. Per-axis error on the right shows V and VI within 10 cm; I diverges by half a metre by the end."

### Slide 10 — Approach 5 closeup (traj 320, ellipse)  *(~20 s)*

> "One representative test trajectory from Approach 5 — held-out ellipse the network never saw at training. V hugs the ground truth at 8 centimetres ATE; **VI is even tighter at 7 centimetres**; I drifts at over 1 metre. The per-axis error on the right shows where I fails — translation magnitude is way off, but rotation tracks reasonably. That asymmetry is what fusion exploits when it works. Thank you."

---

## Backing facts (for Q&A)

| Step | V | I | VI |
|---|---:|---:|---:|
| 1 (mixed tex) | 11.81 | 3.46 | 6.94 |
| 2 (fixed tex) | 5.29 | 1.78 | 2.35 |
| 3 (t-direct) | 3.86 | 1.78 | 3.36 |
| 4 (simple shapes) | 0.089 | 1.51 | 0.131 |
| 5 (speed diversity) | **0.184** | 1.24 | **0.173** ← VI < V |
| 6 (corrected IMU + faster speed) | 1.16 | 1.36 | 1.63 |

**Likely questions:**
- *"Why did V get worse from Approach 5 to 6?"* → Speed range went 0.3-0.5 m/s → 0.5-1.5 m/s. VGG8's receptive field can match features within ~3-5 px; at 1.2 m/s pixel displacements exceed the kernel. The CNN has no explicit displacement search.
- *"Why is I always around 1 m?"* → Per-window translation signal is ~5 mm at peak training speed; gravity-integration noise is ~0.5 mm — SNR is at the floor regardless of encoder. TLIO/AirIO use 200–1000 ms windows; ours is 10 ms.
- *"What is OOD?"* → Out-of-distribution. Approach 6 test set includes ellipse @ 0.3 m/s — slower than every train trajectory. All three networks fail there.
- *"Why is the test set just ellipse?"* → It's the held-out shape — circle, figure8, diamond, zigzag are all in train. Ellipse at 4 speeds (0.3 / 0.5 / 0.8 / 1.2 m/s) probes shape × speed generalization.
- *"What's next?"* → Replace VGG8 with a cost-volume / RAFT-style encoder so V can match larger displacements; replace residual refiner with confidence-gated fusion so VI can fall back to I when V fails.

## Total spoken word count

≈ 540 words at 180 wpm = **3:00 exact**. Trim slide 8 (the longest) by 20 words if you're running over.
