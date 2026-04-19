# Gran Turismo RL Agent — Progress Log

## Table of Contents
- [Project Overview](#project-overview)
- [Hardware & Environment](#hardware--environment)
- [v1: Hand-Crafted Vision Pipeline](#v1-hand-crafted-vision-pipeline)
- [v1 Debugging & Calibration](#v1-debugging--calibration)
- [v1 Training Results](#v1-training-results)
- [Paper Analysis](#paper-analysis)
- [v2: Paper-Aligned Overhaul](#v2-paper-aligned-overhaul)
- [v2 Training Results](#v2-training-results)
- [Takeaways](#takeaways)
- [v3: Hybrid Approach (Proposed)](#v3-hybrid-approach-proposed)

---

## Project Overview

An RL agent that learns to race in **Gran Turismo 4 (PSP)** running on the **PPSSPP emulator**, using only screen capture (no game API/telemetry). The agent observes the game via a virtual camera, processes the visual feed, and outputs steering + throttle/brake actions through a virtual gamepad.

This is fundamentally harder than the Sony papers because:
- **No telemetry** — no velocity, acceleration, angular velocity, tire slip, or track boundaries
- **No synchronous control** — PPSSPP runs in real-time, we capture asynchronously
- **No track data** — no course points, no centerline, no track limits
- **Single consumer GPU** — all training on one machine

---

## Hardware & Environment

| Component | Spec | Constraint |
|---|---|---|
| GPU | NVIDIA RTX 3050 Laptop (6GB VRAM) | Limits CNN size and batch processing |
| RAM | 16GB | Limits replay buffer size (~100K max with 64×64 obs) |
| Game | Gran Turismo 4 (PSP) via PPSSPP | Real-time only, no speed control |
| Capture | USB capture card → OpenCV VideoCapture | 1920×1080 native, resized to 640×480 |
| Controller | Virtual gamepad via `uinput` | Steering + gas/brake (2 continuous actions) |
| Framework | Stable-Baselines3 (SAC) | No QR-SAC, no distributed training |

---

## v1: Hand-Crafted Vision Pipeline

### Architecture

**Observation Space:**
- `frame`: 84×84×3 uint8 — three hand-crafted channels:
  - **Channel 0 — Line**: Distance-transform gradient from blue/red racing line detection. Bright = close to line, dark = far. Gives CNN a spatial "pull toward the line" signal.
  - **Channel 1 — Road**: Binary mask of drivable surface (gray asphalt via HSV thresholding).
  - **Channel 2 — Brake**: Red zone overlay for upcoming brake zones.
- `aux`: 4-dim float32 — `[speed_norm, progress, brake_warning, collision_warning]`

**Vision Pipeline (per step):**
1. `get_frame()` — capture + resize to 640×480
2. `detect_line()` — ROI crop → HSV → blue/red masks → dilate → distance transform → 84×84 gradient + center position
3. `check_collision()` — collision ROI → HSV → spark detection (yellow/orange pixels)
4. `get_progress_percent()` → `get_map_position()` — minimap ROI → red mask → centroid → polar angle = progress
5. `get_road_mask()` — road ROI → HSV → asphalt mask → morphology → 84×84
6. `get_brake_overlay()` — near-field ROI → HSV → red detection → 84×84
7. `get_aux_vector()` — called `get_progress_percent()` and `get_brake_overlay()` AGAIN internally (duplicate!)
8. `_get_collision_warning()` — duplicate of `check_collision()` with lower threshold

Total: **9 vision passes per step**, 3 of which were pure duplication.

**Reward Function (7 components):**
```
reward = r_centering        # +3.0 on line, -2.0 off line
       + r_smoothness       # -|Δsteer| × 2.0
       + r_progress         # Δprogress × 100.0
       + r_speed            # min(speed/50, 1.0) × 0.5
       + r_turning          # bonus for steering toward line
       + r_collision        # -80.0 on spark detection
       + r_loiter           # -20.0 × dt when speed < 2px/s
       + r_time             # -0.1 per step (EXISTENCE PENALTY)
```

**CNN Architecture:**
```
Conv2d(C, 32, 8, stride=4) → ReLU
Conv2d(32, 64, 4, stride=2) → ReLU
Conv2d(64, 64, 3, stride=1) → ReLU → Flatten
FC(flatten_dim, 256) → ReLU  (combined with 32-dim aux MLP)
```

**Training Config:**
```
algorithm:       SAC
learning_rate:   3e-4
entropy:         auto_0.1
buffer_size:     50,000
batch_size:      256
gradient_steps:  2
frame_stack:     4 (VecFrameStack)
observation:     84×84×12 (3 channels × 4 stack)
```

**Termination Conditions:**
- LINE LOST: 1.5s without detecting racing line
- STUCK: 5s with displacement < 1.0
- LOITERING: 3s with estimated speed < 2.0

### v1 Debugging & Calibration

#### Problem 1: All masks broken at 1920×1080
The ROI coordinates were hardcoded for 640×480 but the capture card was feeding 1920×1080. Every vision function was cropping garbage regions.

**Fix:** Resize immediately in `get_frame()` to 640×480 before any processing.

#### Problem 2: Road mask = all white
The tunnel brightness override (`if avg_brightness < 40: return all-white`) was triggering on EVERY frame because the mean V-channel of the 1920×1080 feed was naturally low (~33).

**Fix:** Removed the tunnel override entirely.

#### Problem 3: Line detection = always NONE
Two issues compounding:
- Blue saturation threshold too high (170 → lowered to 120) — capture card desaturates slightly
- Contour area rejection too high (30px → lowered to 10px) — line was detected but rejected as "too small"
- Dilation kernel too large (5×5 → reduced to 3×3) — was blurring the line detection

**Fix:** Recalibrated all HSV thresholds using a new diagnostic tool (`hsv_diag.py`).

#### Problem 4: Distance transform = all white
When the line mask was sparse, `dist * 3` overflowed. The fixed scaling factor (`*3`) didn't adapt to actual mask density.

**Fix:** Normalize by actual maximum: `1.0 - dist / dist.max()`.

#### Problem 5: Instant loitering termination
The loitering counter started at step 0 with speed = 0 (car starts stationary). At 15 FPS, loitering fired after just 45 steps (~3 seconds).

**Fix:** Added 5s grace period, extended loiter window to 8s.

#### Problem 6: Reward farming exploit
The agent could stop in front of a blue advertising sign (SUBARU banner), and the blue pixels would register as "line detected" → +3.0 centering reward while stationary.

**Fix:** Gated centering reward on `displacement > 2.0` (car must be moving).

#### Problem 7: FPS = 5-6 (catastrophically low)
The 9 vision passes per step were bottlenecking at ~200ms/step. Plus `get_frame()` was draining 2 extra frames from the capture buffer (66ms waste).

**Fix:** Removed frame drain, eliminated duplicate vision calls, cached results, throttled dashboard rendering to every 5th step.

---

### v1 Training Results

#### Run 1 (Pre-fix baseline): 216K steps, 12 hours
```
Episodes:     2,610      |  Throughput:  4.8 steps/s
Mean Reward:  -1,152.7   |  Best:        +246.9
Mean Length:  83 steps    |  Median:      61 steps
Reward/Step:  -18.65     |  Trend:       📉 -78.4
Corr (r,l):   -0.119
```
**Diagnosis:** Agent learned "dying at step 61 = optimal" because per-step penalty was -18.65 and dying early minimized cumulative punishment. Histogram spike at 61 steps = LINE LOST firing like clockwork.

#### Run 2 (Post-fix: reward rebalancing + tolerance increase): 271K steps, 8 hours
Fixes applied:
- `r_smoothness`: 2.0 → 0.5
- `r_centering` off-line: -2.0 → -0.5
- `r_loiter`: -20.0 → -5.0
- `r_time`: removed entirely
- `r_progress`: multiplier 100 → 200
- LINE LOST: 1.5s → 10s
- STUCK: 5s → 8s, LOITER: 8s → 12s
- Learning rate: 3e-4, gradient_steps: 2 → 1

```
Episodes:     1,190      |  Throughput:  9.1 steps/s
Mean Reward:  -814.2     |  Best:        +1,601.6
Mean Length:  228 steps   |  Median:      181 steps
Reward/Step:  -6.21      |  Trend:       📈 +65.6
Corr (r,l):   +0.651
```
**Major improvements:** 3× longer episodes, 6.5× better peak reward, positive trend, correct correlation (longer = better). Throughput nearly doubled.

#### Run 3 (Extended training): +271K steps (540K total), 6 hours
```
Episodes:     756        |  Throughput:  12.3 steps/s
Mean Reward:  -697.2     |  Best:        +3,757.3
Mean Length:  359 steps   |  Median:      241 steps
Reward/Step:  -3.77      |  Trend:       📉 -240.7
Corr (r,l):   +0.819
```
**Peak performance reached:** +3,757 best episode, 2,221-step max episode (agent drove ~10-15% of the track). BUT the trend turned negative — the agent peaked around episode 200-400 then regressed. **Catastrophic forgetting** due to:
- 50K replay buffer cycling every ~67 minutes
- Good experiences overwritten before they could be consolidated
- Learning rate 3e-4 causing policy to overreact to noisy batches

---

## Paper Analysis

We studied two Sony research papers to identify improvements:

### Paper 1: "Vision-Based Super-Human Racing" (RLC 2024)
**First vision-based agent to outperform all human drivers in Gran Turismo 7.**

Key architecture:
- **Observation:** 64×64 raw RGB (no masks, no preprocessing) + 17-dim proprioception (velocity, acceleration, angular velocity, steering history)
- **Asymmetric actor-critic:** Policy uses only local features (image + proprio). Critic uses global features (531-dim course points — track shape 6s ahead) during training only.
- **CNN:** 4 layers (64→128→256→512) with stride-2 4×4 kernels → FC128 → 4×2048 MLP
- **Algorithm:** QR-SAC (distributional, 7-step returns, 32 quantiles)

Key hyperparameters:
```
learning_rate:   2.5e-5  (12× lower than our v1)
entropy:         0.01    (fixed, not auto-tuned)
buffer_size:     2,500,000  (50× larger than our v1)
batch_size:      512
training_epochs: 2000-4000 (each = 6000 gradient steps)
```

Reward function:
```
r = r_progress                        # PRIMARY: meters along centerline
  + 10 × r_off_course                 # Penalty proportional to speed when off-track
  + 10 × r_wall                       # Penalty proportional to speed² on wall contact
  +  3 × r_steering_change            # -|Δθ|
  +  5 × r_steering_history           # Penalizes zig-zagging over 3 steps
```

Key insight: **Progress is EVERYTHING.** One positive signal, everything else is a penalty. The reward function doesn't try to teach the agent HOW to drive (centering, turning, speed bonuses) — it only says "go fast, don't cheat."

### Paper 2: "GT Sophy" (Nature, 2022)
**Champion-level racing agent — superhuman in time trial AND head-to-head racing.**

Key differences from Paper 1:
- **No vision** — uses only telemetry (velocity, acceleration, etc.) + 531-dim course points
- **Full racing scenarios** — time trial + 4v4 racing with opponent modeling
- **Training scale:** 10-20 PlayStations simultaneously, GPU server for async gradient updates
- **QR-SAC with 7-step returns**, 2048×4 MLP networks

Reward function:
```
r = R_progress                        # Meters along centerline (masked when off-course)
  + R_off_course                      # -Δoff_time × speed²
  + R_wall                            # -Δwall_time × speed²
  + R_tyre_slip                       # Penalty for tyre slip angle
  + R_passing                         # Bonus for overtaking
  + R_collision                       # Various collision penalties
```

Key insight from ablations:
- **QR-SAC >> vanilla SAC** — distributional RL was critical for performance
- **Course points repr >> wall lidar** — representing the track as point sequences was far superior
- **Off-course penalty is essential** — without it, the agent cuts corners

### What we CAN'T replicate (hardware limits)

| Paper Feature | Why We Can't |
|---|---|---|
| Asymmetric actor-critic | No course point data from PPSSPP |
| QR-SAC | SB3 doesn't implement it |
| 2.5M replay buffer | 16GB RAM constraint |
| 20+ PlayStation distributed | Single laptop |
| Synchronous simulator | PPSSPP is async, we screen-capture |
| Full telemetry (17-dim) | Only OCR speed + minimap |
| 12-24M training steps | Would take 23+ days at 6 FPS |

---

## v2: Paper-Aligned Overhaul

Branch: `v2-paper-aligned`

### Design Philosophy
"Let the CNN learn everything from raw pixels, like Paper 1."

### Changes from v1

| Component | v1 | v2 |
|---|---|---|
| Observation | 84×84 masks (line/road/brake) | **64×64 raw RGB** (road area crop) |
| Vision pipeline | 9 passes per step, HSV/contours/distTransform | **3 calls** (frame + collision + progress) |
| Aux vector | 4 dims (speed, progress, brake, collision) | **8 dims** (speed, progress, steer×3, gas×3) |
| Reward | 7 components competing | **3 components** (progress + collision + steer) |
| CNN | 32→64→64→FC256 | **64→128→256→512→FC128** |
| Learning rate | 3e-4 | **3e-5** |
| Entropy | auto_0.1 | **0.01 fixed** |
| Frame stack | 4 | **2** |
| Buffer | 50K | **100K** |
| Terminations | LINE LOST + STUCK + LOITER | **STUCK + LOITER only** |

---

## v2 Training Results

### Run (10 hours, 241K steps)
```
Episodes:     628        |  Throughput:  6.1 steps/s
Mean Reward:  -473.8     |  Best:        -377.7 (NEVER POSITIVE)
Mean Length:  385 steps   |  Median:      211 steps
Reward/Step:  -2.14      |  Trend:       📉 -16.7
Corr (r,l):   -0.700     |  (INVERTED — longer = worse!)
```

### Observed Behavior
- Agent moves forward but **brute-forces through grass/off-track** — not steering, not following the road
- Goes wrong direction with no termination signal to stop it
- Longer episodes accumulate more steering penalties without enough progress reward to compensate
- **CNN hasn't learned any useful visual features** in 241K steps — raw RGB needs millions of steps

---

## Takeaways

### What works (keep from v1):
1. **Hand-crafted mask observations** — In our compute-limited setting, pre-computing line/road/brake channels is NOT premature optimization — it's necessary. The CNN should focus on learning POLICY, not feature detection.
2. **Line detection as observation channel** — The distance-transform gradient gives a powerful, smooth steering signal. Raw RGB at 64×64 can't communicate this effectively in <1M steps.

### What works (keep from v2):
1. **Lower learning rate (3e-5)** — Papers are emphatic: 2.5e-5. Our v1 at 3e-4 caused catastrophic forgetting.
2. **Fixed entropy (0.01)** — More stable than auto-tuning with noisy visual inputs.
3. **Progress-dominated reward** — 7 competing signals was too noisy. Progress + penalties only.
4. **Bigger buffer (100K)** — Retains good experiences 2× longer.
5. **Steering/gas history in aux** — Proven proprioception from papers.
6. **Frame stack 2** — Sufficient temporal context, half the memory of stack-4.

### What didn't work:
1. **Raw RGB observation** — Needs millions of steps to learn features. We can't afford it.
2. **Bigger CNN (64→128→256→512)** — Halved throughput (12→6 FPS). The 3050 can't handle it.
3. **Removing LINE LOST** — Without off-track detection, agent drives backward/off-track indefinitely.
4. **Removing all centering signal** — Progress from minimap is too noisy/coarse to teach fine steering.

---

## v3: Hybrid Approach (Proposed)

Branch: `v3-hybrid` (to be created)

### Design Philosophy
"Pre-compute what's cheap and informative (vision masks). Let the CNN focus on policy learning, not feature detection. Use paper-aligned hyperparameters for training stability."

### Proposed Architecture

**Observation:**
- `frame`: 64×64×3 uint8 — v1-style mask channels:
  - Ch 0: Line distance-transform gradient (steering signal)
  - Ch 1: Road surface mask (drivable area)
  - Ch 2: Brake zone overlay
- `aux`: 8 dims — v2-style proprioception:
  - `[speed_norm, progress, steer_t-1, steer_t-2, steer_t-3, gas_t-1, gas_t-2, gas_t-3]`

**Reward (3 components, progress-dominated):**
```python
r_progress  = Δprogress × 500.0           # PRIMARY: go forward around the track
r_collision = -max(10, speed² × 0.01)      # Wall hits, proportional to speed
r_steer     = -|Δsteer| × 1.0             # Smooth driving (Paper 1: r_s penalty)
```

**CNN:** v1 size (32→64→64→FC256) — proven 12+ FPS on the 3050.

**Hyperparameters (from v2/papers):**
```
learning_rate:   3e-5     (paper-aligned)
entropy:         0.01     (fixed)
buffer_size:     100,000  (2× v1, fits in 16GB with 64×64 obs)
frame_stack:     2
gradient_steps:  1
batch_size:      256
```

**Terminations:**
- STUCK: 10s with displacement < 1.0 (v2 threshold)
- LOITER: 15s with OCR speed < 5 km/h (v2 threshold)
- WRONG DIRECTION: 5s of negative progress delta (NEW — replaces LINE LOST)
- 10s grace period at episode start

### Expected Improvements over v1 and v2

| Metric | v1 (best) | v2 | v3 (expected) |
|---|---|---|---|
| Throughput | 12.3 steps/s | 6.1 | **12+** (v1 CNN) |
| Peak reward | +3,757 | -377 | **Higher** (stable LR) |
| Forgetting | Severe after 400 eps | Severe | **Mitigated** (100K buffer + low LR) |
| Episode length | 359 mean | 385 mean | **Longer** (wrong-dir term prevents wasted time) |
| Off-track handling | LINE LOST (brittle) | None (agent wanders) | **Wrong-direction** (progress-based) |

### Why this should work
1. **v1's observation + v2's training config** is the key combination we haven't tried
2. The v1 catastrophic forgetting was caused by `lr=3e-4` and `buffer=50K`, NOT the observation design
3. With `lr=3e-5` and `buffer=100K`, the v1 peak of +3,757 should be sustained rather than forgotten
4. Steering history in aux gives the CNN temporal context that v1 lacked

---

## v3: Training Results & Debugging

Branch: `v3-hybrid`

### Run 1: Initial v3 (284 episodes, 460K steps, ~6 hours)

```
Episodes:     284        |  Throughput:  19.9 steps/s (median)
Mean Reward:  +204.7     |  Best:        +616.8
Mean Length:  1619 steps  |  Median:      782 steps
Reward/Step:  +0.50      |  Trend:       📉 -97.1
Corr (r,l):   -0.950     |  ⚠️ FARMING DETECTED
```

**Critical finding:** `short eps earn 0.99/step vs -0.11/step for long eps`.

**Root cause chain — 4 bugs discovered:**

#### Bug 1: Continuous progress jitter is not zero-mean
The original reward: `progress_delta × 1500`, smoothed over 20 frames. We assumed minimap jitter was zero-mean and would cancel. It didn't — the signal had a slight positive bias (~+0.05/step), which competed with the -0.3 time penalty.

**Fix:** Replaced continuous progress with **discrete waypoint system** — 50 checkpoints, +30 reward once per crossing. Jitter can trigger at most 1 false crossing = +30 total.

#### Bug 2: `map_center` in wrong coordinate space
```python
# Was frame-space (y=201), but get_map_position returns ROI-relative (y=0-66):
self.map_center = (39.0, 201.0)  # BUG: dy always ~= -160

# Fixed to ROI-relative:
self.map_center = (39.0, 41.0)   # 201 - 160 = 41
```
**Impact:** ALL progress readings were clustered around 22% regardless of actual car position. The continuous progress reward was essentially random noise — the agent couldn't learn direction from it.

#### Bug 3: Polar angle direction inverted
`atan2(dy, dx)` increases clockwise in screen coordinates (Y-down), but the track goes anti-clockwise. Progress DECREASED when driving forward.

**Fix:** `progress = (-angle + π) / (2π)` — negated the angle.

#### Bug 4: Polar angle non-monotonic on non-circular tracks
Even with bugs 2-3 fixed, the polar angle **backtracks** on sections of the track where the road doubles back in angular space. Some sections covered 30% of angular range in a few seconds, then the angle dropped back. This caused `max_waypoint_idx` to spike ahead, creating dead zones where no rewards could be earned.

```
step  800: progress=27%, max_wp=16
           → SPIKE to wp=36 in between frames (angular non-uniformity)
step  900: progress=35%, wp=17, max_wp=36 ← gap of 19, never catches up
step 1700: progress=55%, wp=27, max_wp=36 ← STILL 9 behind, no rewards for 900 steps
```

**Conclusion:** Polar angle progress is fundamentally broken for non-circular tracks.

### Solution: Calibrated Track Path

Replaced polar angle entirely with **path-based progress** from a manually recorded calibration lap.

#### Calibration Tool (`calibrate_track.py`)
1. Live minimap + red mask display with HSV tuning trackbars
2. Temporal outlier rejection (>5px jump = noise, reset after 20 misses)
3. Records car dot positions → generates 50 equally-spaced waypoints along the actual path
4. Saves `track_waypoints.npy` + auto-updates `vision.py` thresholds
5. Keys: S=start recording, R=clear path, Q=save+quit

#### Red Dot Detection Overhaul
The original `get_map_position` used `np.mean(ALL_red_pixels)` — background noise (audience red, terrain) dragged the centroid everywhere.

New pipeline:
```
1. HSV red mask (tunable S_min, V_min thresholds)
2. Contour detection → filter by area (1-80 px²)
3. Track-path validation: reject if >6px from nearest calibrated waypoint  ← KEY
4. Temporal filter: reject >8px jumps, re-acquire after 20 consecutive misses
5. Pick largest valid contour centroid
```

**Track-path validation** is the most important layer — even if the audience has red pixels, they're nowhere near the minimap track path, so they get rejected immediately.

#### Progress System (current)
```python
# In get_progress_percent():
nearest_idx = argmin(||track_waypoints - current_pos||)
progress = nearest_idx / 50

# In step() reward:
current_wp = int(progress * 50) % 50
fwd_dist = (current_wp - max_wp) % 50
if 0 < fwd_dist <= 25:  # Forward up to half-track
    capped = min(fwd_dist, 5)
    r_progress = capped * 30.0
    max_wp = current_wp
```

#### Reward Function (current)
```python
r_progress = 0-150/step    # +30 per new waypoint crossed (capped at 5/step)
r_time     = -0.1/step     # Constant time penalty
r_steer    = -|Δsteer|×0.3 # Steering smoothness
```

Expected totals per episode:
| Scenario | Waypoints | Lap | Time | Steer | Total |
|---|---|---|---|---|---|
| Full lap (~2400 steps) | +1500 | +1000 | -240 | -288 | **+1972** |
| Half lap (~1200 steps) | +750 | 0 | -120 | -144 | **+486** |
| Standing still (5000 steps) | ~0 | 0 | -500 | 0 | **-500** |

### Anti-Farming History

| Iteration | Exploit Found | Fix Applied |
|---|---|---|
| v1 original | Blue SUBARU banner = line detected | Gate centering on displacement |
| v3 continuous | Minimap jitter ≠ zero-mean, +0.05/step | Discrete waypoints |
| v3 time penalty -0.3 | Short eps still earn 0.99/step | Fix map_center coordinates |
| v3 polar angle | Track backtracks in angular space | Calibrated path-based progress |
| v3 audience noise | Red audience pixels → fake waypoints | Track-path validation (>6px = reject) |

### Current Status (April 19, 2025)

- **Progress tracking:** Path-based with calibrated waypoints ✅
- **Anti-farming:** Track-path validation + discrete one-shot rewards ✅
- **Training:** Starting fresh overnight run
- **Next priority:** Observe if agent discovers forward driving with clean progress signal. If successful, re-enable collision detection.
