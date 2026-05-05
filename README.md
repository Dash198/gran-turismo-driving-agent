# Gran Turismo Driving Agent 🏎️

A **reinforcement learning agent** that learns to drive a full lap in Gran Turismo using only raw visual input — no game API, no telemetry, no modding. Pure computer vision + SAC.

> Runs on Linux/Wayland. Uses **Gran Turismo 4 (emulator: PPSSPP)**.

---

## How It Works

The agent perceives the game through a v4l2loopback virtual camera fed by PPSSPP and controls the emulator via a virtual controller (evdev/uinput). No game internals are accessed.

```
PPSSPP (GT PSP) ──v4l2loopback──► /dev/video2 ──► Linux PC
                                                      │
                                            ┌─────────▼─────────┐
                                            │   vision.py        │
                                            │  (OpenCV pipeline) │
                                            └─────────┬─────────┘
                                                      │ obs (64×64 masks + aux)
                                            ┌─────────▼─────────┐
                                            │   SAC Agent        │
                                            │ (stable-baselines3)│
                                            └─────────┬─────────┘
                                                      │ action (steer, gas/brake)
                                            ┌─────────▼─────────┐
                                            │ virtual_controller │
                                            │    (evdev/uinput)  │
                                            └───────────────────┘
```

### Observation Space

Each step produces a `Dict` observation:

| Key | Shape | Description |
|-----|-------|-------------|
| `frame` | `(64, 64, 3)` | Binary masks: racing line, road surface, brake zones |
| `aux` | `(8,)` | Speed, lap progress, last 3 steering + gas actions |

A `VecFrameStack(n=2)` doubles these to `(64,64,6)` and `(16,)` for motion information.

### Reward Structure

| Source | Value |
|--------|-------|
| Waypoint crossed (451 dense) | `+3.3 × waypoints_crossed` |
| Time penalty | `-0.02 / step` |
| Steering smoothness | `-0.3 × |Δsteer|` |
| Lap completion | `+1000` |
| Stagnation (no waypoint in 15s) | `-50` |

### Termination Conditions

- **Stagnation** — no waypoint progress for 300 steps (~15s). Covers stuck-at-wall and off-track.
- **Lap Completed** — OCR detects lap counter advance from 1 → 2.
- **Max Steps** — hard cap at 5000 steps (~4 min).

---

## Project Structure

```
gran-turismo-driving-agent/
├── gt_env.py              # Gymnasium environment (observation, reward, termination)
├── vision.py              # OpenCV pipeline: masks, minimap tracking, lap OCR
├── virtual_controller.py  # evdev/uinput PS5 controller emulation
├── custom_policy.py       # CNN + MLP feature extractor for SAC
├── sac_train.py           # Main training script (auto-resumes from checkpoints)
├── calibrate_track.py     # Drive manually to record minimap waypoints → track_path.npy
├── reward_test.py         # Manual test dashboard: drive and verify reward/OCR live
├── analysis.py            # Post-training stats and plots from Monitor logs
├── find_rois.py           # Helper to locate HUD ROIs (lap counter, speed, minimap)
├── track_path.npy         # 451-point calibrated minimap waypoint path
└── requirements.txt
```

**Debug/diagnostic tools** (not needed for training):
- `debug_vision.py` — visualise the 3 vision channels live
- `obs_preview.py` — see exactly what the agent sees
- `hsv_diag.py` / `tune_hsv.py` / `tune_red.py` — tune colour masks interactively
- `debug_ocr.py` / `ocr_diag.py` — test Tesseract lap/speed reads
- `frame_inspect.py` — inspect raw capture frames
- `check.py` / `debug_stuck.py` — sanity-check termination logic

---

## Setup

### Requirements

- Linux with **uinput** support (`sudo modprobe uinput`)
- **PPSSPP** emulator (any recent build with display output)
- **v4l2loopback** kernel module to create a virtual camera from PPSSPP's window
- **Tesseract OCR** (`sudo pacman -S tesseract` or `sudo apt install tesseract-ocr`)
- Gran Turismo (PSP) ROM
- The **Racing Line** assist turned on in-game (the blue line is the primary observation)

### Install

```bash
# Clone
git clone https://github.com/yourname/gran-turismo-driving-agent
cd gran-turismo-driving-agent

# Create venv and install (uv recommended)
uv venv && uv pip install -r requirements.txt

# Or with pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Virtual Camera Setup (v4l2loopback)

The agent reads the game from `/dev/video2`. Set this up before launching PPSSPP:

```bash
# Load the virtual camera module
sudo modprobe v4l2loopback video_nr=2 card_label="GT_Cam" exclusive_caps=1
```

Then in PPSSPP, enable display output to the v4l2loopback device (or use any screen-capture-to-v4l2 tool like `ffmpeg` or `obs-v4l2sink`).

Verify the camera is visible:
```bash
v4l2-ctl --list-devices
```

The agent uses a point-based minimap path to track lap progress. A pre-calibrated `track_path.npy` is included for the default track (Deep Forest Raceway).To calibrate a new track:

```bash
uv run calibrate_track.py
```

Drive one full lap manually. The script records the red dot's path on the minimap and saves `track_path.npy`.

### Verify Vision & Rewards

Before training, confirm the vision pipeline is working:

```bash
uv run reward_test.py
```

Drive manually. The dashboard shows real-time vision channels, waypoint rewards, lap OCR, and stagnation counters. Everything should light up correctly before you start training.

### Train

```bash
uv run sac_train.py
```

- **Auto-resumes** from the latest checkpoint in `models/SAC/`.
- Saves checkpoints every 5000 steps.
- Ctrl+C saves `gtr_SAC_interrupted.zip` and exits cleanly.
- Monitor training with TensorBoard: `tensorboard --logdir logs/SAC`

To start fresh (wipe weights + replay buffer):

```bash
rm -rf models/SAC logs/SAC
uv run sac_train.py
```

To keep weights but reset the replay buffer (e.g. after reward changes):

```bash
rm -f models/SAC/*_replay_buffer.pkl
uv run sac_train.py
```

### Analyse a Run

```bash
uv run analysis.py
```

Generates a 6-panel dashboard (reward trends, episode length, farming check, throughput) from the Monitor logs.

---

## Architecture

### Vision Pipeline (`vision.py`)

Raw 640×480 frames are cropped to the game area (capture card has black bars) then processed into three 64×64 binary masks:

| Channel | Detection Method |
|---------|-----------------|
| **Racing Line** | HSV mask for the blue/red GT racing line overlay |
| **Road Surface** | HSV mask for grey asphalt |
| **Brake Zones** | HSV mask for red braking-zone markers |

Minimap tracking uses red-dot contour detection with temporal filtering (rejects teleports > 15px/frame) and validates detections against the calibrated track path.

Lap detection runs Tesseract OCR on the HUD lap counter every 30 steps, with 3-read consistency checking and multiple binary thresholds (200/160/128) for robustness.

### Policy Network (`custom_policy.py`)

```
frame (64×64×6) ──► CNN ──────────────────────► 256-dim features ──► SAC Actor/Critic
                  (3 conv)                  ▲
aux (16,) ────────► MLP (16→64) ───────────┘
```

Standard Nature-DQN CNN architecture (8×8/4, 4×4/2, 3×3/1) proven fast on a 3050 GPU (~22 steps/s).

### SAC Hyperparameters

| Param | Value | Rationale |
|-------|-------|-----------|
| `buffer_size` | 100,000 | 2× default — retains more diverse experiences |
| `learning_starts` | 5,000 | Fill buffer before training begins |
| `batch_size` | 256 | Standard |
| `learning_rate` | 1e-4 | Faster convergence for our smaller setup |
| `gamma` | 0.99 | ~100-step effective horizon |
| `ent_coef` | auto | Adaptive entropy for exploration |
| `n_stack` | 2 | Frame stacking for motion sensing |

---

## Training Notes

### What Works
- Racing line channel is the strongest signal — the agent learns to follow it visually.
- 451-waypoint dense reward gives continuous gradient for forward progress.
- Stagnation-only termination (15s) eliminates the "crash early" local minimum without false positives from displacement misfires.

### Known Challenges
- Minimap tracking can lose the red dot at high-contrast track sections (tight corners, reflective surfaces). Temporal filtering helps but isn't perfect.
- Lap OCR occasionally misreads "2" as "3" — handled by accepting any consistent non-"1" read.
- Ghost car overlay appears at lap 2 start; episode terminates immediately on lap completion to avoid training on corrupted frames.

### Design Decisions
- **No collision detection in reward** — OpenCV spark detection was unreliable. Stagnation handles wall-crashing naturally.
- **No speed gate for lap detection** — speed OCR was too noisy to be a reliable gate.
- **Fixed-step thresholds** — FPS varies significantly during OCR calls; all timeouts are in steps, not seconds, to prevent premature kills on slow frames.

---

## Results

After ~2M steps of SAC training:
- Agent consistently follows the racing line through straights and wide corners.
- Successfully completed full laps during testing, although rarely.
- Best episode reward: **+1167** (equivalent to ~350 waypoints ≈ 78% of track).

---

## Acknowledgements

Inspired by prior work on real-world RL for racing:
- *Outracing Champion Gran Turismo Drivers with Deep Reinforcement Learning* (Wurman et al., 2022)
- *Learning to Drive in a Day* (Kendall et al., 2019)
