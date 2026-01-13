# Initial Sketch

## The Idea

First Attempt: Discrete Action Space

9 actions possible, (3 for throttle/brake/neutral) x (3 for left/right/neutral)

Grab SS $\rightarrow$ Pass to CNN $\rightarrow$ Use DQN $\rightarrow$ Evaluate $\rightarrow$ Repeat

## The CNN

First, trying ResNet 18 for perception.

1. Input: 3-4 stacked frames, downsampled and resized.

2. Processing: Detect edges, shapes, racing lines, get embeds

3. Output: Get some dimensions ig to feed to the RL head.\

# The Time Trial Model 1.0

## Setting
- **No opponents present** -> no blocking, overtaking, collisions.
- **Goal** -> Minimize lap time, stay close to the driving line, avoid off-road or spinouts
- **Available info** -> Full frame with driving line.

## What the Model Should Learn

1. **Track awareness** -> Knowing where the road is and how it curves -> to avoid going off-road
2. **Driving Line Tracking** -> Keeping car aligned with the driving line -> for speed and corner efficiency
3. **Speed control** -> Slowing down for corners, accelerating out -> for stability
4. **Steering smoothness** -> Avoiding jerky turns -> for better lap times
5. **Edge recovery** -> Return to line if drifted off -> Robustness

## Initial Mode Design

1. **ResNet for Perception**
    - Input: stacked screenshots (4 consecutive frames)
    - Output: 512-D embedding (hopefully some meaningful info)

2. **Policy Head (RL)**
    - Small MLP taking the embedding as input.
    - Outputs discrete actions: `[steering, throttle]`
    - Steering = +1 for right and -1 for left, 0 for neutral.
    - Throttle = +1 for throttle and -1 for brake, 0 for neutral

3. **Reward Shaping**
    Yet to decide.


# Model 1: NitroGen-based Agent
This directory (`Model1`) contains all work related to the Neural Time Trial agent based on NVIDIA's NitroGen foundation model.

## Structure
- `Phase_1/`: Baselines, Calibration, and Naive Integration (Completed).
- `Phase_2/`: Residual Learning (Active).

## Phase 1: Naive Integration (Completed)
We validated that NitroGen can drive the car using a "Discrete Mapping" wrapper. 
- **Performance**: ~2.5 min lap time (Conservative).
- **Issues**: Slow speeds due to braking priority and oscillation.
- **Fix**: Implemented Trajectory Averaging (0.3s window) and Aggressive Throttle Threshold (0.3).

## Phase 2: Residual Learning (Current)
Instead of fine-tuning NitroGen (which destroys priors), we train a lightweight **Residual Policy** (PPO) to correct its actions.

### Architecture
- **Base**: Frozen NitroGen (bfloat16).
- **Residual**: Tiny CNN (NatureCNN).
- **Action**: $A_{final} = A_{base} + \Delta_{residual}$
- **Objective**: Minimize Lap Time while keeping $||\Delta_{residual}||$ low to preserve safety.