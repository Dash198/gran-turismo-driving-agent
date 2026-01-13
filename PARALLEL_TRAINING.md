# Parallel Training with Xvfb

This guide explains how to scale the Gran Turismo agent to multiple parallel instances using **Xvfb (X Virtual Framebuffer)**.

## Why Xvfb?
- **Headless**: Runs games without a physical monitor (or interfering windows).
- **Isolated**: Each instance gets its own `$DISPLAY`, preventing input conflicts.
- **Scalable**: You can spawn as many instances as your RAM/GPU allows.

## Prerequisites
Install Xvfb:
```bash
sudo apt-get install xvfb
```

## Running a Single Instance Headlessly

To run one instance on a virtual display `:99`:

```bash
# 1. Start Xvfb on display :99 with specific resolution
Xvfb :99 -screen 0 640x480x24 &

# 2. Export DISPLAY env var so apps use it
export DISPLAY=:99

# 3. Start PPSSPP (it will render to the virtual framebuffer)
PPSSPPQt &

# 4. Run the Agent (it connects to :99 automatically via our window_interface)
uv run python Phase_1/main.py
```

## Running Multiple Instances (Parallel)

To run `N` instances, you wrap the above in a loop, assigning a unique display ID (`:101`, `:102`, etc.) to each pair of (PPSSPP + Agent).

### `launch_parallel.sh` (Example)

```bash
#!/bin/bash

BASE_DISPLAY=100
NUM_INSTANCES=4

for i in $(seq 1 $NUM_INSTANCES); do
    DISPLAY_ID=$((BASE_DISPLAY + i))
    
    echo "Starting Instance $i on :$DISPLAY_ID"
    
    # 1. Start Xvfb
    Xvfb :$DISPLAY_ID -screen 0 640x480x24 &
    XVFB_PID=$!
    
    # 2. Launch PPSSPP and Agent in this context
    (
        export DISPLAY=:$DISPLAY_ID
        
        # Start Game
        PPSSPPQt "Gran Turismo.iso" &
        GAME_PID=$!
        sleep 5 # Wait for load
        
        # Start Agent
        uv run python Phase_1/main.py --instance $i &
        AGENT_PID=$!
        
        # Wait for agent (or game) to exit
        wait $AGENT_PID
    ) &
done

wait
```

## Code Adjustments Needed

1.  **Remove Low Profile**: In `main.py`, remove `window.set_low_profile()` when running in Xvfb, as there is no desktop environment to clutter.
2.  **Resource Management**: Ensure you kill the Xvfb processes when training stops.

## Verification
You can "watch" what's happening on a virtual display using VNC or `xwd`:

```bash
# Dump screenshot of virtual display :101
DISPLAY=:101 xwd -root -out debug.xwd
convert debug.xwd debug.png
```
