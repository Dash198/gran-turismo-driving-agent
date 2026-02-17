import argparse
import glob
import os
import re

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from gt_env import GranTurismoEnv

# --- CONFIGURATION ---
TOTAL_TIMESTEPS = 5_000_000
LOG_DIR = "./logs/SAC/"
MODEL_DIR = "./models/SAC/"
MODEL_PREFIX = "gtr_SAC"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def make_env():
    """Factory function for the environment."""
    # Ensure camera_index=2 matches your ffmpeg/v4l2loopback setup
    env = GranTurismoEnv(camera_index=2)
    env = Monitor(env, LOG_DIR)
    return env


def get_latest_checkpoint(path, prefix):
    """
    Finds gtr_SAC_interrupted.zip first, then falls back to highest number.
    """
    # 1. Prioritize manual interruption save
    interrupted_path = os.path.join(path, f"{prefix}_interrupted.zip")
    if os.path.exists(interrupted_path):
        return interrupted_path, 0

    # 2. Fallback to highest numbered checkpoint
    files = glob.glob(f"{path}/{prefix}_*.zip")
    if not files:
        return None, 0

    highest_step = 0
    latest_file = None
    for f in files:
        match = re.search(rf"{prefix}_(\d+)\.zip", f)
        if match:
            step = int(match.group(1))
            if step > highest_step:
                highest_step = step
                latest_file = f

    return latest_file, highest_step


def main():
    print(">>> 1. Initializing Environment...")
    env = DummyVecEnv([make_env])
    # Stack 4 frames to give SAC temporal context (seeing speed/direction)
    env = VecFrameStack(env, n_stack=4, channels_order="last")

    print(">>> 2. Checking for Saved Models...")
    latest_path, steps_so_far = get_latest_checkpoint(MODEL_DIR, MODEL_PREFIX)

    if latest_path:
        print(f"💾 LOADING MODEL: '{latest_path}'")
        model = SAC.load(latest_path, env=env)

        # CLEANUP: Empty the dir except for the loaded model [cite: 11-02-2026]
        print("🧹 Cleaning up old checkpoints...")
        for f in glob.glob(f"{MODEL_DIR}/*.zip"):
            if os.path.abspath(f) != os.path.abspath(latest_path):
                os.remove(f)
    else:
        print("✨ No save found. Starting FRESH SAC Agent...")
        # REVIVAL SAC CONFIG [cite: 11-02-2026, 19-01-2026]
        model = SAC(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            buffer_size=100000,  # Adjusted for 6GB VRAM [cite: 11-02-2026]
            learning_starts=10000,  # EXTENDED WARMUP: High-entropy exploration [cite: 11-02-2026]
            batch_size=256,
            learning_rate=3e-4,
            train_freq=4,  # Stable updates for 200+ FPS [cite: 11-02-2026]
            gradient_steps=1,
            # --- INCENTIVE ADJUSTMENTS ---
            # Gamma: High value (0.999) ensures the agent 'sees' the 10k finish line
            # from much further back on the track [cite: 19-01-2026, 11-02-2026]
            gamma=0.999,
            # Entropy: 'auto_0.1' ensures it stays curious even after finding
            # a 'safe' path, preventing early convergence on slow driving [cite: 19-01-2026]
            ent_coef="auto_0.1",
        )

    # Save every 5000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=5000, save_path=MODEL_DIR, name_prefix=MODEL_PREFIX
    )

    print(">>> 3. STARTING OVERNIGHT TRAINING...")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name="SAC_Run",
        )
    except KeyboardInterrupt:
        print(f"\n⏸️ Saving {MODEL_PREFIX}_interrupted.zip...")
        model.save(f"{MODEL_DIR}/{MODEL_PREFIX}_interrupted")
    finally:
        env.close()


if __name__ == "__main__":
    main()
