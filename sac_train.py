"""
SAC Training Script v2 — Paper-Aligned
───────────────────────────────────────
Key hyperparameter changes from papers:
  - Learning rate: 3e-4 → 3e-5 (12× reduction, matching Sony papers)
  - Entropy coeff: auto_0.1 → 0.01 fixed (matching papers)
  - Frame stack: 4 → 2 (halves memory, enables bigger buffer)
  - Buffer: 50K → 100K (retains good experiences longer)
  - Gradient steps: 1 (match data collection rate)
"""

import glob
import os
import re

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from custom_policy import policy_kwargs
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
    env = GranTurismoEnv(camera_index=2)
    env = Monitor(env, LOG_DIR)
    return env


def get_latest_checkpoint(path, prefix):
    """Finds gtr_SAC_interrupted.zip first, then falls back to highest number."""
    interrupted_path = os.path.join(path, f"{prefix}_interrupted.zip")
    if os.path.exists(interrupted_path):
        return interrupted_path, 0

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
    env = VecFrameStack(env, n_stack=2, channels_order="last")  # Was 4, now 2

    print(">>> 2. Checking for Saved Models...")
    latest_path, steps_so_far = get_latest_checkpoint(MODEL_DIR, MODEL_PREFIX)

    if latest_path:
        print(f"💾 LOADING MODEL: '{latest_path}'")
        model = SAC.load(latest_path, env=env, policy_kwargs=policy_kwargs)

        print("🧹 Cleaning up old checkpoints...")
        for f in glob.glob(f"{MODEL_DIR}/{MODEL_PREFIX}_*.zip"):
            if os.path.abspath(f) != os.path.abspath(latest_path):
                os.remove(f)
    else:
        print("✨ No save found. Starting FRESH v2 Agent...")
        model = SAC(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            buffer_size=100_000,         # Was 50K — 2× more memory for experiences
            learning_starts=5000,
            batch_size=256,
            learning_rate=3e-5,          # Was 3e-4 — papers use 2.5e-5
            train_freq=1,
            gradient_steps=1,
            gamma=0.99,
            ent_coef=0.01,              # Was "auto_0.1" — papers use fixed 0.01
            policy_kwargs=policy_kwargs,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000, save_path=MODEL_DIR, name_prefix=MODEL_PREFIX
    )

    print(">>> 3. STARTING TRAINING (v2 Paper-Aligned)...")
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
