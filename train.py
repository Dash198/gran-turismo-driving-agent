import glob
import os
import re

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from custom_policy import policy_kwargs
from gt_env import GranTurismoEnv

# --- CONFIGURATION ---
TOTAL_TIMESTEPS = 5_000_000
LOG_DIR = "./logs/PPO/"
MODEL_DIR = "./models/PPO/"
MODEL_PREFIX = "gtr_PPO"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def make_env():
    """
    Factory function for the environment.
    """
    env = GranTurismoEnv(camera_index=2)
    env = Monitor(env, LOG_DIR)
    return env


def get_latest_checkpoint(path, prefix):
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
    env = VecFrameStack(env, n_stack=4, channels_order="last")

    print(">>> 2. Checking for Saved Models...")
    latest_path, steps_so_far = get_latest_checkpoint(MODEL_DIR, MODEL_PREFIX)

    if latest_path:
        print(f"💾 FOUND SAVE: '{latest_path}' (Step {steps_so_far})")
        print("⚡ Loading Model...")
        model = PPO.load(latest_path, env=env, policy_kwargs=policy_kwargs)

        print("🧹 Cleaning up old checkpoints...")
        for f in glob.glob(f"{MODEL_DIR}/{MODEL_PREFIX}_*.zip"):
            if f != latest_path:
                try:
                    os.remove(f)
                    print(f"   Deleted old: {f}")
                except OSError as e:
                    print(f"   Error deleting {f}: {e}")

    else:
        print("✨ No save found. Creating NEW Agent from scratch...")
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            learning_rate=0.0003,
            policy_kwargs=policy_kwargs,
        )
        steps_so_far = 0

    checkpoint_callback = CheckpointCallback(
        save_freq=5000, save_path=MODEL_DIR, name_prefix=MODEL_PREFIX
    )

    print(f">>> 3. STARTING TRAINING (Resuming from {steps_so_far})...")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name="PPO",
        )
        print(">>> Training Finished!")
        model.save(f"{MODEL_DIR}/gtr_final_model")

    except KeyboardInterrupt:
        print("\n>>> Training Paused. Saving emergency backup...")
        model.save(f"{MODEL_DIR}/{MODEL_PREFIX}_interrupted")
    finally:
        env.close()


if __name__ == "__main__":
    main()
