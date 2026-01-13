
import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation

# Imports
import sys
# Add Model1/Phase_1 for environment.py
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Phase_1"))
from environment import GranTurismoEnv

# Import ResidualWrapper
from residual_env import ResidualEnv

# Config
NUM_ENVS = 1 # Start with 1 to test stability (NitroGen uses VRAM)
TOTAL_STEPS = 100000

CURRICULUM_CARS = [
    # 1. Low Power (Smoothness)
    ("Suzuki", "Suzuki Cappuccino (EA21R) '95"),
    ("Mazda", "Mazda MX-5 Miata 1.8 RS (NB, J) '98"),
    
    # 2. FF (Understeer)
    ("Honda", "Honda CIVIC TYPE R (EK) '97"),
    ("Volkswagen", "Volkswagen Golf IV GTI '01"),
    
    # 3. FR Sports (Balanced)
    ("Nissan", "Nissan SILVIA spec-R AERO (S15) '99"),
    ("Toyota", "Toyota SPRINTER TRUENO GT-APEX (AE86) '83"),
    
    # 4. MR (Rotation/Snap) - Replacing Ferrari 360 with NSX
    ("Toyota", "Toyota MR2 GT-S '97"),
    ("Acura", "Acura NSX '04"), 
    
    # 5. AWD (Grip/Power)
    ("Subaru", "Subaru IMPREZA Sedan WRX STi Version VI '99"),
    ("Mitsubishi", "Mitsubishi Lancer Evolution VIII MR    GSR '04"),
    
    # 6. High Power (Restraint)
    ("Dodge", "Dodge VIPER SRT10 '03"),
    ("Ford", "Ford GT '05")
]

CURRICULUM_TRACKS = [
    # 1. Sanity Check
    "Driving Park Beginner Course",
    "Driving Park Test Course",
    
    # 2. Low Speed Technical
    "Autumn Ring Mini",
    "Tsukuba Circuit",
    
    # 3. Medium Flow
    "Grand Valley East",
    "Trial Mountain Circuit",
    
    # 4. Complex Elevation
    "Deep Forest Raceway",
    "Autumn Ring"
]

def make_env(rank, log_dir):
    def _init():
        env = GranTurismoEnv(
            env_index=rank,
            car_subset=CURRICULUM_CARS,
            track_subset=CURRICULUM_TRACKS
        )
        env = ResizeObservation(env, (84, 84))
        env = GrayscaleObservation(env, keep_dim=True)
        # Wrap with ResidualEnv (Injects Base Action)
        env = ResidualEnv(env, device="cuda") 
        return env
    return _init

def train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("Initializing Residual Training...")
    
    # Use DummyVecEnv for single process (avoids spawn issues with CUDA initialized in NitroGen)
    env = DummyVecEnv([make_env(0, logs_dir)])
    
    model = PPO(
        "MultiInputPolicy", # Essential for Dict obs
        env,
        verbose=1,
        tensorboard_log=logs_dir,
        learning_rate=3e-4,
        policy_kwargs=dict(
            net_arch=[64, 64], # Small MLP for Residuals
            # features_extractor_kwargs=dict(features_dim=128)
        )
    )
    
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=models_dir, name_prefix="resid_ppo")
    
    print("Starting Learn...")
    try:
        model.learn(total_timesteps=TOTAL_STEPS, callback=checkpoint_callback, progress_bar=True)
        model.save(os.path.join(models_dir, "final_residual"))
    except KeyboardInterrupt:
        model.save(os.path.join(models_dir, "interrupted_residual"))
        print("Saved.")
        
    env.close()

if __name__ == "__main__":
    train()
