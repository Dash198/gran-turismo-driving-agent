
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import cv2

# Import NitroGenBase
from nitrogen_base import NitrogenBase
# Import Vision Rewards
from vision_rewards import VisionRewardSystem

class ResidualEnv(gym.Wrapper):
    """
    Wraps GranTurismoEnv to support Residual Learning.
    
    Observation Space:
        Dict({
            "image": Box(84, 84, 1), # Grayscale for PPO efficiency
            "base_action": Box(4,)   # [Steer, Gas, Brake, Reverse] from NitroGen
        })
        
    Action Space:
        Box(low=-1, high=1, shape=(4,)) # [Delta_Steer, Delta_Gas, Delta_Brake, Delta_Reverse]
        
    Logic:
        final_action = clip(base_action + residual_action, -1, 1)
    """
    def __init__(self, env, device="cuda"):
        super().__init__(env)
        self.nitrogen = NitrogenBase(device=device)
        self.reward_system = VisionRewardSystem()
        
        # Define New Spaces
        # 1. Action: Continuous Deltas [Steer, Gas, Brake, Reverse]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # 2. Observation: MultiInput (Image + Base Action)
        self.observation_space = spaces.Dict({
            "image": env.observation_space, 
            "base_action": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        })
        
        self.last_base_action = np.zeros(4, dtype=np.float32)
        
        # Residual Scaling Factors:
        # Steer: 10%, Gas/Brake: 15%, Reverse: 15%
        self.residual_scales = np.array([0.10, 0.15, 0.15, 0.15], dtype=np.float32)

    def step(self, residual_action):
        """
        Args:
            residual_action: [delta_steer, delta_gas, delta_brake, delta_rev]
        """
        # 1. Apply Scales to Residuals
        scaled_residual = residual_action * self.residual_scales
        
        # 2. Combine with NitroGen's base action
        base = self.last_base_action
        final_vec = np.clip(base + scaled_residual, -1.0, 1.0) 
        
        # 3. Map Continuous [0..1] to Discrete (for GT Env)
        steer_val = final_vec[0]
        gas_val = final_vec[1]
        brake_val = final_vec[2]
        rev_val = final_vec[3]
        
        # Steering Logic
        steer_discrete = 0
        if steer_val < -0.3: steer_discrete = 1 # Left 
        elif steer_val > 0.3: steer_discrete = 2 # Right
        
        # Pedal Logic (Priority: Gas > Brake > Reverse)
        pedal_discrete = 0
        if gas_val > 0.3: pedal_discrete = 1
        elif brake_val > 0.3: pedal_discrete = 2
        elif rev_val > 0.3: pedal_discrete = 3
        
        discrete_action = np.array([steer_discrete, pedal_discrete])
        
        # 4. Step Underlying Env
        obs, reward, terminated, truncated, info = self.env.step(discrete_action)
        
        # 5. Compute Vision Reward & Metrics
        # Get raw BGR frame and speed from unwrapped env
        rgb_obs = self.env.unwrapped.current_obs
        if rgb_obs is None:
             rgb_obs = np.zeros((272, 480, 3), dtype=np.uint8)
        
        current_speed = self.env.unwrapped.current_speed
             
        # Compute vision reward (passing speed for stuck detection)
        vis_reward, vis_metrics, vis_done = self.reward_system.compute_reward(rgb_obs, speed=current_speed)
        
        # Overwrite original reward or add
        reward = vis_reward
        
        # Merge metrics into info
        info.update(vis_metrics)
        info["vis_speed"] = current_speed
        
        if vis_done:
            terminated = True
            info["termination_reason"] = vis_metrics.get("vis_term_reason_str", "vision_safety_trigger")
            
        # 6. Compute Base Action for NEXT Step
        self.last_base_action = self.nitrogen.get_action_vector(rgb_obs)
        
        # 7. Construct Dict Observation
        new_obs = {
            "image": obs, 
            "base_action": self.last_base_action
        }
        
        # 8. Residual Penalty (Regularization)
        penalty = 0.01 * np.linalg.norm(residual_action) 
        reward -= penalty
        info["vis_residual_penalty"] = penalty
        
        return new_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Initial NitroGen Inference
        rgb_obs = self.env.unwrapped.current_obs
        if rgb_obs is None:
             rgb_obs = np.zeros((272, 480, 3), dtype=np.uint8)

        self.last_base_action = self.nitrogen.get_action_vector(rgb_obs)
        
        new_obs = {
            "image": obs,
            "base_action": self.last_base_action
        }
        return new_obs, info
