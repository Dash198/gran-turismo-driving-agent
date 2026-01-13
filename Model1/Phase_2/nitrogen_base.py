
import sys
import os
import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# Paths
REPO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../NitroGen_Repo")
sys.path.append(os.path.abspath(REPO_PATH))

try:
    from nitrogen.inference_session import InferenceSession
    from nitrogen.shared import BUTTON_ACTION_TOKENS
except ImportError:
    print("FATAL: Could not import NitroGen.")
    sys.exit(1)

class NitrogenBase:
    """
    Reusable NitroGen Wrapper for Phase 2.
    Provides 'get_base_action(obs)' returning flat vector [steer, gas, brake].
    """
    def __init__(self, device="cuda"):
        print("[NitroGenBase] Initializing...")
        self.device = device
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # Load Model
        try:
            ckpt_path = hf_hub_download(repo_id="nvidia/NitroGen", filename="ng.pt")
            self.session = InferenceSession.from_ckpt(ckpt_path, old_layout=False, cfg_scale=1.0)
            self.session.model.to(dtype=self.dtype)
            
            # Warmup
            dummy = Image.new('RGB', (256, 256), (0,0,0))
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    self.session.predict(dummy)
            print("[NitroGenBase] Ready.")
        except Exception as e:
            print(f"[NitroGenBase] Error: {e}")
            raise e
            
        # Indices
        self.idx_rt = BUTTON_ACTION_TOKENS.index('RIGHT_TRIGGER')
        self.idx_lt = BUTTON_ACTION_TOKENS.index('LEFT_TRIGGER')
        self.idx_a  = BUTTON_ACTION_TOKENS.index('SOUTH') # Cross/Gas
        self.idx_x  = BUTTON_ACTION_TOKENS.index('WEST')  # Square/Brake
        self.idx_y  = BUTTON_ACTION_TOKENS.index('NORTH') # Triangle/Reverse

    def get_action_vector(self, obs_np):
        """
        Returns continuous vector: [steer, gas_prob, brake_prob, reverse_prob]
        Range: [-1, 1], [0, 1], [0, 1], [0, 1]
        """
        image = Image.fromarray(obs_np)
        try:
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    outputs = self.session.predict(image)
        except:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # 1. Steering (Trajectory Average 3 steps)
        # Shape: (18, 2)
        steer_vals = outputs['j_left'][:3, 0]
        steer_avg = np.mean(steer_vals)
        
        # 2. Pedals & Buttons (Average 3 steps)
        btns = outputs['buttons'][:3]
        gas_avg = np.mean(np.maximum(btns[:, self.idx_rt], btns[:, self.idx_a]))
        brake_avg = np.mean(np.maximum(btns[:, self.idx_lt], btns[:, self.idx_x]))
        rev_avg = np.mean(btns[:, self.idx_y])
        
        return np.array([steer_avg, gas_avg, brake_avg, rev_avg], dtype=np.float32)
