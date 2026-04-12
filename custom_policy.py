"""
CustomFeaturesExtractor v2 — Paper-Aligned
───────────────────────────────────────────
CNN architecture closer to Paper 1:
  Conv: 64→128→256→512→FC128 (vs old 32→64→64→FC256)
Observation: 64×64×3 RGB + 8-dim aux (with frame stack)
"""

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """
    Extracts features from dict observation:
      - 'frame': (64, 64, C) CNN features — C=3 base, C=6 with n_stack=2
      - 'aux': (N,) auxiliary vector — N=8 base, N=16 with n_stack=2
    """

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=256)

        frame_shape = observation_space["frame"].shape
        aux_dim = observation_space["aux"].shape[0]

        # After VecTransposeImage, shape is (C, H, W); before it's (H, W, C)
        if frame_shape[0] < frame_shape[1]:
            n_input_channels = frame_shape[0]  # CHW
        else:
            n_input_channels = frame_shape[-1]  # HWC

        # CNN — Paper 1 architecture (scaled for 64×64 input)
        # Paper 1: Conv(64,4,2) → Conv(128,4,2) → Conv(256,4,2) → Conv(512,4,2) → FC(128)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size dynamically
        with th.no_grad():
            if frame_shape[0] < frame_shape[1]:
                sample = th.zeros(1, *frame_shape)
            else:
                sample = th.zeros(1, frame_shape[2], frame_shape[0], frame_shape[1])
            cnn_output = self.cnn(sample)
            self.cnn_output_dim = cnn_output.shape[1]

        # FC to compress CNN features (Paper 1: FC → 128)
        self.cnn_fc = nn.Sequential(
            nn.Linear(self.cnn_output_dim, 128),
            nn.ReLU(),
        )

        # MLP for auxiliary input
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, 64),
            nn.ReLU(),
        )

        # Combined features → output dim
        # Paper 1: 128 (CNN) + 17 (proprio) → 2048×4 MLP
        # Ours: 128 (CNN) + 64 (aux MLP) → 256 (budget-friendly)
        combined_dim = 128 + 64
        self.combined = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> th.Tensor:
        frame = observations["frame"]
        cnn_features = self.cnn_fc(self.cnn(frame))

        aux = observations["aux"]
        aux_features = self.aux_mlp(aux)

        combined = th.cat([cnn_features, aux_features], dim=1)
        return self.combined(combined)


# Policy kwargs for SAC
policy_kwargs = dict(
    features_extractor_class=CustomFeaturesExtractor,
    use_sde=False,
)
