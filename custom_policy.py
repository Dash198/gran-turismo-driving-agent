"""
CustomFeaturesExtractor v3 — Hybrid
────────────────────────────────────
v1-size CNN (3050-friendly) with updated input dims:
  frame: 64×64×(3×n_stack) mask channels
  aux: 8×n_stack proprioceptive dims
"""

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """
    Dict observation extractor:
      - 'frame': (64, 64, C) — C=3 base, C=6 with n_stack=2
      - 'aux': (N,) — N=8 base, N=16 with n_stack=2
    """

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=256)

        frame_shape = observation_space["frame"].shape
        aux_dim = observation_space["aux"].shape[0]

        # Detect CHW vs HWC
        if frame_shape[0] < frame_shape[1]:
            n_input_channels = frame_shape[0]
        else:
            n_input_channels = frame_shape[-1]

        # CNN — v1 size (fast on 3050, proven 12+ FPS)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
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

        # Aux MLP — larger than v1 to handle 8-dim proprioception
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, 64),
            nn.ReLU(),
        )

        # Combined → 256 features
        combined_dim = self.cnn_output_dim + 64
        self.combined = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> th.Tensor:
        frame = observations["frame"]
        cnn_features = self.cnn(frame)

        aux = observations["aux"]
        aux_features = self.aux_mlp(aux)

        combined = th.cat([cnn_features, aux_features], dim=1)
        return self.combined(combined)


# Policy kwargs for SAC
policy_kwargs = dict(
    features_extractor_class=CustomFeaturesExtractor,
    use_sde=False,
)
