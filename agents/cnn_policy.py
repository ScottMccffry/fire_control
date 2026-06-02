"""
Spatial feature extractor for the drone-swarm wildfire RL policy.

Combines a small convolutional network over the multi-channel fire/drone map
with an MLP over the per-drone feature vector, then concatenates the two. Using
a CNN over the map (instead of flattening it into a giant vector) makes the
policy translation-aware and lets it scale to large grids; keeping the per-drone
vector preserves drone identity for the per-drone action head.

The CNN uses padding + adaptive average pooling so the same architecture works
for any grid size (32x32, 100x100, ...).
"""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FireSwarmExtractor(BaseFeaturesExtractor):
    """CNN over the map channels + MLP over the drone vector."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_out: int = 128,
        drone_out: int = 64,
        pool: int = 4,
    ):
        super().__init__(observation_space, features_dim=cnn_out + drone_out)

        n_ch = observation_space["map"].shape[0]
        drone_dim = observation_space["drones"].shape[0]
        self.pool = pool

        self.cnn = nn.Sequential(
            nn.Conv2d(n_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((pool, pool)),
            nn.Flatten(),
        )
        self.cnn_head = nn.Sequential(
            nn.Linear(32 * pool * pool, cnn_out), nn.ReLU()
        )
        self.drone_mlp = nn.Sequential(
            nn.Linear(drone_dim, drone_out), nn.ReLU()
        )

    def forward(self, observations: dict) -> torch.Tensor:
        m = self.cnn_head(self.cnn(observations["map"]))
        d = self.drone_mlp(observations["drones"])
        return torch.cat([m, d], dim=1)
