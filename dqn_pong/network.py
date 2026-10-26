import torch
import torch.nn as nn
import numpy as np


class DuelingDQN(nn.Module):
    """
    Dueling DQN with the Nature-DQN CNN backbone.

    Q(s,a) = V(s) + A(s,a) - mean_a'[A(s,a')]

    Set dueling=False to use a plain DQN head (shared FC -> Q-values).
    Input: (B, 4, 84, 84) uint8 tensor (normalised internally to [0,1]).
    """

    def __init__(self, n_actions: int, dueling: bool = True):
        super().__init__()
        self.dueling = dueling

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out = self._conv_out_size()

        if dueling:
            self.value_stream = nn.Sequential(
                nn.Linear(conv_out, 512), nn.ReLU(), nn.Linear(512, 1)
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(conv_out, 512), nn.ReLU(), nn.Linear(512, n_actions)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(conv_out, 512), nn.ReLU(), nn.Linear(512, n_actions)
            )

    def _conv_out_size(self) -> int:
        dummy = torch.zeros(1, 4, 84, 84)
        return int(np.prod(self.conv(dummy).shape[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        features = self.conv(x).flatten(1)

        if self.dueling:
            value = self.value_stream(features)          # (B, 1)
            advantage = self.advantage_stream(features)  # (B, n_actions)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            return self.fc(features)
