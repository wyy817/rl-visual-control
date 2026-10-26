"""
network.py - CNN Actor-Critic 网络（连续动作空间）

输入: (batch, 4, 96, 96) float32
Actor 输出: 3维 Gaussian 分布的 mean（经 tanh 约束到各动作范围）
           + log_std（独立可学习参数）
Critic 输出: 标量 V(s)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

# CarRacing-v3 动作边界
ACTION_LOW = torch.tensor([-1.0, 0.0, 0.0])   # [steering, gas, brake]
ACTION_HIGH = torch.tensor([1.0, 1.0, 1.0])


def _action_scale(raw_mean: torch.Tensor) -> torch.Tensor:
    """将 tanh 输出 [-1,1]^3 线性映射到各动作实际范围"""
    low = ACTION_LOW.to(raw_mean.device)
    high = ACTION_HIGH.to(raw_mean.device)
    return low + (torch.tanh(raw_mean) + 1.0) * 0.5 * (high - low)


class CNNBackbone(nn.Module):
    """共享 CNN 特征提取器，输入 (B, 4, 96, 96)"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # -> (B, 32, 23, 23)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # -> (B, 64, 10, 10)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # -> (B, 64, 8, 8)
            nn.ReLU(),
            nn.Flatten(),                                # -> (B, 4096)
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
        )
        self.output_dim = 512

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    """
    共享 CNN backbone 的 Actor-Critic。

    Actor:  CNN -> Linear(512, 3) 输出 raw_mean，再经 _action_scale 映射
            log_std 为独立可学习参数（不依赖状态）
    Critic: CNN -> Linear(512, 1) 输出 V(s)
    """

    LOG_STD_MIN = -3.0
    LOG_STD_MAX = 0.5

    def __init__(self):
        super().__init__()
        self.backbone = CNNBackbone()

        self.actor_head = nn.Linear(self.backbone.output_dim, 3)
        self.log_std = nn.Parameter(torch.zeros(3))   # 可学习 log σ

        self.critic_head = nn.Linear(self.backbone.output_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Actor 输出层用更小的初始化
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)

    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: (B, 4, 96, 96) float32, 已归一化到 [0, 1]
        Returns:
            mean:  (B, 3)  动作均值（已映射到实际范围）
            std:   (B, 3)  动作标准差
            value: (B, 1)  状态价值 V(s)
        """
        feat = self.backbone(obs)
        raw_mean = self.actor_head(feat)
        mean = _action_scale(raw_mean)

        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp().expand_as(mean)

        value = self.critic_head(feat)
        return mean, std, value

    def get_dist(self, obs: torch.Tensor) -> Normal:
        mean, std, _ = self.forward(obs)
        return Normal(mean, std)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        PPO 更新时使用：计算 log_prob、entropy、value。
        Returns:
            log_prob: (B,)  每个动作的联合 log 概率
            entropy:  (B,)  每个分布的联合熵
            value:    (B, 1)
        """
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)   # 3个维度相加
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic=False):
        """
        单步采样，用于 rollout 收集。
        Returns:
            action:   (3,) numpy  clipped 到合法范围
            log_prob: scalar tensor
            value:    scalar tensor
        """
        mean, std, value = self.forward(obs)
        if deterministic:
            action_t = mean
        else:
            dist = Normal(mean, std)
            action_t = dist.rsample()
            action_t = action_t.clamp(
                ACTION_LOW.to(obs.device), ACTION_HIGH.to(obs.device)
            )

        dist = Normal(mean, std)
        log_prob = dist.log_prob(action_t).sum(dim=-1)

        action_np = action_t.squeeze(0).cpu().numpy()
        return action_np, log_prob.squeeze(), value.squeeze()
