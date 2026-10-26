"""
rollout_buffer.py - On-policy Rollout Buffer with GAE (支持多环境)

数据维度: (n_steps, n_envs, ...)
GAE 按每个 env 独立计算，get_batches 时展平成 (n_steps * n_envs, ...)
"""

import numpy as np
import torch


class RolloutBuffer:

    def __init__(self, n_steps: int, n_envs: int, obs_shape: tuple,
                 action_dim: int, gamma: float = 0.99, gae_lambda: float = 0.95,
                 device: torch.device = torch.device("cpu")):
        self.n_steps    = n_steps
        self.n_envs     = n_envs
        self.obs_shape  = obs_shape
        self.action_dim = action_dim
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.device     = device
        self._ptr       = 0
        self._full      = False

        self.obs       = np.zeros((n_steps, n_envs, *obs_shape),  dtype=np.float32)
        self.actions   = np.zeros((n_steps, n_envs, action_dim),  dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs),              dtype=np.float32)
        self.rewards   = np.zeros((n_steps, n_envs),              dtype=np.float32)
        self.values    = np.zeros((n_steps, n_envs),              dtype=np.float32)
        self.dones     = np.zeros((n_steps, n_envs),              dtype=np.float32)

        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns    = np.zeros((n_steps, n_envs), dtype=np.float32)

    # ------------------------------------------------------------------
    # 数据收集
    # ------------------------------------------------------------------

    def add(self, obs, actions, log_probs, rewards, values, dones):
        """
        每步添加所有 env 的数据。
        obs:       (n_envs, *obs_shape)
        actions:   (n_envs, action_dim)
        log_probs: (n_envs,)
        rewards:   (n_envs,)
        values:    (n_envs,)
        dones:     (n_envs,)  bool or float
        """
        self.obs[self._ptr]       = obs
        self.actions[self._ptr]   = actions
        self.log_probs[self._ptr] = log_probs
        self.rewards[self._ptr]   = rewards
        self.values[self._ptr]    = values
        self.dones[self._ptr]     = dones.astype(np.float32)
        self._ptr += 1
        if self._ptr >= self.n_steps:
            self._full = True

    def is_full(self) -> bool:
        return self._full

    def reset(self):
        self._ptr  = 0
        self._full = False

    # ------------------------------------------------------------------
    # GAE 计算（每个 env 独立）
    # ------------------------------------------------------------------

    def compute_gae(self, last_values: np.ndarray, last_dones: np.ndarray):
        """
        last_values: (n_envs,)  V(s_{T+1})
        last_dones:  (n_envs,)  bool/float
        """
        last_gae = np.zeros(self.n_envs, dtype=np.float32)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]

            delta = (self.rewards[t]
                     + self.gamma * next_values * next_non_terminal
                     - self.values[t])
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    # ------------------------------------------------------------------
    # Mini-batch 迭代器
    # ------------------------------------------------------------------

    def get_batches(self, batch_size: int):
        """展平 (n_steps, n_envs) -> (n_steps*n_envs,) 后打乱，按 batch_size 切分"""
        assert self._full
        n_total = self.n_steps * self.n_envs
        indices = np.random.permutation(n_total)

        obs_flat      = self.obs.reshape(n_total, *self.obs_shape)
        actions_flat  = self.actions.reshape(n_total, self.action_dim)
        lp_flat       = self.log_probs.reshape(n_total)
        returns_flat  = self.returns.reshape(n_total)
        adv_flat      = self.advantages.reshape(n_total)

        # 归一化优势（在全部数据上统计）
        adv_norm = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        obs_t    = torch.FloatTensor(obs_flat[indices]).to(self.device)
        acts_t   = torch.FloatTensor(actions_flat[indices]).to(self.device)
        lp_t     = torch.FloatTensor(lp_flat[indices]).to(self.device)
        rets_t   = torch.FloatTensor(returns_flat[indices]).to(self.device)
        adv_t    = torch.FloatTensor(adv_norm[indices]).to(self.device)

        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            yield obs_t[start:end], acts_t[start:end], lp_t[start:end], rets_t[start:end], adv_t[start:end]
