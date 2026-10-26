"""
ppo_agent.py - PPO 更新逻辑（支持多环境）

collect_step 接受 (n_envs, C, H, W) 批量 obs，
返回 (n_envs, action_dim) / (n_envs,) 的批量结果。
"""

import numpy as np
import torch
import torch.nn as nn
from network import ActorCritic, ACTION_LOW, ACTION_HIGH
from torch.distributions import Normal
from rollout_buffer import RolloutBuffer


class PPOAgent:

    def __init__(
        self,
        obs_shape: tuple,
        action_dim: int,
        n_envs: int = 1,
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 256,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        n_steps: int = 512,
        device: torch.device = torch.device("cpu"),
    ):
        self.clip_epsilon  = clip_epsilon
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.value_coef    = value_coef
        self.entropy_coef  = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device        = device
        self.n_envs        = n_envs

        self.policy    = ActorCritic().to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

        self.buffer = RolloutBuffer(
            n_steps    = n_steps,
            n_envs     = n_envs,
            obs_shape  = obs_shape,
            action_dim = action_dim,
            gamma      = gamma,
            gae_lambda = gae_lambda,
            device     = device,
        )

    # ------------------------------------------------------------------
    # 采样（批量，支持 n_envs）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect_step(self, obs_np: np.ndarray):
        """
        obs_np: (n_envs, C, H, W)
        returns:
            actions:   (n_envs, action_dim) numpy
            log_probs: (n_envs,) numpy
            values:    (n_envs,) numpy
        """
        obs_t = torch.FloatTensor(obs_np).to(self.device)   # (n_envs, C, H, W)
        mean, std, value = self.policy.forward(obs_t)

        dist     = Normal(mean, std)
        action_t = dist.rsample()
        action_t = action_t.clamp(
            ACTION_LOW.to(self.device), ACTION_HIGH.to(self.device)
        )
        log_prob = dist.log_prob(action_t).sum(dim=-1)       # (n_envs,)

        return (action_t.cpu().numpy(),
                log_prob.cpu().numpy(),
                value.squeeze(-1).cpu().numpy())

    # ------------------------------------------------------------------
    # PPO 更新
    # ------------------------------------------------------------------

    def update(self, last_obs_np: np.ndarray, last_dones: np.ndarray):
        """
        last_obs_np:  (n_envs, C, H, W)
        last_dones:   (n_envs,) bool
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(last_obs_np).to(self.device)
            _, _, last_vals = self.policy.forward(obs_t)
            last_vals = last_vals.squeeze(-1).cpu().numpy()   # (n_envs,)

        self.buffer.compute_gae(last_vals, last_dones)

        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        total_clip_frac   = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            for obs_b, acts_b, old_lp_b, rets_b, adv_b in self.buffer.get_batches(self.batch_size):

                new_lp, entropy, values = self.policy.evaluate_actions(obs_b, acts_b)

                ratio    = torch.exp(new_lp - old_lp_b)
                clip_adv = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_b
                policy_loss = -torch.min(ratio * adv_b, clip_adv).mean()

                value_loss  = nn.functional.mse_loss(values.squeeze(-1), rets_b)
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += (-entropy_loss.item())
                total_clip_frac   += clip_frac
                n_updates += 1

        self.buffer.reset()

        return {
            "policy_loss":   total_policy_loss / n_updates,
            "value_loss":    total_value_loss  / n_updates,
            "entropy":       total_entropy     / n_updates,
            "clip_fraction": total_clip_frac   / n_updates,
        }

    # ------------------------------------------------------------------
    # 存取检查点
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "policy_state":    self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
