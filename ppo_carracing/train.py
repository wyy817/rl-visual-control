"""
train.py - PPO 训练主循环（支持多并行环境）

用法示例:
  python train.py                                  # 默认 8 个并行环境
  python train.py --n_envs 1                       # 单环境（调试）
  python train.py --n_envs 8 --n_steps 512        # HPC 推荐配置

有效 batch size = n_steps * n_envs（默认 512*8=4096）
"""

import os
import time
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from env_wrappers import make_vec_env, make_env
from ppo_agent import PPOAgent


# -----------------------------------------------------------------------
# 超参数默认值
# -----------------------------------------------------------------------
DEFAULTS = dict(
    total_timesteps = 2_000_000,
    n_envs          = 8,          # 并行环境数
    n_steps         = 512,        # 每个 env 的 rollout 长度（总 batch = n_steps * n_envs）
    n_epochs        = 10,
    batch_size      = 256,        # mini-batch size
    lr              = 3e-4,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_epsilon    = 0.2,
    value_coef      = 0.5,
    entropy_coef    = 0.01,
    max_grad_norm   = 0.5,
    seed            = 42,
    log_dir         = "runs/ppo_carracing",
    save_dir        = "checkpoints",
    save_interval   = 100_000,
)


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    if var_y == 0:
        return float("nan")
    return float(1 - np.var(y_true - y_pred) / var_y)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}  |  n_envs: {args.n_envs}  |  n_steps: {args.n_steps}")
    print(f"[Train] Effective batch size: {args.n_envs * args.n_steps}")

    # 创建并行环境
    envs = make_vec_env(n_envs=args.n_envs, seed=args.seed)
    obs_shape  = envs.single_observation_space.shape   # (4, 96, 96)
    action_dim = envs.single_action_space.shape[0]     # 3

    agent = PPOAgent(
        obs_shape     = obs_shape,
        action_dim    = action_dim,
        n_envs        = args.n_envs,
        lr            = args.lr,
        clip_epsilon  = args.clip_epsilon,
        n_epochs      = args.n_epochs,
        batch_size    = args.batch_size,
        value_coef    = args.value_coef,
        entropy_coef  = args.entropy_coef,
        max_grad_norm = args.max_grad_norm,
        gamma         = args.gamma,
        gae_lambda    = args.gae_lambda,
        n_steps       = args.n_steps,
        device        = device,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    # ------------------------------------------------------------------
    # 训练循环
    # ------------------------------------------------------------------
    obs, _ = envs.reset()          # (n_envs, 4, 96, 96)
    total_steps     = 0
    episode_count   = 0
    last_save_step  = 0
    start_time      = time.time()

    # 每个 env 独立跟踪 episode reward
    ep_rewards      = np.zeros(args.n_envs, dtype=np.float32)
    recent_rewards  = []           # 用于打印近 10 episode 平均

    while total_steps < args.total_timesteps:

        # ---- Rollout 收集 ----
        rollout_vals = []

        for _ in range(args.n_steps):
            actions, log_probs, values = agent.collect_step(obs)
            next_obs, rewards, terminated, truncated, _ = envs.step(actions)
            dones = terminated | truncated

            agent.buffer.add(obs, actions, log_probs, rewards, values, dones)
            rollout_vals.append(values.copy())

            ep_rewards += rewards
            for i, done in enumerate(dones):
                if done:
                    recent_rewards.append(ep_rewards[i])
                    writer.add_scalar("train/episode_reward", ep_rewards[i], total_steps)
                    ep_rewards[i] = 0.0
                    episode_count += 1

            obs = next_obs
            total_steps += args.n_envs

        # ---- PPO 更新 ----
        stats = agent.update(obs, terminated | truncated)

        # Explained variance
        vals_arr = np.array(rollout_vals)                     # (n_steps, n_envs)
        rets_arr = agent.buffer.returns if not agent.buffer.is_full() else (
            vals_arr + agent.buffer.advantages
        )
        ev = explained_variance(vals_arr.flatten(), rets_arr.flatten()
                                if hasattr(rets_arr, 'flatten') else vals_arr.flatten())

        # ---- TensorBoard ----
        writer.add_scalar("train/policy_loss",       stats["policy_loss"],   total_steps)
        writer.add_scalar("train/value_loss",        stats["value_loss"],    total_steps)
        writer.add_scalar("train/entropy",           stats["entropy"],       total_steps)
        writer.add_scalar("train/clip_fraction",     stats["clip_fraction"], total_steps)
        writer.add_scalar("train/explained_variance", ev,                    total_steps)

        elapsed = time.time() - start_time
        fps = total_steps / max(elapsed, 1e-8)
        mean_rew = np.mean(recent_rewards[-10:]) if recent_rewards else 0.0
        print(
            f"[{total_steps:>8d}/{args.total_timesteps}] "
            f"ep={episode_count} "
            f"rew={mean_rew:.1f} "
            f"policy_loss={stats['policy_loss']:.4f} "
            f"value_loss={stats['value_loss']:.4f} "
            f"entropy={stats['entropy']:.4f} "
            f"fps={fps:.0f}"
        )

        # ---- 定期保存 ----
        if total_steps - last_save_step >= args.save_interval:
            ckpt_path = os.path.join(args.save_dir, f"ppo_step{total_steps}.pt")
            agent.save(ckpt_path)
            print(f"  -> Saved: {ckpt_path}")
            last_save_step = total_steps

    agent.save(os.path.join(args.save_dir, "ppo_final.pt"))
    envs.close()
    writer.close()
    print("Training complete.")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key, val in DEFAULTS.items():
        t = type(val) if not isinstance(val, bool) else lambda x: x.lower() != "false"
        parser.add_argument(f"--{key}", type=t, default=val)
    args = parser.parse_args()
    train(args)
