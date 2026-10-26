"""
evaluate.py - 评估训练好的 PPO 策略并录制视频

用法:
  python evaluate.py --checkpoint checkpoints/ppo_final.pt --n_episodes 5
  python evaluate.py --checkpoint checkpoints/ppo_final.pt --record --video_dir videos/
"""

import os
import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from env_wrappers import GrayscaleWrapper, FrameStackWrapper
from network import ActorCritic


def make_eval_env(record: bool = False, video_dir: str = "videos"):
    """创建评估环境，可选录制视频"""
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    if record:
        os.makedirs(video_dir, exist_ok=True)
        env = RecordVideo(env, video_folder=video_dir,
                          episode_trigger=lambda ep: True,
                          name_prefix="ppo_eval")
    env = GrayscaleWrapper(env)
    env = FrameStackWrapper(env, n_stack=4)
    return env


@torch.no_grad()
def evaluate(checkpoint: str, n_episodes: int = 5,
             record: bool = False, video_dir: str = "videos",
             deterministic: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    policy = ActorCritic().to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    policy.load_state_dict(ckpt["policy_state"])
    policy.eval()
    print(f"Loaded checkpoint: {checkpoint}")

    env = make_eval_env(record=record, video_dir=video_dir)

    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        step = 0

        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, _, _ = policy.act(obs_t, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            step += 1

        rewards.append(ep_reward)
        print(f"  Episode {ep+1:>2d}: reward = {ep_reward:.2f}  steps = {step}")

    env.close()

    mean_r = np.mean(rewards)
    std_r  = np.std(rewards)
    print(f"\nResults over {n_episodes} episodes:")
    print(f"  Mean reward : {mean_r:.2f}")
    print(f"  Std  reward : {std_r:.2f}")
    print(f"  Min  reward : {np.min(rewards):.2f}")
    print(f"  Max  reward : {np.max(rewards):.2f}")
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    type=str,  required=True)
    parser.add_argument("--n_episodes",    type=int,  default=5)
    parser.add_argument("--record",        action="store_true")
    parser.add_argument("--video_dir",     type=str,  default="videos")
    parser.add_argument("--stochastic",    action="store_true",
                        help="Use stochastic policy instead of deterministic")
    args = parser.parse_args()

    evaluate(
        checkpoint   = args.checkpoint,
        n_episodes   = args.n_episodes,
        record       = args.record,
        video_dir    = args.video_dir,
        deterministic= not args.stochastic,
    )
