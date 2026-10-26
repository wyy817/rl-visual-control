import numpy as np
import torch
import argparse

from env_wrappers import make_atari_env
from agent import DDQNAgent

ENV_ID = "ALE/Pong-v5"


def _run_episodes(env, action_fn, n_episodes: int):
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = action_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        print(f"  Episode {ep + 1:>2d}: {ep_reward:>6.1f}")
    return rewards


def evaluate_random(n_episodes: int = 30, render: bool = False, seed: int = 0):
    env = make_atari_env(
        ENV_ID,
        clip_reward=False,
        episodic_life=False,
        render_mode="human" if render else None,
        seed=seed,
    )
    print(f"[Random Agent] Running {n_episodes} episodes...")
    rewards = _run_episodes(env, lambda obs: env.action_space.sample(), n_episodes)
    env.close()
    return rewards


def evaluate(checkpoint_path: str, n_episodes: int = 10, render: bool = False, seed: int = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_atari_env(
        ENV_ID,
        clip_reward=False,
        episodic_life=False,
        render_mode="human" if render else None,
        seed=seed,
    )

    n_actions = env.action_space.n
    agent = DDQNAgent(n_actions=n_actions, device=device)
    agent.load(checkpoint_path)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Agent steps: {agent.steps:,} | eps: {agent.eps:.3f}")

    def trained_action(obs):
        obs_t = np.transpose(obs, (2, 0, 1))
        return agent.select_action(obs_t, eval_mode=True)

    rewards = _run_episodes(env, trained_action, n_episodes)
    env.close()
    return rewards


def print_summary(rewards, n_episodes, label=""):
    tag = f"[{label}] " if label else ""
    print(f"\n{tag}Results over {n_episodes} episodes:")
    print(f"  Mean ± Std : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Min / Max  : {np.min(rewards):.1f} / {np.max(rewards):.1f}")
    print(f"  Human score: ~9.3  |  Random score: ~-20.7")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints_per_v1/best_model.pth")
    parser.add_argument("--episodes",   type=int, default=30)
    parser.add_argument("--render",     action="store_true")
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--random",     action="store_true", help="Run random agent baseline instead of trained model")
    args = parser.parse_args()

    if args.random:
        rewards = evaluate_random(args.episodes, args.render, args.seed)
        print_summary(rewards, args.episodes, "Random Agent")
    else:
        rewards = evaluate(args.checkpoint, args.episodes, args.render, args.seed)
        print_summary(rewards, args.episodes, "Trained Agent")
