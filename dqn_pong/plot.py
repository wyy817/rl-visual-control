import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

TOTAL_STEPS    = 5_000_000
WARMUP_STEPS   = 10_000
EPS_START      = 1.0
EPS_END        = 0.01
EPS_DECAY_STEPS = 500_000


def smooth(data: np.ndarray, window: int) -> np.ndarray:
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def plot_training(results_dir: str = "results"):
    rewards_path = os.path.join(results_dir, "episode_rewards.npy")
    losses_path  = os.path.join(results_dir, "losses.npy")

    rewards = np.load(rewards_path)
    losses  = np.load(losses_path) if os.path.exists(losses_path) else np.array([])

    # ── 1. Raw episode reward ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rewards, alpha=0.3, color='steelblue', linewidth=0.8, label='Episode Reward')
    ax.axhline(y=9.3,   color='green', linestyle='--', linewidth=1.5, label='Human (~9.3)')
    ax.axhline(y=-20.7, color='red',   linestyle='--', linewidth=1.5, label='Random (~-20.7)')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Episode Reward during Training (Pong, DDQN+Dueling+PER)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(results_dir, "plot_reward_raw.png")
    plt.savefig(p, dpi=150, bbox_inches='tight')
    print(f"Saved: {p}")
    plt.close()

    # ── 2. Mean reward (100-episode rolling average) ──────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    if len(rewards) >= 100:
        mean100 = smooth(rewards, window=100)
        ax.plot(range(99, len(rewards)), mean100, color='steelblue', linewidth=1.8)
    ax.axhline(y=9.3,   color='green', linestyle='--', linewidth=1.5, label='Human (~9.3)')
    ax.axhline(y=-20.7, color='red',   linestyle='--', linewidth=1.5, label='Random (~-20.7)')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Reward (100-episode window)")
    ax.set_title("100-Episode Moving Average Reward (Pong, DDQN+Dueling+PER)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(results_dir, "plot_reward_mean100.png")
    plt.savefig(p, dpi=150, bbox_inches='tight')
    print(f"Saved: {p}")
    plt.close()

    # ── 3. TD Loss ────────────────────────────────────────────────────────────
    if len(losses) > 0:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(losses, alpha=0.15, color='orange', linewidth=0.5)
        if len(losses) >= 5000:
            sm_loss = smooth(losses, window=5000)
            ax.plot(range(4999, len(losses)), sm_loss,
                    color='darkorange', linewidth=2, label='Moving Average (5000 steps)')
        ax.set_xlabel("Update Step")
        ax.set_ylabel("Huber Loss")
        ax.set_title("TD Loss during Training (Pong, DDQN+Dueling+PER)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        p = os.path.join(results_dir, "plot_loss.png")
        plt.savefig(p, dpi=150, bbox_inches='tight')
        print(f"Saved: {p}")
        plt.close()

    # ── 4. Epsilon decay ──────────────────────────────────────────────────────
    steps = np.arange(1, TOTAL_STEPS + 1)
    eps = np.where(
        steps < WARMUP_STEPS,
        EPS_START,
        np.maximum(EPS_END, EPS_START - (EPS_START - EPS_END) * (steps - WARMUP_STEPS) / EPS_DECAY_STEPS)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, eps, color='mediumpurple', linewidth=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Epsilon (ε)")
    ax.set_title("Epsilon-Greedy Exploration Schedule (Pong, DDQN+Dueling+PER)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(results_dir, "plot_epsilon.png")
    plt.savefig(p, dpi=150, bbox_inches='tight')
    print(f"Saved: {p}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    plot_training(args.results_dir)
