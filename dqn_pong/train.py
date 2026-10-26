import numpy as np
import torch
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from env_wrappers import make_atari_env
from replay_buffer import PrioritizedReplayBuffer
from agent import DDQNAgent

# ── Hyperparameters ──────────────────────────────────────────────────────────
ENV_ID             = "ALE/Pong-v5"
TOTAL_STEPS        = 5_000_000
RESUME_FROM        = None             # set to checkpoint path to resume
WARMUP_STEPS       = 10_000
BATCH_SIZE         = 32
BUFFER_CAPACITY    = 100_000
LEARNING_RATE      = 1e-4
GAMMA              = 0.99
EPS_START          = 1.0
EPS_END            = 0.01
EPS_DECAY_STEPS    = 500_000
TARGET_UPDATE_FREQ = 1_000
UPDATE_FREQ        = 4
CHECKPOINT_FREQ    = 50_000
LOG_FREQ           = 10_000
SEED               = 42

# PER
PER_ALPHA          = 0.6   # prioritization exponent (0 = uniform, 1 = full priority)
PER_BETA_START     = 0.4   # IS weight exponent — anneals to 1.0 over training
PER_BETA_END       = 1.0

CHECKPOINT_DIR = "checkpoints_per_v1"
RESULTS_DIR    = "results_per_v1"
TB_DIR         = "results_per_v1/tb"
# ─────────────────────────────────────────────────────────────────────────────


def hwc_to_chw(obs: np.ndarray) -> np.ndarray:
    return np.transpose(obs, (2, 0, 1))


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=TB_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env = make_atari_env(ENV_ID, clip_reward=True, episodic_life=True, seed=SEED)
    n_actions = env.action_space.n
    hw_shape  = env.observation_space.shape       # (84, 84, 4)
    obs_shape = (hw_shape[2], hw_shape[0], hw_shape[1])  # (4, 84, 84) channel-first
    print(f"Env: {ENV_ID} | Actions: {n_actions} | Obs: {obs_shape}")

    buffer = PrioritizedReplayBuffer(BUFFER_CAPACITY, obs_shape, alpha=PER_ALPHA)
    agent  = DDQNAgent(
        n_actions=n_actions,
        device=device,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        eps_start=EPS_START,
        eps_end=EPS_END,
        target_update_freq=TARGET_UPDATE_FREQ,
        dueling=True,
    )
    eps_decay_per_step = (EPS_START - EPS_END) / EPS_DECAY_STEPS

    episode_rewards = []
    losses          = []
    recent_rewards  = deque(maxlen=100)
    start_step      = 0

    if RESUME_FROM:
        agent.load(RESUME_FROM)
        start_step = int(os.path.basename(RESUME_FROM).replace("ckpt_", "").replace(".pth", ""))
        rewards_path = os.path.join(RESULTS_DIR, "episode_rewards.npy")
        losses_path  = os.path.join(RESULTS_DIR, "losses.npy")
        if os.path.exists(rewards_path):
            episode_rewards = list(np.load(rewards_path))
        if os.path.exists(losses_path):
            losses = list(np.load(losses_path))
        recent_rewards = deque(episode_rewards[-100:], maxlen=100)
        print(f"Resumed from step {start_step} | eps={agent.eps:.4f} | episodes so far={len(episode_rewards)}")

    obs, _         = env.reset()
    obs            = hwc_to_chw(obs)
    episode_reward = 0.0
    episode_count  = len(episode_rewards)
    best_mean      = max(np.mean(recent_rewards), -float('inf')) if recent_rewards else -float('inf')
    start_time     = time.time()

    for step in range(start_step + 1, TOTAL_STEPS + 1):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = hwc_to_chw(next_obs)
        done     = terminated or truncated

        buffer.add(obs, action, reward, next_obs, terminated)
        obs             = next_obs
        episode_reward += reward

        if done:
            obs, _ = env.reset()
            obs    = hwc_to_chw(obs)
            episode_rewards.append(episode_reward)
            recent_rewards.append(episode_reward)
            writer.add_scalar("train/episode_reward", episode_reward, step)
            episode_count  += 1
            episode_reward  = 0.0

        # ── Epsilon decay ─────────────────────────────────────────────────
        if step >= WARMUP_STEPS:
            agent.eps = max(EPS_END, agent.eps - eps_decay_per_step)

        # ── Training update ───────────────────────────────────────────────
        if len(buffer) >= WARMUP_STEPS and step % UPDATE_FREQ == 0:
            beta = PER_BETA_START + (PER_BETA_END - PER_BETA_START) * (step / TOTAL_STEPS)
            obs_b, act_b, rew_b, next_obs_b, done_b, tree_indices, weights = buffer.sample(BATCH_SIZE, beta=beta)
            loss, td_errors = agent.update((obs_b, act_b, rew_b, next_obs_b, done_b), weights)
            buffer.update_priorities(tree_indices, td_errors)
            losses.append(loss)
            writer.add_scalar("train/loss", loss, step)

        # ── Logging ───────────────────────────────────────────────────────
        if step % LOG_FREQ == 0:
            mean_r  = np.mean(recent_rewards) if recent_rewards else 0.0
            elapsed = time.time() - start_time
            fps     = (step - start_step) / elapsed
            writer.add_scalar("train/mean_reward_100ep", mean_r, step)
            writer.add_scalar("train/epsilon", agent.eps, step)
            writer.add_scalar("train/fps", fps, step)
            print(
                f"Step {step:>8d} | Ep: {episode_count:>5d} | "
                f"Mean(100): {mean_r:>7.2f} | Eps: {agent.eps:.3f} | "
                f"FPS: {fps:.0f} | Time: {elapsed:.0f}s"
            )

        # ── Checkpointing ─────────────────────────────────────────────────
        if step % CHECKPOINT_FREQ == 0:
            path = os.path.join(CHECKPOINT_DIR, f"ckpt_{step:08d}.pth")
            agent.save(path)
            np.save(os.path.join(RESULTS_DIR, "episode_rewards.npy"), np.array(episode_rewards))
            np.save(os.path.join(RESULTS_DIR, "losses.npy"), np.array(losses))

            mean_r = np.mean(recent_rewards) if recent_rewards else 0.0
            if mean_r > best_mean:
                best_mean = mean_r
                agent.save(os.path.join(CHECKPOINT_DIR, "best_model.pth"))
                print(f"  ↑ New best mean reward: {best_mean:.2f} → saved best_model.pth")

    # ── Final save ────────────────────────────────────────────────────────────
    np.save(os.path.join(RESULTS_DIR, "episode_rewards.npy"), np.array(episode_rewards))
    np.save(os.path.join(RESULTS_DIR, "losses.npy"), np.array(losses))
    writer.close()
    print("Training complete!")
    env.close()


if __name__ == "__main__":
    train()
