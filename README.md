# Reinforcement Learning from Pixels: PPO and Double DQN

Two from-scratch PyTorch implementations of deep reinforcement-learning agents that learn to act from raw pixels (XJTLU DTS307TC Reinforcement Learning, 2026, individual coursework).

| Project | Environment | Algorithm | Result |
| --- | --- | --- | --- |
| [`ppo_carracing/`](ppo_carracing/) | Gymnasium `CarRacing-v3` (continuous control) | PPO with a CNN actor–critic, GAE, Gaussian policy, 8 parallel environments | After 2M timesteps: **853.83 ± 55.47** mean reward vs 768.35 ± 181.75 for a Stable-Baselines3 PPO baseline: about 11% higher return and about 3x lower standard deviation |
| [`dqn_pong/`](dqn_pong/) | Atari `ALE/Pong-v5` | Double DQN with dueling architecture and prioritised experience replay | Trained for 5M environment steps; formal 30-episode evaluation 5.13 ± 5.68 (see the README for how training and evaluation numbers differ) |

## Highlights

- Everything is implemented from scratch in PyTorch (agents, rollout / replay buffers, environment wrappers, training loops), not wrapped from a library; Stable-Baselines3 is used only as a reference baseline.
- PPO: clipped surrogate objective, generalised advantage estimation, continuous Gaussian policy, vectorised environments.
- DQN: Double-DQN targets, dueling value / advantage heads, proportional prioritised replay with annealed importance sampling.
- The PPO training run was launched as a SLURM job on a GPU node (`ppo_carracing/run_ppo.sh`).

## Running

Each folder is self-contained; see its README or the scripts' `--help`. Requires Python, PyTorch and Gymnasium (`ale-py` for Pong).

```bash
cd ppo_carracing && python train.py --n_envs 8 --n_steps 512 --total_timesteps 2000000
cd dqn_pong && python train.py
```

Checkpoints, logs, videos and the coursework reports are not included, and the code has not been re-run for this repository.
