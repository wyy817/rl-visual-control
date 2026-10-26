"""
Evaluate pretrained SB3 DQN on Pong as benchmark baseline.

Install deps first:
    pip install stable-baselines3 huggingface-sb3 gymnasium shimmy ale-py
"""
import numpy as np
import ale_py
import gymnasium
gymnasium.register_envs(ale_py)

# Also register in old gym namespace so SB3 internals can find PongNoFrameskip-v4
try:
    import gym as _gym
    _gym.register_envs(ale_py)
except Exception:
    pass

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env as sb3_make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from huggingface_sb3 import load_from_hub

N_EPISODES = 30
MODEL_REPO = "sb3/dqn-PongNoFrameskip-v4"
MODEL_FILE = "dqn-PongNoFrameskip-v4.zip"

print(f"Downloading pretrained SB3 DQN from: {MODEL_REPO}")
checkpoint = load_from_hub(MODEL_REPO, MODEL_FILE)

custom_objects = {
    "learning_rate": 1e-4,
    "lr_schedule": lambda _: 1e-4,
    "exploration_schedule": lambda _: 0.01,
    "optimize_memory_usage": False,
    "handle_timeout_termination": False,
}
model = DQN.load(checkpoint, device="cpu", custom_objects=custom_objects)
print("Model loaded.\n")

# Use SB3's own Atari preprocessing pipeline (matches training exactly)
# clip_reward=False and terminal_on_life_loss=False for fair evaluation scoring
vec_env = sb3_make_atari_env(
    "PongNoFrameskip-v4",
    n_envs=1,
    seed=0,
    wrapper_kwargs={"clip_reward": False, "terminal_on_life_loss": False},
)
vec_env = VecFrameStack(vec_env, n_stack=4)

rewards = []
for ep in range(N_EPISODES):
    obs = vec_env.reset()
    ep_reward = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)
        ep_reward += float(reward[0])
        done = bool(dones[0])
    # infos[0]["episode"]["r"] is more accurate when Monitor wrapper is present
    if infos[0].get("episode"):
        ep_reward = float(infos[0]["episode"]["r"])
    rewards.append(ep_reward)
    print(f"  Episode {ep + 1:>2d}: {ep_reward:>6.1f}")

vec_env.close()

print(f"\n[SB3 DQN Pretrained] Results over {N_EPISODES} episodes:")
print(f"  Mean ± Std : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
print(f"  Min / Max  : {np.min(rewards):.1f} / {np.max(rewards):.1f}")
print(f"\nReference scores:")
print(f"  Random agent (ours) : -20.60 ± 0.61")
print(f"  Human               : ~9.3")
