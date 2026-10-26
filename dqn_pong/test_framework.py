import sys
import numpy as np
import torch

print("=" * 50)
print("Test 1: Network shapes")
print("=" * 50)
from network import DuelingDQN

net_dueling = DuelingDQN(n_actions=6, dueling=True)
net_plain   = DuelingDQN(n_actions=6, dueling=False)
dummy = torch.zeros(2, 4, 84, 84)
out1 = net_dueling(dummy)
out2 = net_plain(dummy)
assert out1.shape == (2, 6), f"Expected (2,6), got {out1.shape}"
assert out2.shape == (2, 6), f"Expected (2,6), got {out2.shape}"
print(f"  Dueling DQN output: {out1.shape}  OK")
print(f"  Plain   DQN output: {out2.shape}  OK")

print()
print("=" * 50)
print("Test 2: Replay buffer")
print("=" * 50)
from replay_buffer import ReplayBuffer

buf = ReplayBuffer(capacity=1000, obs_shape=(4, 84, 84))
for _ in range(50):
    buf.add(
        np.zeros((4, 84, 84), dtype=np.uint8),
        action=2,
        reward=1.0,
        next_obs=np.ones((4, 84, 84), dtype=np.uint8),
        done=False,
    )
obs, act, rew, nobs, done = buf.sample(8)
assert obs.shape  == (8, 4, 84, 84), f"obs shape wrong: {obs.shape}"
assert act.shape  == (8,),           f"act shape wrong: {act.shape}"
assert rew.shape  == (8,),           f"rew shape wrong: {rew.shape}"
assert nobs.shape == (8, 4, 84, 84), f"nobs shape wrong: {nobs.shape}"
print(f"  obs:       {obs.shape}  OK")
print(f"  actions:   {act.shape}        OK")
print(f"  rewards:   {rew.shape}        OK")
print(f"  next_obs:  {nobs.shape}  OK")

print()
print("=" * 50)
print("Test 3: Agent update (CPU)")
print("=" * 50)
from agent import DDQNAgent

device = torch.device("cpu")
agent = DDQNAgent(n_actions=6, device=device, dueling=True)

# Fake batch
batch = (
    np.zeros((32, 4, 84, 84), dtype=np.uint8),
    np.zeros(32, dtype=np.int64),
    np.ones(32,  dtype=np.float32),
    np.zeros((32, 4, 84, 84), dtype=np.uint8),
    np.zeros(32, dtype=np.float32),
)
loss = agent.update(batch)
print(f"  Loss value: {loss:.6f}  OK")
print(f"  Agent steps after update: {agent.steps}")

action = agent.select_action(np.zeros((4, 84, 84), dtype=np.uint8))
assert 0 <= action < 6
print(f"  select_action returned: {action}  OK")

print()
print("=" * 50)
print("Test 4: Environment wrappers")
print("=" * 50)
from env_wrappers import make_atari_env

env = make_atari_env("ALE/Pong-v5", clip_reward=True, episodic_life=True, seed=42)
obs, info = env.reset()
assert obs.shape == (84, 84, 4), f"Obs shape wrong: {obs.shape}"
print(f"  Reset obs shape: {obs.shape}  OK")

obs2, r, terminated, truncated, info = env.step(env.action_space.sample())
assert obs2.shape == (84, 84, 4)
print(f"  Step obs shape:  {obs2.shape}  OK")
print(f"  Reward: {r}  (clipped, should be -1/0/1)")
print(f"  n_actions: {env.action_space.n}")
env.close()

print()
print("=" * 50)
print("ALL TESTS PASSED")
print("=" * 50)
