"""
env_wrappers.py - CarRacing-v3 预处理 Wrapper

Action space: Box(3,) continuous
  action[0]: steering  [-1, 1]
  action[1]: gas        [0, 1]
  action[2]: brake      [0, 1]

Observation pipeline:
  RGB (96, 96, 3) uint8
  -> Grayscale (1, 96, 96) float32  [0, 1]
  -> FrameStack (4, 96, 96) float32 [0, 1]
"""

import gymnasium as gym
import numpy as np


class GrayscaleWrapper(gym.ObservationWrapper):
    """RGB (H, W, 3) uint8 -> Grayscale (1, H, W) float32, 归一化到 [0, 1]"""

    def __init__(self, env):
        super().__init__(env)
        h, w, _ = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1, h, w), dtype=np.float32
        )

    def observation(self, obs):
        # ITU-R BT.601 亮度权重
        gray = 0.299 * obs[:, :, 0] + 0.587 * obs[:, :, 1] + 0.114 * obs[:, :, 2]
        return (gray[np.newaxis, :, :] / 255.0).astype(np.float32)


class FrameStackWrapper(gym.Wrapper):
    """堆叠最近 n_stack 帧: (1, H, W) -> (n_stack, H, W)"""

    def __init__(self, env, n_stack=4):
        super().__init__(env)
        self.n_stack = n_stack
        c, h, w = env.observation_space.shape  # c=1
        self.frames = np.zeros((n_stack, h, w), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(n_stack, h, w), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames[:] = 0.0
        self.frames[-1] = obs[0]   # 最新帧放最后一位
        return self.frames.copy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames = np.roll(self.frames, shift=-1, axis=0)
        self.frames[-1] = obs[0]
        return self.frames.copy(), reward, terminated, truncated, info


def make_env(seed=None):
    """创建预处理好的 CarRacing-v3 环境"""
    env = gym.make("CarRacing-v3", render_mode=None)
    if seed is not None:
        env.reset(seed=seed)
    env = GrayscaleWrapper(env)
    env = FrameStackWrapper(env, n_stack=4)
    return env


def make_env_render():
    """用于可视化/录制的环境（带渲染）"""
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    env = GrayscaleWrapper(env)
    env = FrameStackWrapper(env, n_stack=4)
    return env


def make_vec_env(n_envs: int = 8, seed: int = 0):
    """创建 n_envs 个并行环境（AsyncVectorEnv），每个 env 用不同 seed"""
    def make_single(rank):
        def _init():
            return make_env(seed=seed + rank)
        return _init
    return gym.vector.AsyncVectorEnv([make_single(i) for i in range(n_envs)])
