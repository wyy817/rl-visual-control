import numpy as np


class SumTree:
    """
    Binary sum tree for O(log n) priority sampling and update.
    Leaves store individual priorities; internal nodes store subtree sums.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.write = 0
        self.n_entries = 0

    @property
    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float):
        idx = self.write + self.capacity - 1
        self._update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, tree_idx: int, priority: float):
        self._update(tree_idx, priority)

    def _update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s: float):
        """Return (tree_idx, priority, data_idx) for sampled value s."""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], data_idx


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer.
    Samples transitions proportional to |TD error|^alpha.
    Importance sampling weights correct for the non-uniform sampling bias.
    """

    def __init__(self, capacity: int, obs_shape: tuple, alpha: float = 0.6, epsilon: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

        self.observations      = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions           = np.zeros(capacity, dtype=np.int64)
        self.rewards           = np.zeros(capacity, dtype=np.float32)
        self.dones             = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        data_idx = self.tree.write
        self.observations[data_idx]      = obs
        self.next_observations[data_idx] = next_obs
        self.actions[data_idx]           = action
        self.rewards[data_idx]           = reward
        self.dones[data_idx]             = float(done)
        self.tree.add(self.max_priority ** self.alpha)

    def sample(self, batch_size: int, beta: float = 0.4):
        tree_indices = np.empty(batch_size, dtype=np.int64)
        priorities   = np.empty(batch_size, dtype=np.float32)
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            tree_idx, priority, _ = self.tree.get(s)
            tree_indices[i] = tree_idx
            priorities[i]   = max(priority, 1e-8)

        # Importance sampling weights
        probs   = priorities / self.tree.total
        weights = (self.tree.n_entries * probs) ** (-beta)
        weights /= weights.max()
        weights  = weights.astype(np.float32)

        data_indices = np.clip(tree_indices - self.capacity + 1, 0, self.capacity - 1)

        return (
            self.observations[data_indices],
            self.actions[data_indices],
            self.rewards[data_indices],
            self.next_observations[data_indices],
            self.dones[data_indices],
            tree_indices,
            weights,
        )

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray):
        for idx, td_err in zip(tree_indices, td_errors):
            priority = (abs(float(td_err)) + self.epsilon) ** self.alpha
            self.tree.update(int(idx), priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return self.tree.n_entries
