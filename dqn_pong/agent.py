import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from network import DuelingDQN


class DDQNAgent:
    """
    Double DQN agent with an optional Dueling network architecture.

    Double DQN decouples action selection (online net) from value evaluation
    (target net) to reduce Q-value overestimation:
        y = r + gamma * Q_target(s', argmax_a Q_online(s', a))
    """

    def __init__(
        self,
        n_actions: int,
        device: torch.device,
        lr: float = 1e-4,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        target_update_freq: int = 1000,
        dueling: bool = True,
    ):
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.target_update_freq = target_update_freq
        self.steps = 0

        self.online_net = DuelingDQN(n_actions, dueling=dueling).to(device)
        self.target_net = DuelingDQN(n_actions, dueling=dueling).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr, eps=1.5e-4)

    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        eps = 0.05 if eval_mode else self.eps
        if np.random.random() < eps:
            return np.random.randint(self.n_actions)
        obs_t = torch.from_numpy(np.array(obs, dtype=np.uint8)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.online_net(obs_t).argmax(dim=1).item()

    def update(self, batch, weights=None):
        obs, actions, rewards, next_obs, dones = batch

        obs_t      = torch.from_numpy(obs).to(self.device)
        next_obs_t = torch.from_numpy(next_obs).to(self.device)
        actions_t  = torch.from_numpy(actions).long().to(self.device)
        rewards_t  = torch.from_numpy(rewards).to(self.device)
        dones_t    = torch.from_numpy(dones).to(self.device)

        with torch.no_grad():
            # Online net picks the best next action
            next_actions = self.online_net(next_obs_t).argmax(dim=1, keepdim=True)
            # Target net evaluates that action's value
            next_q = self.target_net(next_obs_t).gather(1, next_actions).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        current_q = self.online_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        td_errors = (target_q - current_q).abs().detach().cpu().numpy()

        elementwise_loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
        if weights is not None:
            weights_t = torch.from_numpy(weights).to(self.device)
            loss = (weights_t * elementwise_loss).mean()
        else:
            loss = elementwise_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self.steps += 1

        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item(), td_errors

    def save(self, path: str):
        torch.save({
            'online_net':  self.online_net.state_dict(),
            'target_net':  self.target_net.state_dict(),
            'optimizer':   self.optimizer.state_dict(),
            'steps':       self.steps,
            'eps':         self.eps,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt['online_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.steps = ckpt['steps']
        self.eps   = ckpt['eps']
