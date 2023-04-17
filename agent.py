from collections import namedtuple, deque
from random import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from actions import Action
from environment import Environment

device = 'cpu'
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_dim):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class Agent:
    def __init__(self, env: Environment, epsilon: float, n_hidden: int = 25):
        self.env = env
        self.epsilon: float = epsilon
        self.policy_net = DQN(env.Nobs, env.Na, n_hidden).to(device)
        self.target_net = DQN(env.Nobs, env.Na, n_hidden).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory = ReplayMemory(1000)

    def sample_action(self, state: np.ndarray) -> Action:
        # Epsilon-greedy policy
        if np.random.uniform(0, 1) < self.epsilon:
            return torch.tensor([[
                Action(np.random.randint(self.env.Na))
            ]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)




