



import torch.optim as optim

from agent import Agent
from environment import Environment

EPSILON = 0.3
LR = 1e-4

env = Environment()
agent = Agent(env, EPSILON, n_hidden=25)
optimizer = optim.AdamW(agent.policy_net.parameters(), lr=LR, amsgrad=True)
