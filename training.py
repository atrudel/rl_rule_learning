import matplotlib.pyplot as plt
import torch
from tqdm import trange

from agent import Agent, device
from environment import Environment

# Learning rate
LR = 1e-2
# Epsilon decay parameters
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100

# Discount factor
GAMMA = 0.99

BATCH_SIZE = 6
TAU = 0.1

# Nb of hidden neurons of the neural network's layers
N_HIDDEN = 20

N_EPISODES = 100


env = Environment()
agent = Agent(env, LR, EPS_START, EPS_END, EPS_DECAY, GAMMA, BATCH_SIZE, TAU, N_HIDDEN)

episode_rewards = []
epsilons = []
successes = []

for episode in trange(N_EPISODES):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    done = False
    episode_reward = 0

    while not done:
        action = agent.sample_action(state)
        next_state, reward, done = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        if done:
            next_state = None
        else:
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

        agent.observe(state, action, next_state, reward)
        state = next_state
        agent.update()
        agent.update_target_network()

        episode_reward += reward.item()
        if done:
            episode_rewards.append(episode_reward)
            epsilons.append(agent.epsilon())
            successes.append(env.success)

fig, axes = plt.subplots(3, 1)

ax: plt.Axes = axes[0]
ax.plot(episode_rewards)
ax.set_title('Evolution of the total reward over episodes')
ax.set_xlabel('Episode')
ax.set_ylabel('Total reward')

ax = axes[1]
ax.bar(list(range(len(successes))), successes)
ax.set_title('Time step of first correct answer for each episode')
ax.set_xlabel('Episode')
ax.set_ylabel('Time step')

ax = axes[2]
ax.plot(epsilons)
ax.set_title('Evolution of the exploration parameter over episodes')
ax.set_xlabel('Episode')
ax.set_ylabel('Epsilon')

plt.tight_layout()
plt.show()