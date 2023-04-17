import numpy as np

from actions import Action
from environment import Environment


class Agent:
    def __init__(self, env: Environment, epsilon: float):
        n_observations = 6 * 6
        self.q_table: np.ndarray = np.zeros((n_observations, 6))
        self.epsilon: float = epsilon
        self.env: Environment = env

    def sample_action(self, state) -> Action:
        # Epsilon-greedy policy
        if np.random.uniform(0, 1) < self.epsilon:
            return Action(np.random.randint(self.env.Na))
        else:
            return np.argmax(self.q_table[state.to_int()])




