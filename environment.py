import random
from typing import Tuple

import numpy as np

from actions import Action, is_reporting, locate_gaze, identify_reported_color
from state import State, Rule, Where, Color

MAX_STEPS = 6

WRONG_ANSWER_REWARD = -1
UNDECIDED_REWARD = -2


class Environment:
    Na = 6
    Nobs = 14

    def reset(self, rule: Rule = None) -> State:
        self.rule = Rule(random.randint(0, 2)) if rule is None else rule
        self.cues, self.correct_color = self.create_cue_permutation(self.rule)
        self.state = State()
        self.count = 0
        return self.state

    def step(self, action: Action) -> Tuple[np.ndarray, int, bool]:
        gaze_direction: Where = locate_gaze(action)
        seen_color: Color = self.get_seen_color(gaze_direction)
        reported_color: Color = identify_reported_color(action)

        reward: int = self.calculate_reward(action)
        self.state.add_information(gaze_direction, seen_color, reported_color)
        self.count += 1
        done = (self.count >= MAX_STEPS)
        return self.state.to_array(), reward, done

    def get_seen_color(self, gaze_direction: Where) -> Color:
        if gaze_direction == Where.CENTER:
            return Color.UNKNOWN
        else:
            return Color(self.cues[gaze_direction])

    def calculate_reward(self, action: Action) -> int:
        if is_reporting(action):
            if identify_reported_color(action) == self.correct_color:
                return 0
            else:
                return WRONG_ANSWER_REWARD
        else:
            if self.count > 2:
                return UNDECIDED_REWARD
            else:
                return 0

    def create_cue_permutation(self, rule: Rule) -> Tuple[np.ndarray, Color]:
        permutation = np.random.randint(3, size=3)
        permutation[1] = Color(rule)
        correct_color = permutation[rule]
        return permutation, correct_color

    def __repr__(self) -> str:
        return f"Environment[Rule={self.rule}, Corr_Color={self.correct_color} | Cues={self.cues}]"


if __name__ == '__main__':
    env = Environment()
    state = env.reset()
    print(env)
    print(state)
    print()
    done = False
    while not done:
        action = int(input("enter action"))
        state, reward, done = env.step(Action(action))
        print(env)
        print(state)
        print(f"Reward={reward}")
        print(f"done={done}")
        print()
