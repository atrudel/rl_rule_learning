import random
from typing import Tuple

import numpy as np

from actions import Action, is_reporting, locate_gaze, identify_color
from state import State, Rule, Where, Color

MAX_STEPS = 6

WRONG_ANSWER_REWARD = -1
UNDECIDED_REWARD = -2


class Environment:
    Na = 6

    def reset(self, rule: Rule = None) -> State:
        self.rule = Rule(random.randint(0, 2)) if rule is None else rule
        self.cues, self.correct_color = self.create_cue_permutation(self.rule)
        self.state = State(
            count=0,
            seen_color=Color.UNKNOWN,
            where_gaze=Where.CENTER
        )
        return self.state

    def step(self, action: Action) -> Tuple[State, int, bool]:
        current_state = self.state
        seen_color = self.get_seen_color(action)
        reward = self.calculate_reward(action)
        new_state = State(
            count=current_state.count + 1,
            seen_color=seen_color,
            where_gaze=locate_gaze(action)
        )
        self.state = new_state
        done = (new_state.count >= MAX_STEPS)
        return new_state, reward, done

    def get_seen_color(self, action: Action) -> Color:
        gaze_direction: Where = locate_gaze(action)
        if gaze_direction == Where.CENTER:
            return Color.UNKNOWN
        else:
            return Color(self.cues[gaze_direction])

    def calculate_reward(self, action: Action) -> int:
        if is_reporting(action):
            if identify_color(action) == self.correct_color:
                return 0
            else:
                return WRONG_ANSWER_REWARD
        else:
            if state.count > 2:
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
