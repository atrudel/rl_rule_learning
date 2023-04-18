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
    Nobs = 17

    def reset(self, rule: Rule = None) -> np.ndarray:
        self.rule = Rule(random.randint(0, 2)) if rule is None else rule
        self.cues, self.correct_color = self.create_cue_permutation(self.rule)
        self.state = State()
        self.count = 0
        self.success = 0
        return self.state.to_array()

    def step(self, action: Action) -> Tuple[np.ndarray, int, bool]:
        gaze_direction: Where = locate_gaze(action)
        seen_color: Color = self.get_seen_color(gaze_direction)
        reported_color: Color = identify_reported_color(action)
        reward: int = self.calculate_reward(action)
        self.state.add_information(gaze_direction, seen_color, reported_color)
        self.count += 1
        self.register_success(action)
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

    def register_success(self, action: Action) -> None:
        if identify_reported_color(action) == self.correct_color:
            if self.success == 0:
                self.success = self.count

    def __repr__(self) -> str:
        return f"Environment[Rule={self.rule}, Corr_Color={self.correct_color} | Cues={self.cues}]"

