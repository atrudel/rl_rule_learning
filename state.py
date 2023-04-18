from enum import IntEnum
from dataclasses import dataclass

import numpy as np


class Rule(IntEnum):
    LEFT = 0
    UP = 1
    RIGHT = 2

class Color(IntEnum):
    RED = 0
    GREEN = 1
    BLUE = 2
    UNKNOWN = 3

class Where(IntEnum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    CENTER = 3


@dataclass
class State:
    def __init__(self):
        self.information = np.zeros(shape=(3, 3))
        self.position = Where.CENTER
        self.attempts = np.zeros(4)

    def add_information(self, gaze_direction: Where, seen_color: Color, reported_color: Color):
        if gaze_direction != Where.CENTER:
            self.information[gaze_direction, seen_color] = 1
        self.position = gaze_direction
        self.attempts[reported_color] += 1

    def __repr__(self) -> str:
        return f"State[Position={self.position} | info=\n{self.information}\n attempts=\n{self.attempts}\n]"

    def to_array(self) -> np.ndarray:
        array = np.zeros(9 + 4 + 4)
        array[:9] = self.information.flatten()
        array[9 + self.position] = 1
        array[13:17] = self.attempts
        return array


# Test
if __name__ == '__main__':
    state = State()
    print(state)
    print(state.to_array())

    state.add_information(Where.UP, Color.RED, reported_color=Color.UNKNOWN)
    print(state)
    print(state.to_array())

    state.add_information(Where.CENTER, Color.UNKNOWN, reported_color=Color.RED)
    print(state)
    print(state.to_array())

    state.add_information(Where.LEFT, Color.BLUE, reported_color=Color.UNKNOWN)
    print(state)
    print(state.to_array())

    state.add_information(Where.UP, Color.RED, reported_color=Color.UNKNOWN)
    print(state)
    print(state.to_array())