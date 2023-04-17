from enum import IntEnum
from dataclasses import dataclass

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
    count: int
    seen_color: Color
    where_gaze: Where

    def __repr__(self) -> str:
        return f"State({self.count})[Color={self.seen_color}, where={self.where_gaze}]"

    def to_int(self) -> int:
        if self.where_gaze == Where.CENTER:
            assert self.seen_color == Color.UNKNOWN
            return 10
        return 3 * self.where_gaze + self.seen_color