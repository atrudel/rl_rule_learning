from enum import IntEnum

from state import Where, Color


class Action(IntEnum):
    LOOK_LEFT = 0
    LOOK_UP = 1
    LOOK_RIGHT = 2

    REPORT_RED = 3
    REPORT_GREEN = 4
    REPORT_BLUE = 5


def is_looking(action: Action):
    return action < 3


def is_reporting(action: Action):
    return action >= 3


def locate_gaze(action: Action) -> Where:
    if is_looking(action):
        return Where(action)
    else:
        return Where.CENTER


def identify_color(action: Action) -> Color:
    if is_reporting(action):
        return Color(action - 3)
    else:
        return Color.UNKNOWN
