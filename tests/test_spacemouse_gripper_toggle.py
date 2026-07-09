"""Gripper toggle must fire on the button's rising edge, not its level.

Regression test for the bug where holding the gripper button flipped the
gripper open/closed once per control tick (~100 Hz) for as long as it was held.
"""

import numpy as np

from flexiv_control.teleop import ScriptedSpaceMouseSource, SpaceMouseState, SpaceMouseTeleop


def _teleop(**kw):
    return SpaceMouseTeleop(robot=object(), source=ScriptedSpaceMouseSource(), **kw)


def _press(t, buttons):
    return SpaceMouseState(buttons=buttons)


def test_held_button_toggles_exactly_once():
    t = _teleop()
    held = SpaceMouseState(buttons=[0, 1])
    commands = [t._gripper_from_buttons(held) for _ in range(50)]
    fired = [c for c in commands if c is not None]
    assert len(fired) == 1, "a held button must toggle exactly once, not per tick"


def test_release_then_press_toggles_again():
    t = _teleop()
    held = SpaceMouseState(buttons=[0, 1])
    released = SpaceMouseState(buttons=[0, 0])

    first = t._gripper_from_buttons(held)
    assert first is not None
    assert t._gripper_from_buttons(held) is None  # still held: no re-fire
    assert t._gripper_from_buttons(released) is None
    second = t._gripper_from_buttons(held)
    assert second is not None
    # Open and close alternate, using the GN01 widths.
    assert {round(first.width, 3), round(second.width, 3)} == {0.09, 0.01}


def test_first_press_opens_by_default():
    t = _teleop()
    cmd = t._gripper_from_buttons(SpaceMouseState(buttons=[0, 1]))
    assert cmd is not None
    assert cmd.width == t.gripper_open_width
    assert cmd.grasp is False


def test_signs_flip_axes():
    t = _teleop(signs=[-1, 1, 1, 1, 1, 1], deadband=0.0)
    st = SpaceMouseState(translation=np.array([0.5, 0.5, 0.0]), rotation=np.zeros(3))
    delta = t.to_delta(st)
    assert delta[0] == -delta[1]
