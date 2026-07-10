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


class _FakeRobot:
    """Records servo calls; reports a wide-open gripper."""

    def __init__(self, width=0.09):
        self._width = width
        self.calls = []

    def get_state(self):
        from types import SimpleNamespace
        return SimpleNamespace(gripper_width=self._width)

    def acquire_lease(self, *a, **k):
        pass

    def start_cartesian_impedance(self, *a, **k):
        pass

    def servo_cartesian_delta(self, delta, duration=None, frame=None, gripper=None):
        self.calls.append((np.asarray(delta, float).copy(), gripper))


def test_initial_state_inferred_from_robot_width():
    robot = _FakeRobot(width=0.09)  # physically open
    t = SpaceMouseTeleop(robot=robot, source=ScriptedSpaceMouseSource())
    cmd = t._gripper_from_buttons(SpaceMouseState(buttons=[0, 1]))
    assert cmd is not None
    assert cmd.width == t.gripper_close_width  # open -> first press closes
    assert cmd.grasp is True


def test_gripper_actuates_with_deadman_released():
    robot = _FakeRobot()
    presses = iter([
        SpaceMouseState(buttons=[0, 0]),
        SpaceMouseState(buttons=[0, 1]),  # gripper press, deadman released
        SpaceMouseState(buttons=[0, 1]),
        SpaceMouseState(buttons=[0, 0]),
    ])
    t = SpaceMouseTeleop(
        robot=robot,
        source=ScriptedSpaceMouseSource(lambda _t: next(presses)),
        publish_hz=1000.0,
    )
    t.run(max_ticks=4)
    grips = [g for _d, g in robot.calls if g is not None]
    assert len(grips) == 1, "one rising edge -> exactly one gripper command"
    deltas = [d for d, g in robot.calls if g is not None]
    assert np.allclose(deltas[0], 0.0), "no motion while the deadman is released"
