"""The unified [-1,1] gripper action encoding (GripperCommand.from_signed_action),
shared by the gym env, the LeRobot adapter, and teleop -- the [-1,1]->[0,1] mirror
of from_normalized, so the delta-action and traj surfaces agree."""

from __future__ import annotations

import numpy as np

from flexiv_control import GripperCommand


def test_from_signed_action_endpoints():
    op = GripperCommand.from_signed_action(1.0, span=0.08)
    assert abs(op.width - 0.08) < 1e-12 and op.grasp is False
    cl = GripperCommand.from_signed_action(-1.0, span=0.08)
    assert abs(cl.width - 0.0) < 1e-12 and cl.grasp is True
    mid = GripperCommand.from_signed_action(0.0, span=0.08)
    assert abs(mid.width - 0.04) < 1e-12 and mid.grasp is False  # half-open, not grasping


def test_from_signed_action_mirrors_from_normalized():
    for a in (-1.0, -0.3, 0.0, 0.5, 1.0):
        signed = GripperCommand.from_signed_action(a, span=0.08, min_width=0.0)
        norm = GripperCommand.from_normalized((a + 1.0) / 2.0, span=0.08, min_width=0.0)
        assert abs(signed.width - norm.width) < 1e-12
        assert signed.grasp == (a < 0.0)


def test_from_signed_action_clips_and_respects_min_width():
    assert GripperCommand.from_signed_action(5.0, span=0.10).width == 0.10  # clipped to +1
    g = GripperCommand.from_signed_action(-1.0, span=0.10, min_width=0.01)
    assert abs(g.width - 0.01) < 1e-12  # closed maps to min_width, not 0
    assert np.isclose(GripperCommand.from_signed_action(1.0, span=0.10, min_width=0.01).width, 0.11)
