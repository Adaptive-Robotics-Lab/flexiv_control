"""Numpy-only tests for the intended-motion preview math (no viser needed).

The load-bearing property: the preview IS the executor's command stream. If
``Robot.execute_cartesian_trajectory`` ever changes its resolution/cap/interpolation
logic without the preview following, the equality test here fails -- the
preview must never lie about what the robot will do.
"""

from __future__ import annotations

import numpy as np
import pytest

from flexiv_control import (
    CartesianTrajectory,
    CartesianWaypoint,
    GripperCommand,
    Robot,
    RobotConfig,
    SafetyProfile,
)
from flexiv_control.viz.preview import (
    effective_caps,
    plan_trajectory_preview,
    pose_distance,
    time_colors,
    trail_segments,
    workspace_box_edges,
)


def _traj(extra_kwargs=None, gripper=True):
    wps = [
        CartesianWaypoint(
            position=[0.50, 0.02, 0.30],
            quaternion=None,
            gripper=GripperCommand(width=0.02, grasp=True) if gripper else None,
            duration=0.8,
        ),
        CartesianWaypoint(
            position=[0.55, -0.04, 0.34],
            quaternion=None,
            gripper=GripperCommand(width=0.08) if gripper else None,
            duration=0.6,
        ),
    ]
    return CartesianTrajectory(waypoints=wps, **(extra_kwargs or {}))


def test_preview_equals_executed_command_stream():
    """The headline guarantee: preview setpoints == what the executor streams.

    The fake backend records every commanded pose; for an in-spec traj the
    safety filter is a no-op by design, so the executed log must equal the
    preview's per-tick poses exactly."""
    r = Robot(RobotConfig(backend="fake"))
    r.connect()
    r.acquire_lease("t")
    start = r.get_state()
    traj = _traj({"max_tcp_linear_speed": 0.12})
    pv = plan_trajectory_preview(traj, start.tcp_pose, r.profile, control_hz=r.control_hz)
    result = r.execute_cartesian_trajectory(traj)
    assert result.success
    executed = np.asarray(r.backend.cartesian_log, float)
    assert executed.shape == pv.setpoints.shape
    np.testing.assert_allclose(executed, pv.setpoints, atol=1e-9)


def test_effective_caps_tightening_only():
    p = SafetyProfile()  # 0.25 m/s, 0.60 rad/s
    lin, ang = effective_caps(_traj({"max_tcp_linear_speed": 0.12}), p)
    assert lin == pytest.approx(0.12)            # traj tightens
    lin, ang = effective_caps(_traj({"max_tcp_linear_speed": 10.0}), p)
    assert lin == pytest.approx(0.25)            # profile is never relaxed
    assert ang == pytest.approx(0.60)


def test_preview_reports_time_stretch():
    p = SafetyProfile()
    fast = CartesianTrajectory(
        waypoints=[CartesianWaypoint(position=[0.95, 0.0, 0.30], quaternion=None,
                                     duration=0.1)],
        max_tcp_linear_speed=0.05,
    )
    pv = plan_trajectory_preview(fast, np.array([0.45, 0, 0.3, 1, 0, 0, 0], float), p)
    assert pv.time_stretched
    assert pv.duration_s > pv.nominal_duration_s
    # the 50 cm hop also violates the workspace box -> annotated
    assert any("outside" in w for w in pv.warnings)


def test_gripper_events_and_polarity():
    pv = plan_trajectory_preview(
        _traj(), np.array([0.45, 0, 0.3, 1, 0, 0, 0], float), None
    )
    assert len(pv.gripper_events) == 2
    assert pv.gripper_events[0].closing is True      # grasp = closing
    assert pv.gripper_events[1].closing is False     # re-open
    assert pv.gripper_events[0].tick < pv.gripper_events[1].tick


def test_pose_distance():
    a = np.array([0.5, 0.0, 0.3, 1, 0, 0, 0], float)
    b = np.array([0.5, 0.003, 0.3, 1, 0, 0, 0], float)
    lin, ang = pose_distance(a, b)
    assert lin == pytest.approx(0.003)
    assert ang == pytest.approx(0.0, abs=1e-9)


def test_workspace_box_edges_shape_and_extent():
    p = SafetyProfile()
    edges = workspace_box_edges(p)
    assert edges.shape == (12, 2, 3)
    pts = edges.reshape(-1, 3)
    assert pts[:, 0].min() == pytest.approx(p.ws_x[0])
    assert pts[:, 0].max() == pytest.approx(p.ws_x[1])
    assert pts[:, 2].max() == pytest.approx(p.ws_z[1])


def test_time_colors_and_trail_segments():
    c = time_colors(10)
    assert c.shape == (10, 3) and c.dtype == np.uint8
    assert c[0, 2] > c[-1, 2]    # blue fades
    assert c[-1, 0] > c[0, 0]    # red rises
    segs = trail_segments(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], float))
    assert segs.shape == (2, 2, 3)
    assert trail_segments(np.zeros((1, 3))).shape == (0, 2, 3)
