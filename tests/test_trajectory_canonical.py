"""Tests for the canonical action-traj upgrades: representation (absolute vs
relative-to-start), predict/execute horizon split, orientation-carrying array
constructor, and the normalized-gripper -> width mapping."""

from __future__ import annotations

import numpy as np

from flexiv_control import (
    CartesianTrajectory,
    TrajectoryRepresentation,
    GripperCommand,
    Robot,
    RobotConfig,
)


def test_gripper_from_normalized_maps_to_width():
    assert GripperCommand.from_normalized(1.0, span=0.08).width == 0.08
    assert GripperCommand.from_normalized(0.0, span=0.08).width == 0.0
    g = GripperCommand.from_normalized(0.5, span=0.1, min_width=0.01, force=30.0)
    assert abs(g.width - (0.01 + 0.5 * 0.1)) < 1e-9
    assert g.force == 30.0
    # clips out-of-range
    assert GripperCommand.from_normalized(2.0, span=0.08).width == 0.08


def test_from_pose_array_carries_orientation():
    # (H,9): x,y,z, qw,qx,qy,qz, w, n
    u = np.array([
        [0.45, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0, 1.0, 20],
        [0.50, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0, 20],  # 180deg about x
    ])
    c = CartesianTrajectory.from_pose_array(u)
    assert c.horizon == 2
    assert c.waypoints[0].quaternion is not None
    assert np.allclose(c.waypoints[1].quaternion, [0, 1, 0, 0])
    assert c.waypoints[0].gripper.width == 0.08  # w=1 -> open
    assert c.waypoints[1].gripper.width == 0.0  # w=0 -> closed


def test_horizon_pred_vs_exec():
    wps = CartesianTrajectory.from_waypoint_array(
        [[0.45, 0, 0.30, 1, 10], [0.46, 0, 0.30, 1, 10], [0.47, 0, 0.30, 1, 10]]
    )
    assert wps.horizon_pred == 3
    assert wps.horizon_exec == 3  # n_execute None -> all
    wps.n_execute = 1
    assert wps.horizon_exec == 1
    wps.n_execute = 99  # clamped to len
    assert wps.horizon_exec == 3


def test_relative_to_start_resolves_by_composition():
    # 90deg about z start; a +x relative delta -> +y in base
    start = np.array([0.4, 0.0, 0.3, np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    c = CartesianTrajectory(
        waypoints=CartesianTrajectory.from_waypoint_array([[0.1, 0.0, 0.0, 1, 10]]).waypoints,
        representation=TrajectoryRepresentation.RELATIVE_TO_START,
    )
    res = c.resolve_to_absolute(start)
    assert res.representation == TrajectoryRepresentation.ABSOLUTE
    # +0.1 in tcp x -> +0.1 in base y after 90deg z rotation
    assert np.allclose(res.waypoints[0].position, [0.4, 0.1, 0.3], atol=1e-6)
    # absolute traj resolve is identity
    abs_c = CartesianTrajectory.from_waypoint_array([[0.5, 0, 0.3, 1, 10]])
    assert abs_c.resolve_to_absolute(start) is abs_c


def test_for_execution_slices_and_resolves():
    c = CartesianTrajectory(
        waypoints=CartesianTrajectory.from_waypoint_array(
            [[0.1, 0, 0, 1, 10], [0.2, 0, 0, 1, 10], [0.3, 0, 0, 1, 10]]
        ).waypoints,
        representation=TrajectoryRepresentation.RELATIVE_TO_START,
        n_execute=1,
    )
    start = np.array([0.4, 0.0, 0.3, 1.0, 0, 0, 0])
    e = c.for_execution(start)
    assert len(e.waypoints) == 1  # sliced to H_exec
    assert e.representation == TrajectoryRepresentation.ABSOLUTE
    assert np.allclose(e.waypoints[0].position, [0.5, 0.0, 0.3])  # 0.4 + 0.1


def test_execute_relative_chunk_executes_only_exec_horizon_on_fake():
    robot = Robot(RobotConfig(backend="fake"))
    robot.connect()
    robot.start_cartesian_impedance()
    s0 = robot.get_state()
    # two relative +x steps, but execute only the first
    c = CartesianTrajectory(
        waypoints=CartesianTrajectory.from_waypoint_array(
            [[0.05, 0, 0, 1, 20], [0.10, 0, 0, 1, 20]]
        ).waypoints,
        representation=TrajectoryRepresentation.RELATIVE_TO_START,
        n_execute=1,
    )
    res = robot.execute_cartesian_trajectory(c)
    assert res.success
    end = robot.get_state()
    # moved ~+0.05 in x from start (only first waypoint), not +0.10
    dx = end.tcp_position[0] - s0.tcp_position[0]
    assert 0.03 < dx < 0.07, dx
    robot.disconnect()
