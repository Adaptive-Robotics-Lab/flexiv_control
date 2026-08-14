import numpy as np
import pytest

from flexiv_control import (
    CartesianDelta,
    CartesianTrajectory,
    CartesianWaypoint,
    ExecutionResult,
    JointTrajectory,
    JointWaypoint,
)


def test_waypoint_requires_duration_or_frames():
    with pytest.raises(ValueError):
        CartesianWaypoint(position=[0.4, 0.0, 0.3])


def test_waypoint_resolve_duration_from_frames():
    wp = CartesianWaypoint(position=[0.4, 0.0, 0.3], n_frames=20)
    assert wp.resolve_duration(100.0) == pytest.approx(0.2)
    wp2 = CartesianWaypoint(position=[0.4, 0.0, 0.3], duration=0.5)
    assert wp2.resolve_duration(1000.0) == 0.5


def test_quaternion_normalised_or_held():
    wp = CartesianWaypoint(position=[0, 0, 0], quaternion=[0, 0, 0, 2], duration=0.1)
    assert np.isclose(np.linalg.norm(wp.quaternion), 1.0)
    held = CartesianWaypoint(position=[0, 0, 0], duration=0.1)
    assert held.quaternion is None


def test_from_waypoint_array_shape_and_mapping():
    u = np.array(
        [
            [0.45, 0.0, 0.30, 1.0, 20],
            [0.50, 0.05, 0.25, 0.0, 10],
        ]
    )
    traj = CartesianTrajectory.from_waypoint_array(u)
    assert traj.horizon == 2
    # gripper command width scales with w in [0,1] -> [0, 0.08]
    assert traj.waypoints[0].gripper.width == pytest.approx(0.08)
    assert traj.waypoints[1].gripper.width == pytest.approx(0.0)
    # n_frames preserved as integers
    assert traj.waypoints[0].n_frames == 20
    assert traj.waypoints[1].n_frames == 10
    # orientation held
    assert traj.waypoints[0].quaternion is None
    # total duration at 100 Hz = (20+10)/100
    assert traj.total_duration(100.0) == pytest.approx(0.30)


def test_from_waypoint_array_rejects_bad_shape():
    with pytest.raises(ValueError):
        CartesianTrajectory.from_waypoint_array(np.zeros((3, 4)))


def test_cartesian_delta_shape():
    d = CartesianDelta(delta=[0.01, 0, 0, 0, 0, 0])
    assert d.delta.shape == (6,)


def test_execution_result_defaults():
    r = ExecutionResult()
    assert r.success and not r.clipped and r.stop_reason == "none"


@pytest.mark.parametrize("scale", [0.0, -0.1, 1.1, np.nan, np.inf])
def test_joint_trajectory_rejects_invalid_speed_scale(scale):
    with pytest.raises(ValueError, match="max_joint_speed_scale"):
        JointTrajectory(
            waypoints=[JointWaypoint(np.zeros(7), duration=0.1)],
            max_joint_speed_scale=scale,
        )


def test_joint_trajectory_rejects_unknown_interpolation():
    with pytest.raises(ValueError, match="interpolation"):
        JointTrajectory(
            waypoints=[JointWaypoint(np.zeros(7), duration=0.1)],
            interpolation="cubic",
        )


def test_joint_trajectory_preserves_legacy_positional_field_order():
    traj = JointTrajectory(
        [JointWaypoint(np.zeros(7), duration=0.1)],
        0.2,
        "linear",
        "tabletop_safe",
    )
    assert traj.max_joint_speed_scale == pytest.approx(0.2)
    assert traj.interpolation == "linear"
    assert traj.safety_profile == "tabletop_safe"
    assert traj.initial_positions is None
    assert traj.strict_timing is False
