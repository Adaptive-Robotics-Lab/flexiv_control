import numpy as np
import pytest

from flexiv_control import (
    CartesianTrajectory,
    CartesianWaypoint,
    JointTrajectory,
    JointWaypoint,
)
from flexiv_control import transforms as T
from flexiv_control.interpolation import (
    CartesianTrajectoryInterpolator,
    JointTrajectoryInterpolator,
)


def test_quat_mul_identity():
    q = T.quat_normalize([0.3, 0.1, -0.2, 0.5])
    ident = np.array([1.0, 0, 0, 0])
    assert np.allclose(T.quat_mul(ident, q), q)


def test_quat_conj_inverse():
    q = T.quat_normalize([0.3, 0.1, -0.2, 0.5])
    prod = T.quat_mul(q, T.quat_conj(q))
    assert np.allclose(prod, [1, 0, 0, 0], atol=1e-9)


def test_slerp_endpoints():
    a = np.array([1.0, 0, 0, 0])
    b = T.rotvec_to_quat([0, 0, np.pi / 2])
    assert np.allclose(T.quat_slerp(a, b, 0.0), a)
    assert np.allclose(T.quat_slerp(a, b, 1.0), T.quat_normalize(b))


def test_rotvec_roundtrip():
    rv = np.array([0.1, -0.2, 0.3])
    q = T.rotvec_to_quat(rv)
    assert np.allclose(T.quat_to_rotvec(q), rv, atol=1e-9)


def test_integrate_pose_translation():
    pose = np.array([0.4, 0.0, 0.3, 1, 0, 0, 0], float)
    out = T.integrate_pose(pose, [0.05, -0.02, 0.01, 0, 0, 0])
    assert np.allclose(out[:3], [0.45, -0.02, 0.31])
    assert np.allclose(out[3:7], [1, 0, 0, 0])


def test_interpolator_tick_count_and_endpoint():
    start = np.array([0.4, 0.0, 0.3, 1, 0, 0, 0], float)
    wp = CartesianWaypoint(position=[0.5, 0.0, 0.3], n_frames=10)
    traj = CartesianTrajectory(waypoints=[wp])
    interp = CartesianTrajectoryInterpolator(traj, start, control_hz=100.0)
    setpoints = interp.setpoints()
    assert len(setpoints) == 10  # n_frames at the matching rate
    last_pose, _ = setpoints[-1]
    assert np.allclose(last_pose[:3], [0.5, 0.0, 0.3], atol=1e-6)


def test_interpolator_holds_orientation():
    start = np.array([0.4, 0.0, 0.3, 1, 0, 0, 0], float)
    wp = CartesianWaypoint(position=[0.5, 0.0, 0.3], duration=0.05)  # quaternion None -> hold
    interp = CartesianTrajectoryInterpolator(CartesianTrajectory(waypoints=[wp]), start, 100.0)
    for pose, _ in interp:
        assert np.allclose(pose[3:7], [1, 0, 0, 0], atol=1e-9)


def test_joint_linear_interpolation_uses_cumulative_tick_rounding():
    start = np.zeros(7)
    first = np.ones(7)
    second = np.full(7, 2.0)
    traj = JointTrajectory(
        waypoints=[
            JointWaypoint(positions=first, duration=0.1632),
            JointWaypoint(positions=second, duration=0.1632),
        ],
        interpolation="linear",
    )
    interp = JointTrajectoryInterpolator(traj, start, control_hz=100.0)
    setpoints = interp.setpoints()

    assert interp.nominal_segment_ticks == [16, 17]
    assert interp.nominal_total_ticks == 33
    assert interp.scheduled_segment_ticks == [16, 17]
    assert len(setpoints) == 33
    assert np.allclose(setpoints[0], first / 16.0)
    assert np.allclose(setpoints[15], first)
    assert np.allclose(setpoints[16], first + (second - first) / 17.0)
    assert np.allclose(setpoints[-1], second)


def test_joint_cosine_remains_the_backward_compatible_default():
    traj = JointTrajectory(
        waypoints=[JointWaypoint(positions=np.ones(7), n_frames=4)]
    )
    setpoints = JointTrajectoryInterpolator(
        traj,
        np.zeros(7),
        control_hz=100.0,
    ).setpoints()

    assert traj.interpolation == "cosine"
    assert np.allclose(setpoints[0], np.full(7, 0.1464466094067262))
    assert np.allclose(setpoints[-1], np.ones(7))


def test_joint_speed_limit_is_enforced_per_joint():
    target = np.array([0.5, 0.5, 0, 0, 0, 0, 0], float)
    limits = np.array([1.0, 0.25, 1, 1, 1, 1, 1], float)
    traj = JointTrajectory(
        waypoints=[
            JointWaypoint(positions=target, duration=0.1)
        ],
        interpolation="linear",
    )
    interp = JointTrajectoryInterpolator(
        traj,
        np.zeros(7),
        control_hz=100.0,
        max_joint_speed=limits,
    )
    points = interp.setpoints()

    # Joint 2 is limiting: 0.5 rad / 0.25 rad/s = 2 seconds.
    assert interp.scheduled_total_ticks == 200
    stream = np.vstack([np.zeros(7), points])
    observed = np.max(np.abs(np.diff(stream, axis=0)), axis=0) * 100.0
    assert np.all(observed <= limits + 1e-12)


def test_joint_speed_vector_validation_is_fail_closed():
    traj = JointTrajectory(
        waypoints=[JointWaypoint(positions=np.zeros(7), duration=0.1)]
    )
    with pytest.raises(ValueError, match="match start_q shape"):
        JointTrajectoryInterpolator(
            traj,
            np.zeros(7),
            control_hz=100.0,
            max_joint_speed=np.ones(6),
        )
    with pytest.raises(ValueError, match="finite and > 0"):
        JointTrajectoryInterpolator(
            traj,
            np.zeros(7),
            control_hz=100.0,
            max_joint_speed=np.zeros(7),
        )
