import numpy as np
import pytest

from flexiv_control import (
    CartesianTrajectory,
    GripperCommand,
    LeaseError,
    Robot,
    RobotConfig,
)


def _robot():
    return Robot(RobotConfig(backend="fake", control_hz=200.0))


def test_connect_and_state():
    r = _robot()
    r.connect()
    s = r.get_state()
    assert s.q.shape == (7,)
    assert s.tcp_pose.shape == (7,)
    r.disconnect()


def test_execute_waypoint_chunk_tracks():
    r = _robot()
    with r:
        r.acquire_lease("test")
        r.start_cartesian_impedance()
        u = [[0.45, 0.0, 0.30, 1.0, 20],
             [0.50, 0.0, 0.28, 0.0, 20]]
        res = r.execute_cartesian_trajectory(CartesianTrajectory.from_waypoint_array(u))
        assert res.success
        # FakeBackend tracks perfectly -> tracking error tiny, ends near target
        assert res.path_tracking_error < 0.02
        assert np.allclose(res.final_state.tcp_position, [0.50, 0.0, 0.28], atol=1e-3)


def test_servo_delta_moves_tcp():
    r = _robot()
    with r:
        r.start_cartesian_impedance()
        before = r.get_state().tcp_position.copy()
        r.servo_cartesian_delta([0.01, 0.0, 0.0, 0, 0, 0], duration=0.05)
        after = r.get_state().tcp_position
        assert after[0] > before[0]


def test_gripper_command_logged():
    r = _robot()
    with r:
        r.start_cartesian_impedance()
        r.command_gripper(GripperCommand(width=0.02, force=15.0, grasp=True))
        assert r.get_state().gripper_width == pytest.approx(0.02)
        assert len(r.backend.gripper_log) >= 1


def test_move_joint_and_home():
    r = _robot()
    with r:
        r.acquire_lease("test")
        target = np.array([0.1, -0.5, 0.0, 1.4, 0.0, 0.8, 0.0])
        res = r.move_joint(target, duration=0.2)
        assert res.success
        assert np.allclose(r.get_state().q, target, atol=1e-2)
        r.home()
        assert np.allclose(r.get_state().q, r.cfg.q_home, atol=1e-2)


def test_lease_conflict():
    r = _robot()
    r.connect()
    r.acquire_lease("alice")
    with pytest.raises(LeaseError):
        r.acquire_lease("bob")
    r.release_lease()
    r.acquire_lease("bob")  # now ok
    r.disconnect()


def test_from_config_fake():
    r = Robot.from_config("fake")
    with r:
        assert r.cfg.backend == "fake"
        assert r.get_state().q.shape == (7,)


def test_workspace_clip_reported_during_chunk():
    # Drive far outside tabletop_safe workspace; expect clipping flagged.
    r = Robot(RobotConfig(backend="fake", control_hz=200.0))
    with r:
        r.start_cartesian_impedance()
        wp = CartesianTrajectory.from_waypoint_array([[2.0, 0.0, 0.30, 1.0, 20]])  # x way out of box
        res = r.execute_cartesian_trajectory(wp)
        assert res.clipped
