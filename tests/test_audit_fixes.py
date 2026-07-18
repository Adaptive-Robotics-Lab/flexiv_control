"""Regression tests for the audit-round fixes (frame handling, angular-speed
enforcement, LeRobot feature schema, and the joint-traj remote bridge)."""

from __future__ import annotations

import numpy as np
import pytest

from flexiv_control import transforms as T
from flexiv_control.safety import SafetyFilter, SafetyProfile
from flexiv_control.types import RobotState


def test_integrate_pose_tcp_frame_rotates_translation():
    # 90 deg about z; a +x delta in the TCP frame must come out as +y in base.
    pose = np.array([0, 0, 0, np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    base = T.integrate_pose(pose, [0.1, 0, 0, 0, 0, 0], frame="base")
    tcp = T.integrate_pose(pose, [0.1, 0, 0, 0, 0, 0], frame="tcp")
    assert np.allclose(base[:3], [0.1, 0, 0])
    assert np.allclose(tcp[:3], [0, 0.1, 0], atol=1e-6)


def test_integrate_pose_rejects_unknown_frame():
    with pytest.raises(ValueError):
        T.integrate_pose(np.array([0, 0, 0, 1, 0, 0, 0]), np.zeros(6), frame="world")


def test_quat_rotate_matches_known_rotation():
    q = T.rotvec_to_quat([0, 0, np.pi / 2])  # 90 deg about z
    assert np.allclose(T.quat_rotate(q, [1, 0, 0]), [0, 1, 0], atol=1e-6)


def test_angular_speed_is_enforced_on_command_stream():
    prof = SafetyProfile()
    dt = 0.01
    f = SafetyFilter(prof, dt)
    start = np.array([0, 0, 0, 1, 0, 0, 0.0])
    f.reset(pose=start)
    st = RobotState(tcp_pose=start.copy())
    big = T.rotvec_to_quat([0, 0, 2.0])  # 2 rad, far over max_angular_speed*dt
    res = f.filter_cartesian(np.concatenate([[0, 0, 0], big]), st)
    ang = T.quat_angle(start[3:7], res.pose[3:7])
    assert res.clipped
    assert ang <= prof.max_angular_speed * dt + 1e-6


def test_lerobot_adapter_feature_schema_is_flat_and_consistent():
    from flexiv_control import Robot, RobotConfig
    from flexiv_control.adapters.lerobot_robot import LeRobotFlexivAdapter

    ad = LeRobotFlexivAdapter(robot=Robot(RobotConfig(backend="fake")))
    ad.robot.connect()
    feats = ad.observation_features
    assert all(v is float for v in feats.values())  # flat scalar features
    assert all(v is float for v in ad.action_features.values())
    obs = ad.get_observation()
    assert set(obs.keys()) == set(feats.keys())
    assert all(isinstance(v, float) for v in obs.values())
    act = ad.send_action(np.zeros(7))
    assert set(act.keys()) == set(ad.action_features.keys())
    assert ad.is_calibrated is True


def test_remote_joint_chunk_and_disconnect_alias():
    import time

    from flexiv_control import RobotConfig
    from flexiv_control.trajectory import JointTrajectory, JointWaypoint
    from flexiv_control.client import RemoteRobot
    from flexiv_control.server import FlexivControlServer

    # ephemeral-ish port; serve in a background thread
    srv = FlexivControlServer(config=RobotConfig(backend="fake"), port=8801, lease_ttl=5.0)
    srv.serve_in_thread()
    time.sleep(0.3)
    try:
        rr = RemoteRobot("127.0.0.1", 8801, owner="tester")
        rr.connect()
        rr.acquire_lease("tester")  # positional owner must not raise
        rr.start_joint_impedance()
        traj = JointTrajectory(waypoints=[JointWaypoint(positions=np.zeros(7), duration=0.05)])
        res = rr.execute_joint_trajectory(traj)
        assert res.success
        rr.disconnect()  # alias for close()
    finally:
        srv.shutdown()


def test_gym_reset_seeding_is_reproducible_and_isolated():
    pytest.importorskip("gymnasium", reason="gymnasium not installed")
    import numpy as _np

    from flexiv_control import RobotConfig
    from flexiv_control.envs import FlexivRealEnv

    env = FlexivRealEnv(config=RobotConfig(backend="fake"))
    env.reset(seed=42)
    a = env.np_random.random()
    env.reset(seed=42)
    b = env.np_random.random()
    assert a == b  # same seed -> same env RNG draw (Gymnasium contract)
    # global RNG must be untouched by reset(seed=...)
    _np.random.seed(0)
    before = _np.random.random()
    _np.random.seed(0)
    env.reset(seed=123)
    after = _np.random.random()
    assert before == after
    env.close()
