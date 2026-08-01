import numpy as np

from flexiv_control import RobotState, SafetyFilter, SafetyProfile, StopReason


def _state(pos):
    s = RobotState()
    s.tcp_pose = np.array([pos[0], pos[1], pos[2], 1, 0, 0, 0], float)
    return s


def test_workspace_clip():
    p = SafetyProfile()  # tabletop defaults: x in [0.25, 0.75]
    f = SafetyFilter(p, control_dt=0.01)
    s = _state([0.74, 0.0, 0.30])
    # ask to go past x-max; pose-jump small enough not to dominate
    res = f.filter_cartesian(np.array([0.80, 0.0, 0.30, 1, 0, 0, 0]), s)
    assert res.ok and res.clipped
    assert res.pose[0] <= 0.75 + 1e-9


def test_pose_jump_limit():
    p = SafetyProfile(max_pose_jump_linear=0.01)
    f = SafetyFilter(p, control_dt=0.01)
    s = _state([0.40, 0.0, 0.30])
    res = f.filter_cartesian(np.array([0.60, 0.0, 0.30, 1, 0, 0, 0]), s)
    step = np.linalg.norm(res.pose[:3] - s.tcp_pose[:3])
    assert res.clipped and step <= 0.01 + 1e-9


def test_contact_wrench_stop():
    p = SafetyProfile()
    f = SafetyFilter(p, control_dt=0.01)
    s = _state([0.40, 0.0, 0.30])
    s.wrench = np.array([100, 0, 0, 0, 0, 0], float)  # exceeds 40 N
    res = f.filter_cartesian(np.array([0.41, 0.0, 0.30, 1, 0, 0, 0]), s)
    assert not res.ok and res.reason == StopReason.CONTACT_WRENCH


def test_joint_clip_to_limits():
    p = SafetyProfile()
    f = SafetyFilter(p, control_dt=0.01)
    s = RobotState()
    s.q = np.zeros(7)
    way_over = np.full(7, 10.0)
    res = f.filter_joint(way_over, s)
    assert res.ok and res.clipped
    assert np.all(res.q <= p.joint_upper - p.joint_margin_rad + 1e-9)


def test_joint_speed_filter_uses_per_joint_runtime_limits():
    p = SafetyProfile(max_joint_speed_scale=0.5)
    limits = np.array([2.0, 0.2, 2, 2, 2, 2, 2], float)
    f = SafetyFilter(
        p,
        control_dt=0.01,
        joint_velocity_max=limits,
    )
    s = RobotState()
    s.q = np.zeros(7)
    res = f.filter_joint(np.full(7, 0.1), s)
    expected = limits * 0.5 * 0.01
    assert np.allclose(res.q, expected)


def test_joint_speed_filter_rejects_invalid_runtime_limits():
    with np.testing.assert_raises_regex(
        ValueError,
        "joint_velocity_max",
    ):
        SafetyFilter(
            SafetyProfile(),
            control_dt=0.01,
            joint_velocity_max=np.ones(6),
        )


def test_command_age_watchdog():
    p = SafetyProfile(command_timeout_ms=100)
    f = SafetyFilter(p, control_dt=0.01)
    assert f.check_command_age(50).ok
    assert not f.check_command_age(150).ok


def test_profile_from_dict():
    d = {
        "name": "x",
        "workspace": {"x": [0.1, 0.2], "y": [-0.3, 0.3], "z": [0.0, 0.5]},
        "tcp_limits": {"max_linear_speed": 0.1},
        "contact": {"max_wrench": [10, 10, 10, 1, 1, 1]},
        "watchdog": {"command_timeout_ms": 50},
    }
    p = SafetyProfile.from_dict(d)
    assert p.name == "x"
    assert p.ws_x == (0.1, 0.2)
    assert p.max_linear_speed == 0.1
    assert p.command_timeout_ms == 50
    assert np.allclose(p.max_contact_wrench, [10, 10, 10, 1, 1, 1])
