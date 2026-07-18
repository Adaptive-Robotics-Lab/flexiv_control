"""Tests for the ActAhead friction-audit fixes (v0.1.1).

Each test pins one of the audited failure modes:
* move_joint(max_joint_speed=...) -- the home-restore call shape that used to
  TypeError and silently break the exit ritual;
* execute_cartesian_trajectory auto-ensures the Cartesian mode (and the FakeBackend
  now REJECTS streaming in IDLE, so dry runs reveal mode-sequencing bugs);
* the traj kinematic envelope tightens (never relaxes) the active profile;
* traj.safety_profile is verified against the active profile;
* workspace_action: reject protective-stops instead of silently clipping;
* SafetyProfile.validate_trajectory / to_config_dict round-trip (the get_safety_profile RPC);
* blocking gripper, home() with gripper_home_width, go_home_safe;
* cooperative cancel (request_stop) mid-traj;
* Lease.hold keeps an in-flight RPC's lease alive past the TTL;
* from_topdown_array / frames_hz constructors; ExecutionResult.summary;
  raise_on_stop.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from flexiv_control import (
    CartesianTrajectory,
    CartesianWaypoint,
    TrajectoryStoppedError,
    GripperCommand,
    Robot,
    RobotConfig,
    SafetyProfile,
)
from flexiv_control import transforms as T
from flexiv_control.server.lease import Lease, LeaseError


def _robot(**profile_overrides) -> Robot:
    r = Robot(RobotConfig(backend="fake"))
    for k, v in profile_overrides.items():
        setattr(r.profile, k, v)
    r.connect()
    r.acquire_lease("test")
    return r


def _traj(positions, duration=0.05, **kwargs) -> CartesianTrajectory:
    wps = [
        CartesianWaypoint(position=np.asarray(p, float), quaternion=None, duration=duration)
        for p in positions
    ]
    return CartesianTrajectory(waypoints=wps, **kwargs)


# ---------------------------------------------------------------------------
# move_joint speed parameterization (the home-restore bug)
# ---------------------------------------------------------------------------
def test_move_joint_accepts_max_joint_speed():
    r = _robot()
    q0 = r.get_state().q.copy()
    q1 = q0 + 0.1  # stays inside the joint-limit margin on every joint
    result = r.move_joint(q1, max_joint_speed=0.3)
    assert result.success
    np.testing.assert_allclose(r.get_state().q, q1, atol=1e-6)


def test_joint_move_duration_derivation():
    q0 = np.zeros(7)
    q1 = np.zeros(7)
    q1[3] = 0.6
    d = Robot._joint_move_duration(q0, q1, duration=None, max_joint_speed=0.3)
    assert d == pytest.approx(2.0)
    # explicit duration wins; default applies when neither is given
    assert Robot._joint_move_duration(q0, q1, duration=1.2, max_joint_speed=0.3) == 1.2
    assert Robot._joint_move_duration(q0, q1, duration=None, max_joint_speed=None) == 3.0
    # the floor protects tiny corrections from a near-zero duration
    assert Robot._joint_move_duration(q0, q0, duration=None, max_joint_speed=0.3) == 1.0


# ---------------------------------------------------------------------------
# mode auto-ensure + strict FakeBackend
# ---------------------------------------------------------------------------
def test_execute_chunk_autostarts_cartesian_mode():
    r = _robot()
    # No start_cartesian_impedance() anywhere -- this used to pass on fake and
    # fault on the first real traj.
    res = r.execute_cartesian_trajectory(_traj([[0.45, 0.0, 0.31]]))
    assert res.success
    assert res.log.get("mode_autostarted") is True
    from flexiv_control.types import ControlMode

    assert ControlMode.NRT_CARTESIAN_MOTION_FORCE in r.backend.mode_log


def test_fake_backend_rejects_streaming_in_idle():
    r = _robot()
    with pytest.raises(RuntimeError, match="Cartesian"):
        r.backend.stream_cartesian(np.array([0.45, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]))
    with pytest.raises(RuntimeError, match="joint"):
        r.backend.stream_joint(np.zeros(7))


# ---------------------------------------------------------------------------
# envelope: tightening-only
# ---------------------------------------------------------------------------
def test_chunk_speed_cap_tightens_profile():
    r = _robot()
    target = [[0.55, 0.0, 0.30]]  # 10 cm from the fake start pose
    slow = r.execute_cartesian_trajectory(_traj(target, duration=0.05, max_tcp_linear_speed=0.02))
    n_slow = len(r.backend.cartesian_log)
    r.backend.cartesian_log.clear()
    r2 = _robot()
    fast = r2.execute_cartesian_trajectory(_traj(target, duration=0.05, max_tcp_linear_speed=10.0))
    n_fast = len(r2.backend.cartesian_log)
    assert slow.success and fast.success
    # the traj's lower cap stretches the segment to many more ticks; the
    # higher-than-profile cap is clamped to the profile (never relaxed)
    assert n_slow > 4 * n_fast
    assert slow.log["linear_speed_cap"] == pytest.approx(0.02)
    assert fast.log["linear_speed_cap"] == pytest.approx(r2.profile.max_linear_speed)


# ---------------------------------------------------------------------------
# safety_profile verification
# ---------------------------------------------------------------------------
def test_chunk_profile_mismatch_raises():
    r = _robot()
    with pytest.raises(ValueError, match="safety profile"):
        r.execute_cartesian_trajectory(_traj([[0.45, 0.0, 0.31]], safety_profile="free_space_fast"))


def test_chunk_profile_match_and_empty_ok():
    r = _robot()
    ok1 = r.execute_cartesian_trajectory(_traj([[0.45, 0.0, 0.31]], safety_profile=""))
    ok2 = r.execute_cartesian_trajectory(
        _traj([[0.45, 0.0, 0.32]], safety_profile=r.profile.name)
    )
    assert ok1.success and ok2.success
    assert ok2.log["requested_profile"] == r.profile.name
    assert ok2.log["active_profile"] == r.profile.name


# ---------------------------------------------------------------------------
# workspace reject + validate_trajectory + profile round-trip
# ---------------------------------------------------------------------------
def test_workspace_reject_stops_instead_of_clipping():
    r = _robot(workspace_action="reject")
    res = r.execute_cartesian_trajectory(_traj([[5.0, 0.0, 0.31]]))
    assert not res.success
    assert res.stop_reason == "workspace_limit"


def test_workspace_clip_default_still_clips():
    r = _robot()
    res = r.execute_cartesian_trajectory(_traj([[5.0, 0.0, 0.31]]))
    assert res.success
    assert res.clipped


def test_validate_chunk_lists_violations():
    p = SafetyProfile()
    traj = _traj([[0.45, 0.0, 0.31], [5.0, 0.0, 0.31]], duration=1.0)
    problems = p.validate_trajectory(traj)
    assert any("waypoint 1" in m and "outside" in m for m in problems)
    # the 4.55 m hop in 1 s also gets a time-stretch note
    assert any("time-stretched" in m for m in problems)
    assert p.validate_trajectory(_traj([[0.45, 0.0, 0.31]])) == []


def test_profile_config_roundtrip():
    p = SafetyProfile(name="x", workspace_action="reject")
    q = SafetyProfile.from_dict(p.to_config_dict())
    assert q.name == "x"
    assert q.workspace_action == "reject"
    assert q.ws_x == tuple(p.ws_x)
    assert q.max_linear_speed == p.max_linear_speed
    np.testing.assert_allclose(q.max_contact_wrench, p.max_contact_wrench)


# ---------------------------------------------------------------------------
# gripper / home / go_home_safe
# ---------------------------------------------------------------------------
def test_command_gripper_wait_returns_width():
    r = _robot()
    w = r.command_gripper(GripperCommand(width=0.05), wait=True, timeout=2.0)
    assert w == pytest.approx(0.05, abs=1e-3)  # float, matching RemoteRobot
    assert r.command_gripper(GripperCommand(width=0.08)) is None  # fire-and-forget


def test_home_restores_joints_and_gripper():
    cfg = RobotConfig(backend="fake", gripper_home_width=0.085)
    r = Robot(cfg)
    r.connect()
    r.acquire_lease("test")
    r.command_gripper(GripperCommand(width=0.02))
    r.home()
    s = r.get_state()
    np.testing.assert_allclose(s.q, cfg.q_home, atol=1e-6)
    assert s.gripper_width == pytest.approx(0.085, abs=1e-3)


def test_go_home_safe_full_ritual():
    cfg = RobotConfig(backend="fake", gripper_home_width=0.085)
    r = Robot(cfg)
    r.connect()
    r.acquire_lease("test")
    r.start_cartesian_impedance()
    r.command_gripper(GripperCommand(width=0.01, grasp=True))
    result = r.go_home_safe(lift_m=0.05, max_joint_speed=0.5)
    assert result.success
    assert "lift" in result.log and "gripper" in result.log
    s = r.get_state()
    np.testing.assert_allclose(s.q, cfg.q_home, atol=1e-6)
    assert s.gripper_width == pytest.approx(0.085, abs=1e-3)


def test_go_home_safe_explicit_targets():
    r = _robot()
    q_home = np.array([0.1, -0.6, 0.0, 1.5, 0.0, 0.8, 0.05])
    result = r.go_home_safe(q_home=q_home, open_gripper_width=0.07, lift_m=0.02)
    assert result.success
    np.testing.assert_allclose(r.get_state().q, q_home, atol=1e-6)
    assert r.get_state().gripper_width == pytest.approx(0.07, abs=1e-3)


# ---------------------------------------------------------------------------
# cooperative cancel
# ---------------------------------------------------------------------------
def test_request_stop_cancels_mid_chunk():
    r = _robot()
    traj = _traj([[0.55, 0.0, 0.30]], duration=3.0)
    done: dict = {}

    def _run():
        done["result"] = r.execute_cartesian_trajectory(traj)

    t = threading.Thread(target=_run)
    t.start()
    time.sleep(0.3)
    r.request_stop()
    t.join(timeout=5.0)
    assert not t.is_alive()
    res = done["result"]
    assert not res.success
    assert res.stop_reason == "user"
    assert res.executed_duration < 2.0


def test_pending_cancel_aborts_next_chunk():
    """A stop issued between trajs must not be silently erased: the next
    traj aborts at entry (consume-on-abort) instead of running."""
    r = _robot()
    r.request_stop()
    res = r.execute_cartesian_trajectory(_traj([[0.55, 0.0, 0.30]], duration=1.0))
    assert not res.success
    assert res.stop_reason == "user"
    assert res.log.get("aborted_at_entry") is True
    assert len(r.backend.cartesian_log) == 0  # nothing was streamed
    # the cancel was consumed: the following traj runs normally
    ok = r.execute_cartesian_trajectory(_traj([[0.46, 0.0, 0.30]]))
    assert ok.success


def test_backend_fault_stops_chunk():
    r = _robot()
    r.backend._fault = True
    res = r.move_joint(r.get_state().q + 0.05, max_joint_speed=0.5)
    assert not res.success
    assert res.stop_reason == "backend_fault"


def test_record_trajectory_and_stopped_at_waypoint():
    r = _robot()
    res = r.execute_cartesian_trajectory(_traj([[0.46, 0.0, 0.30]], duration=0.1), record=True)
    traj = res.log.get("trajectory")
    assert traj and len(traj[0]) == 1 + 7 + 7 + 6  # t + pose_cmd + pose_meas + wrench
    r2 = _robot(workspace_action="reject")
    bad = r2.execute_cartesian_trajectory(
        _traj([[0.46, 0.0, 0.30], [5.0, 0.0, 0.30]], duration=0.1)
    )
    assert not bad.success
    assert bad.log.get("stopped_at_waypoint") == 1


def test_frames_hz_must_be_positive():
    u = [[0.5, 0.0, 0.2, 1.0, 24]]
    with pytest.raises(ValueError, match="frames_hz"):
        CartesianTrajectory.from_waypoint_array(u, frames_hz=0.0)
    with pytest.raises(ValueError, match="frames_hz"):
        CartesianTrajectory.from_topdown_array([[0.5, 0.0, 0.2, 0.0, 1.0, 24]], frames_hz=-1.0)


# ---------------------------------------------------------------------------
# lease hold
# ---------------------------------------------------------------------------
def test_lease_hold_outlives_ttl():
    lease = Lease(ttl_seconds=0.1)
    lease.acquire("a")
    with lease.hold("a"):
        time.sleep(0.25)  # well past the TTL
        assert lease.owner == "a"
        with pytest.raises(LeaseError):
            lease.acquire("b")  # cannot steal an in-flight owner's lease
    # after the hold the expiry was refreshed
    assert lease.owner == "a"
    lease.release("a")


def test_lease_expires_without_hold():
    lease = Lease(ttl_seconds=0.05)
    lease.acquire("a")
    time.sleep(0.15)
    assert lease.owner == ""
    lease.acquire("b")  # free after expiry


# ---------------------------------------------------------------------------
# constructors / summary / raise_on_stop
# ---------------------------------------------------------------------------
def test_from_topdown_array_composes_yaw():
    yaw = 0.7
    u = [[0.5, 0.1, 0.2, yaw, 1.0, 20]]
    traj = CartesianTrajectory.from_topdown_array(u, gripper_span=0.10)
    expected = T.top_down_quat(yaw)
    np.testing.assert_allclose(traj.waypoints[0].quaternion, expected, atol=1e-9)
    assert traj.waypoints[0].gripper.width == pytest.approx(0.10)


def test_frames_hz_converts_to_duration():
    u = [[0.5, 0.0, 0.2, 1.0, 24]]
    traj = CartesianTrajectory.from_waypoint_array(u, frames_hz=12.0)
    assert traj.waypoints[0].duration == pytest.approx(2.0)
    assert traj.waypoints[0].n_frames is None
    td = CartesianTrajectory.from_topdown_array([[0.5, 0.0, 0.2, 0.0, 1.0, 24]], frames_hz=12.0)
    assert td.waypoints[0].duration == pytest.approx(2.0)


def test_execution_result_summary():
    r = _robot()
    res = r.execute_cartesian_trajectory(_traj([[0.45, 0.0, 0.31]]))
    s = res.summary()
    assert "ok" in s and "stop=none" in s and "grip=" in s


def test_raise_on_stop():
    r = _robot(workspace_action="reject")
    with pytest.raises(TrajectoryStoppedError) as ei:
        r.execute_cartesian_trajectory(_traj([[5.0, 0.0, 0.31]]), raise_on_stop=True)
    assert ei.value.result.stop_reason == "workspace_limit"


# ---------------------------------------------------------------------------
# config: gripper_home_width loads from YAML; lab profile is the pick-place one
# ---------------------------------------------------------------------------
def test_actahead_lab_config_loads_gripper_home_width():
    cfg = RobotConfig.load("rizon4s_actahead_lab")
    assert cfg.gripper_home_width == pytest.approx(0.085)
    # j7 sign matches the read-only RDK probe (lab_home_posture.json)
    assert cfg.q_home[6] == pytest.approx(0.003576, abs=1e-6)


def test_actahead_lab_profile_is_pickplace_envelope():
    from flexiv_control import load_safety_profile

    p = load_safety_profile("actahead_lab")
    assert p.workspace_action == "reject"
    assert p.ws_x[1] >= 0.95 - 1e-9   # covers the lab scene at x ~ 0.78
    assert p.max_linear_speed >= 0.12  # planner traj caps bind, not the profile
