"""v0.1.7 features: gripper close-intent tracking gate + payload wrench allowance.

Both were queued from a real hardware failure (2026-07-07): a descend stalled
on unmodeled contact BELOW the wrench cap, the close fired at the unplanned
height (rim pinch), and the resulting static wrench instant-stopped every
subsequent held-transport traj at its first tick.
"""
import numpy as np
import pytest

from flexiv_control import CartesianTrajectory, GripperCommand, Robot, RobotConfig
from flexiv_control.server import protocol as P


def _robot():
    return Robot(RobotConfig(backend="fake", control_hz=200.0))


def _freeze_arm(robot, frozen_pos):
    """Make read_state report a stalled arm (position frozen) regardless of
    what is being commanded -- the impedance-deflection stall signature."""
    original = robot.backend.read_state

    def stalled():
        s = original()
        s.tcp_pose = s.tcp_pose.copy()
        s.tcp_pose[:3] = np.asarray(frozen_pos, float)
        return s

    robot.backend.read_state = stalled


def _grasp_chunk(*, gate=None, descend_dz=0.03):
    """hover -> descend -> close(Move) -> lift(Grasp), the acquire anatomy."""
    top = [0.45, 0.0, 0.30]
    bottom = [0.45, 0.0, 0.30 - descend_dz]
    u = [
        [*top, 0.085, 20, 0],     # hover, open
        [*bottom, 0.085, 20, 0],  # descend, still open
        [*bottom, 0.030, 20, 0],  # detector Move close
        [*top, 0.030, 40, 1],     # lift with force-grasp
    ]
    wps = CartesianTrajectory.from_waypoint_array(
        [row[:5] for row in u]).waypoints
    for wp, row in zip(wps, u):
        wp.gripper = GripperCommand(width=row[3], force=row[4], grasp=bool(row[5]))
        wp.duration = 0.05
    return CartesianTrajectory(waypoints=wps, grip_tracking_gate_m=gate)


def _closes(robot):
    return [g for g in robot.backend.gripper_log if g.grasp or g.width < 0.08]


def test_close_fires_when_tracking_healthy():
    r = _robot()
    with r:
        r.acquire_lease("t")
        r.start_cartesian_impedance()
        res = r.execute_cartesian_trajectory(_grasp_chunk(gate=0.012))
        assert res.success
        assert "close_aborted" not in res.log
        assert len(_closes(r)) >= 2  # detector Move + force Grasp


def test_stalled_descend_aborts_close_and_stays_sticky():
    r = _robot()
    with r:
        r.acquire_lease("t")
        r.start_cartesian_impedance()
        start = r.get_state().tcp_position.copy()
        _freeze_arm(r, start)  # arm never moves; command marches 30mm down
        res = r.execute_cartesian_trajectory(_grasp_chunk(gate=0.012))
        aborted = res.log.get("close_aborted")
        assert aborted is not None
        assert aborted["tracking_error_m"] > 0.012
        # STICKY: neither the detector Move nor the later force Grasp fired.
        assert len(_closes(r)) == 0
        # Opens are never gated (the hover/descend open commands still pass).
        assert any(g.width >= 0.08 and not g.grasp for g in r.backend.gripper_log)


def test_gate_disabled_by_default_preserves_old_behavior():
    r = _robot()
    with r:
        r.acquire_lease("t")
        r.start_cartesian_impedance()
        start = r.get_state().tcp_position.copy()
        _freeze_arm(r, start)
        res = r.execute_cartesian_trajectory(_grasp_chunk(gate=None))
        assert "close_aborted" not in res.log
        assert len(_closes(r)) >= 2


def _static_wrench(robot, wrench):
    original = robot.backend.read_state

    def biased():
        s = original()
        s.wrench = np.asarray(wrench, float)
        return s

    robot.backend.read_state = biased


def _transport_chunk(allowance=None):
    c = CartesianTrajectory.from_waypoint_array(
        [[0.45, 0.0, 0.30, 0.03, 40], [0.55, 0.0, 0.30, 0.03, 40]],
        contact_wrench_allowance=allowance)
    for wp in c.waypoints:
        wp.duration = 0.05
    return c


def test_payload_wrench_stops_chunk_without_allowance():
    r = _robot()
    with r:
        r.acquire_lease("t")
        r.start_cartesian_impedance()
        _static_wrench(r, [42, 0, 0, 0, 0, 0])  # payload bias > profile 40N
        res = r.execute_cartesian_trajectory(_transport_chunk())
        assert not res.success
        assert res.stop_reason == "contact_wrench"


def test_allowance_needs_profile_grant():
    r = _robot()
    with r:
        r.acquire_lease("t")
        r.start_cartesian_impedance()
        _static_wrench(r, [42, 0, 0, 0, 0, 0])
        # Default profile grants ZERO allowance -> request is clamped away.
        res = r.execute_cartesian_trajectory(
            _transport_chunk(allowance=[15, 15, 15, 3, 3, 3]))
        assert not res.success
        assert res.stop_reason == "contact_wrench"
        assert "contact_wrench_allowance" not in res.log


def test_granted_allowance_lets_held_transport_run():
    r = _robot()
    with r:
        r.acquire_lease("t")
        r.start_cartesian_impedance()
        r.profile.max_wrench_allowance = np.array([15, 15, 15, 3, 3, 3], float)
        _static_wrench(r, [42, 0, 0, 0, 0, 0])  # above 40, below 40+15
        res = r.execute_cartesian_trajectory(
            _transport_chunk(allowance=[15, 15, 15, 3, 3, 3]))
        assert res.success, res.stop_reason
        assert res.log["contact_wrench_allowance"] == [15, 15, 15, 3, 3, 3]
        # Firmware guard re-armed at the profile cap after the traj.
        assert np.allclose(r.backend._max_wrench, r.profile.max_contact_wrench)


def test_allowance_request_clamped_to_grant():
    r = _robot()
    with r:
        r.acquire_lease("t")
        r.start_cartesian_impedance()
        r.profile.max_wrench_allowance = np.array([15, 15, 15, 3, 3, 3], float)
        _static_wrench(r, [60, 0, 0, 0, 0, 0])  # above 40+15 even with grant
        res = r.execute_cartesian_trajectory(
            _transport_chunk(allowance=[100, 100, 100, 50, 50, 50]))
        assert not res.success
        assert res.stop_reason == "contact_wrench"
        assert res.log["contact_wrench_allowance"] == [15, 15, 15, 3, 3, 3]


def test_negative_allowance_rejected():
    with pytest.raises(ValueError):
        _transport_chunk(allowance=[-1, 0, 0, 0, 0, 0])


def test_new_fields_roundtrip_the_wire():
    c = _grasp_chunk(gate=0.012)
    c.contact_wrench_allowance = np.array([10, 10, 10, 2, 2, 2], float)
    d = P.trajectory_to_dict(c)
    c2 = P.trajectory_from_dict(d)
    assert c2.grip_tracking_gate_m == pytest.approx(0.012)
    assert np.allclose(c2.contact_wrench_allowance, [10, 10, 10, 2, 2, 2])
    # And None defaults survive too (old clients / plain trajs).
    plain = P.trajectory_from_dict(P.trajectory_to_dict(_transport_chunk()))
    assert plain.grip_tracking_gate_m is None
    assert plain.contact_wrench_allowance is None


def test_sustain_never_gated_after_close_was_issued():
    """Adversarial-review CRITICAL: once the detector Move actually fired (the
    object may be between the fingers), the follow-up Grasp SUSTAIN must NOT
    be gated -- 40 N clamp reaction can deflect the impedance arm past the
    gate right after a legitimate close, and skipping the sustain would leave
    only a stalled position-hold (the filmed slide-out) while mislabeling a
    physical grasp as close_aborted (three concrete paths to a drop)."""
    r = _robot()
    with r:
        r.acquire_lease("t")
        r.start_cartesian_impedance()
        original = r.backend.read_state

        def deflected_after_close():
            s = original()
            # Once ANY close command reached the gripper, report the arm
            # deflected 30mm from wherever it is commanded (clamp reaction).
            if any(g.grasp or g.width < 0.08 for g in r.backend.gripper_log):
                s.tcp_pose = s.tcp_pose.copy()
                s.tcp_pose[2] += 0.03
            return s

        r.backend.read_state = deflected_after_close
        res = r.execute_cartesian_trajectory(_grasp_chunk(gate=0.012))
        assert "close_aborted" not in res.log
        grasps = [g for g in r.backend.gripper_log if g.grasp]
        moves_close = [g for g in r.backend.gripper_log
                       if not g.grasp and g.width < 0.08]
        assert len(moves_close) >= 1   # detector Move fired (healthy descend)
        assert len(grasps) >= 1        # force-closure sustain NOT gated


def test_transit_lag_converges_and_close_fires_settled():
    """Tonight's live failure: at the descend->close segment boundary the arm
    trails its command by ~20-25mm of NORMAL dynamic lag (measured 23mm on the
    real Rizon), which sat inside the old instantaneous gate and aborted every
    healthy grasp. The gate now samples AFTER a settle window: a pure-delay
    arm (measured = command from N ticks ago) converges once the command
    holds, so the deferred close must FIRE."""
    r = _robot()
    with r:
        r.acquire_lease("t")
        r.start_cartesian_impedance()
        backend = r.backend
        original_read = backend.read_state
        original_stream = backend.stream_cartesian
        cmd_hist = []

        def lagged_stream(pose, wrench=None):
            cmd_hist.append(np.asarray(pose, float).copy())
            return original_stream(pose, wrench=wrench)

        def lagged_read():
            s = original_read()
            if len(cmd_hist) > 15:  # 15-tick transport delay (~23mm at speed)
                s.tcp_pose = cmd_hist[-15].copy()
            elif cmd_hist:
                s.tcp_pose = cmd_hist[0].copy()
            return s

        backend.stream_cartesian = lagged_stream
        backend.read_state = lagged_read
        try:
            # Long close-hold segment (0.5 s = 100 ticks at 200 Hz) so the
            # settle window (min(70, 50) = 50 ticks) clears the 15-tick lag.
            top = [0.45, 0.0, 0.30]
            bottom = [0.45, 0.0, 0.26]
            u = [[*top, 0.085, 20, 0], [*bottom, 0.085, 20, 0],
                 [*bottom, 0.030, 20, 0], [*top, 0.030, 40, 1]]
            wps = CartesianTrajectory.from_waypoint_array(
                [row[:5] for row in u]).waypoints
            for wp, row in zip(wps, u):
                wp.gripper = GripperCommand(width=row[3], force=row[4],
                                            grasp=bool(row[5]))
                wp.duration = 0.5
            res = r.execute_cartesian_trajectory(
                CartesianTrajectory(waypoints=wps, grip_tracking_gate_m=0.012))
        finally:
            backend.read_state = original_read
            backend.stream_cartesian = original_stream
        assert res.success
        assert "close_aborted" not in res.log, res.log["close_aborted"]
        assert len(_closes(r)) >= 2  # detector Move AND force Grasp both fired
