import copy

import numpy as np
import pytest

from flexiv_control import (
    ControlMode,
    FakeBackend,
    GripperCommand,
    JointGripperTarget,
    JointTrajectory,
    JointWaypoint,
    Robot,
    RobotConfig,
)
from flexiv_control.client import RemoteRobot, RemoteRobotError
from flexiv_control.interpolation import JointTrajectoryInterpolator
from flexiv_control.server import FlexivControlServer
from flexiv_control.server import protocol as P


class OrderedFakeBackend(FakeBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.write_events = []

    def move_gripper(self, cmd):
        self.write_events.append(("gripper", float(cmd.width), float(cmd.velocity)))
        super().move_gripper(cmd)

    def stream_joint(self, q):
        self.write_events.append(("joint", np.asarray(q, float).copy()))
        super().stream_joint(q)


class LimitedOrderedFakeBackend(OrderedFakeBackend):
    def runtime_info(self):
        return {
            "gripper_limits": {
                "source": "flexivrdk.Gripper.params",
                "device_name": "fake-limited",
                "min_width_m": 0.0,
                "max_width_m": 0.1,
                "min_velocity_m_s": 0.06,
                "max_velocity_m_s": 0.2,
                "min_force_n": 5.0,
                "max_force_n": 40.0,
            }
        }


class ModeDriftBackend(LimitedOrderedFakeBackend):
    def __init__(self, *, q_after_mode=None, gripper_after_mode=None, **kwargs):
        super().__init__(**kwargs)
        self.q_after_mode = q_after_mode
        self.gripper_after_mode = gripper_after_mode

    def set_mode(self, mode, **kwargs):
        super().set_mode(mode, **kwargs)
        if self.q_after_mode is not None:
            self._q = np.asarray(self.q_after_mode, float).copy()
        if self.gripper_after_mode is not None:
            self._gripper_width = float(self.gripper_after_mode)


def strict_traj(initial, targets, frames, grippers=None):
    grippers = grippers or [None] * len(targets)
    return JointTrajectory(
        initial_positions=np.asarray(initial, float),
        initial_gripper_width=0.08 if any(g is not None for g in grippers) else None,
        waypoints=[
            JointWaypoint(
                positions=np.asarray(q, float),
                n_frames=n,
                gripper=g,
            )
            for q, n, g in zip(targets, frames, grippers)
        ],
        interpolation="linear",
        strict_timing=True,
        max_joint_speed_scale=0.3,
    )


def test_strict_prefix_32_1_is_authoritative_and_never_stretched():
    q0 = np.zeros(7)
    q1 = q0.copy()
    q1[0] = 0.1
    q2 = q1.copy()
    q2[0] += 0.005
    traj = strict_traj(q0, [q1, q2], [32, 1])
    interp = JointTrajectoryInterpolator(traj, q0, 100.0, max_joint_speed=np.full(7, 1.0))
    assert interp.requested_segment_ticks == [32, 1]
    assert interp.scheduled_segment_ticks == [32, 1]
    assert interp.requested_total_ticks == interp.scheduled_total_ticks == 33
    assert len(interp.setpoints()) == 33

    too_fast = strict_traj(q0, [np.full(7, 0.2)], [1])
    with pytest.raises(ValueError, match="n_frames=1 is authoritative"):
        JointTrajectoryInterpolator(too_fast, q0, 100.0, max_joint_speed=np.full(7, 1.0))


def test_joint_v4_schema_round_trip_and_old_or_loose_payload_refusal():
    q0 = np.zeros(7)
    traj = strict_traj(q0, [q0], [2], [JointGripperTarget(width=0.079, velocity=None)])
    payload = P.joint_trajectory_to_dict(traj)
    restored = P.joint_trajectory_from_dict(payload)
    assert restored.strict_timing is True
    assert restored.waypoints[0].gripper.velocity is None

    old = {"waypoints": [{"positions": q0.tolist(), "n_frames": 2, "duration": None}]}
    with pytest.raises(ValueError, match="keys do not match"):
        P.joint_trajectory_from_dict(old)
    extra = copy.deepcopy(payload)
    extra["legacy"] = True
    with pytest.raises(ValueError, match=r"extra=\['legacy'\]"):
        P.joint_trajectory_from_dict(extra)
    wrong = copy.deepcopy(payload)
    wrong["schema"] = "flexiv-control.joint-trajectory.v2"
    with pytest.raises(ValueError, match="unsupported JointTrajectory schema"):
        P.joint_trajectory_from_dict(wrong)

    invalid_numeric_payloads = []
    for mutate in (
        lambda d: d.__setitem__("max_joint_speed_scale", "0.3"),
        lambda d: d["initial_positions"].__setitem__(0, "0.0"),
        lambda d: d.__setitem__("initial_gripper_width", "0.08"),
        lambda d: d["waypoints"][0]["positions"].__setitem__(0, True),
        lambda d: d["waypoints"][0].__setitem__("n_frames", 2.0),
        lambda d: d["waypoints"][0]["gripper"].__setitem__("width", "0.079"),
        lambda d: d["waypoints"][0]["gripper"].__setitem__("force_limit", True),
        lambda d: d["waypoints"][0]["gripper"].__setitem__("velocity", "0.05"),
    ):
        invalid = copy.deepcopy(payload)
        mutate(invalid)
        invalid_numeric_payloads.append(invalid)
    for invalid in invalid_numeric_payloads:
        with pytest.raises(ValueError, match="JSON"):
            P.joint_trajectory_from_dict(invalid)


def test_first_strict_call_bounds_actual_first_setpoint_from_measured_state():
    q0 = np.zeros(7)
    backend = OrderedFakeBackend(start_q=q0)
    robot = Robot(
        config=RobotConfig(backend="fake", control_hz=100.0),
        backend=backend,
        control_hz=100.0,
    )
    server = FlexivControlServer(
        robot=robot, host="127.0.0.1", port=0, lease_ttl=1.0, host_lock=False
    )
    server.start()
    port = server._tcp.server_address[1]
    server.serve_in_thread()
    try:
        with RemoteRobot("127.0.0.1", port, owner="prefix") as remote:
            initial = q0.copy()
            initial[0] = 0.006
            unsafe_target = initial.copy()
            unsafe_target[0] = 0.012
            with pytest.raises(RemoteRobotError, match="first emitted joint target"):
                remote.execute_joint_trajectory(strict_traj(initial, [unsafe_target], [1]))
            assert backend.write_events == []
            assert backend.mode_log == []

            # Knot 0 need not equal telemetry: only the first actual command is
            # constrained, here exactly 0.6 rad/s * 10 ms from measured q.
            safe = remote.execute_joint_trajectory(strict_traj(initial, [initial], [1]))
            assert safe.success
            assert safe.log["joint_filter_anchor"] == q0.tolist()
            assert backend.joint_log[-1][0] == pytest.approx(0.006)
    finally:
        server.shutdown()


def test_predispatch_joint_recheck_uses_state_after_mode_switch():
    q0 = np.zeros(7)
    drifted_q = q0.copy()
    drifted_q[0] = -0.004
    backend = ModeDriftBackend(start_q=q0, q_after_mode=drifted_q)
    robot = Robot(
        config=RobotConfig(backend="fake", control_hz=100.0),
        backend=backend,
        control_hz=100.0,
    )
    robot.connect()
    target = q0.copy()
    target[0] = 0.005
    result = robot.execute_joint_trajectory(strict_traj(q0, [target], [1]))
    assert not result.success
    assert result.stop_reason == "joint_limit"
    assert "predispatch_joint_rejection" in result.log
    assert backend.mode_log == [ControlMode.NRT_JOINT_IMPEDANCE]
    assert backend.joint_log == []
    assert backend.gripper_log == []


def test_first_gripper_event_rechecks_mode_switch_width_drift():
    q0 = np.zeros(7)
    backend = ModeDriftBackend(start_q=q0, gripper_after_mode=0.075)
    robot = Robot(
        config=RobotConfig(backend="fake", control_hz=100.0),
        backend=backend,
        control_hz=100.0,
    )
    robot.connect()
    traj = strict_traj(
        q0,
        [q0],
        [1],
        [JointGripperTarget(width=0.079, force=15.0)],
    )
    result = robot.execute_joint_trajectory(traj)
    assert not result.success
    assert result.stop_reason == "gripper_limit"
    assert "predispatch_gripper_rejection" in result.log
    assert backend.joint_log == []
    assert backend.gripper_log == []


def test_gripper_move_dispatches_before_arm_and_streams_concurrently():
    q0 = np.zeros(7)
    backend = OrderedFakeBackend(start_q=q0)
    robot = Robot(
        config=RobotConfig(backend="fake", control_hz=100.0),
        backend=backend,
        control_hz=100.0,
    )
    robot.connect()
    traj = strict_traj(
        q0,
        [np.full(7, 0.002), np.full(7, 0.003)],
        [2, 1],
        [
            JointGripperTarget(width=0.079, force=15.0),
            JointGripperTarget(width=0.0785, force=15.0),
        ],
    )
    result = robot.execute_joint_trajectory(traj)
    assert result.success
    assert result.log["requested_segment_ticks"] == [2, 1]
    assert result.log["scheduled_segment_ticks"] == [2, 1]
    assert [event["velocity_m_s"] for event in result.log["gripper_events"]] == pytest.approx(
        [0.05, 0.05]
    )
    assert [kind for kind, *_ in backend.write_events] == [
        "gripper",
        "joint",
        "joint",
        "gripper",
        "joint",
    ]
    assert result.log["initial_joint_target"] == q0.tolist()
    assert result.log["ending_joint_target"] == pytest.approx(np.full(7, 0.003).tolist())
    assert len(result.log["gripper_tracking"]) == 3


def test_explicit_gripper_velocity_must_realize_same_segment_duration():
    q0 = np.zeros(7)
    backend = OrderedFakeBackend(start_q=q0)
    robot = Robot(
        config=RobotConfig(backend="fake", control_hz=100.0),
        backend=backend,
        control_hz=100.0,
    )
    robot.connect()
    traj = strict_traj(
        q0,
        [q0],
        [2],
        [JointGripperTarget(width=0.079, velocity=0.1)],
    )
    with pytest.raises(ValueError, match="does not realize width delta"):
        robot.execute_joint_trajectory(traj)
    assert backend.write_events == []


def test_runtime_gripper_params_are_prevalidated_before_any_write():
    q0 = np.zeros(7)
    backend = LimitedOrderedFakeBackend(start_q=q0)
    robot = Robot(
        config=RobotConfig(backend="fake", control_hz=100.0),
        backend=backend,
        control_hz=100.0,
    )
    robot.connect()
    # 1 mm / 20 ms = 0.05 m/s, below the runtime 0.06 m/s floor.
    traj = strict_traj(q0, [q0], [2], [JointGripperTarget(width=0.079)])
    with pytest.raises(ValueError, match="Gripper.params"):
        robot.execute_joint_trajectory(traj)
    assert backend.write_events == []


def test_first_call_gripper_ramp_uses_measured_width_not_declared_knot_zero():
    q0 = np.zeros(7)
    backend = LimitedOrderedFakeBackend(start_q=q0)
    robot = Robot(
        config=RobotConfig(backend="fake", control_hz=100.0),
        backend=backend,
        control_hz=100.0,
    )
    server = FlexivControlServer(
        robot=robot, host="127.0.0.1", port=0, lease_ttl=1.0, host_lock=False
    )
    server.start()
    port = server._tcp.server_address[1]
    server.serve_in_thread()
    try:
        with RemoteRobot("127.0.0.1", port, owner="prefix") as remote:
            traj = strict_traj(
                q0,
                [q0],
                [1],
                [JointGripperTarget(width=0.0786, force=15.0)],
            )
            traj.initial_gripper_width = 0.07
            result = remote.execute_joint_trajectory(traj)
            event = result.log["gripper_events"][0]
            assert result.success
            assert result.log["gripper_execution_anchor_m"] == pytest.approx(0.08)
            assert event["start_width_m"] == pytest.approx(0.08)
            assert event["velocity_m_s"] == pytest.approx(0.14)
    finally:
        server.shutdown()


def test_server_ack_is_provenance_and_prevalidation_preserves_it():
    q0 = np.zeros(7)
    backend = FakeBackend(start_q=q0, tracking_alpha=0.1)
    robot = Robot(
        config=RobotConfig(backend="fake", control_hz=100.0),
        backend=backend,
        control_hz=100.0,
    )
    server = FlexivControlServer(
        robot=robot, host="127.0.0.1", port=0, lease_ttl=1.0, host_lock=False
    )
    server.start()
    port = server._tcp.server_address[1]
    server.serve_in_thread()
    try:
        with RemoteRobot("127.0.0.1", port, owner="prefix") as remote:
            first_end = q0.copy()
            first_end[0] = 0.02
            first = strict_traj(q0, [first_end], [10])
            r1 = remote.execute_joint_trajectory(first)
            assert r1.log["continuity_source"] == "measured_rebase"
            assert server._last_ack_joint_target.tolist() == first_end.tolist()
            assert remote.get_state().q[0] < first_end[0]

            writes_before_old_rpc = len(backend.joint_log)
            old_rpc = server._dispatch(
                {
                    "id": 99,
                    "method": "execute_joint_trajectory",
                    "params": {
                        "owner": "prefix",
                        "traj": P.joint_trajectory_to_dict(first),
                    },
                }
            )
            assert old_rpc["ok"] is False
            assert "old joint payload refused" in old_rpc["error"]
            assert len(backend.joint_log) == writes_before_old_rpc
            assert server._last_ack_joint_target.tolist() == first_end.tolist()

            measured = remote.get_state().q
            rebased_end = measured.copy()
            rebased_end[0] += 0.002
            rebased = strict_traj(measured, [rebased_end], [1])
            r2 = remote.execute_joint_trajectory(rebased)
            assert r2.log["continuity_source"] == "measured_rebase_with_prior_ack"
            assert r2.log["previous_acknowledged_joint_target"] == first_end.tolist()
            assert np.max(np.abs(r2.log["initial_joint_delta_from_previous_ack_rad"])) > 1e-3
            assert server._last_ack_joint_target.tolist() == rebased_end.tolist()

            current = remote.get_state().q
            too_fast_end = current.copy()
            too_fast_end[0] += 0.5
            too_fast = strict_traj(current, [too_fast_end], [1])
            writes_before_prevalidation = len(backend.joint_log)
            with pytest.raises(RemoteRobotError, match="strict_timing joint rate"):
                remote.execute_joint_trajectory(too_fast)
            assert len(backend.joint_log) == writes_before_prevalidation
            assert server._last_ack_joint_target.tolist() == rebased_end.tolist()

            corrected_q = remote.get_state().q
            corrected = remote.execute_joint_trajectory(
                strict_traj(corrected_q, [corrected_q], [1])
            )
            assert corrected.log["continuity_source"] == "measured_rebase_with_prior_ack"

            remote.command_gripper(GripperCommand(width=0.07))
            assert server._last_ack_joint_target is None
            post_reset_q = remote.get_state().q
            post_reset = remote.execute_joint_trajectory(
                strict_traj(post_reset_q, [post_reset_q], [1])
            )
            assert post_reset.log["continuity_source"] == "measured_rebase"
    finally:
        server.shutdown()


def test_second_feedback_prefix_rebases_joint_and_gripper_to_current_measurement():
    q0 = np.zeros(7)
    backend = LimitedOrderedFakeBackend(start_q=q0, tracking_alpha=0.1)
    robot = Robot(
        config=RobotConfig(backend="fake", control_hz=100.0),
        backend=backend,
        control_hz=100.0,
    )
    server = FlexivControlServer(
        robot=robot, host="127.0.0.1", port=0, lease_ttl=1.0, host_lock=False
    )
    server.start()
    port = server._tcp.server_address[1]
    server.serve_in_thread()
    try:
        with RemoteRobot("127.0.0.1", port, owner="prefix") as remote:
            first_end = q0.copy()
            first_end[0] = 0.02
            first = strict_traj(
                q0,
                [first_end],
                [10],
                [JointGripperTarget(width=0.068, force=15.0)],
            )
            assert remote.execute_joint_trajectory(first).success
            assert backend._q[0] < first_end[0]

            # Model feedback between horizons: the next plan starts at current
            # telemetry, not the previous commanded endpoint.
            backend._gripper_width = 0.077
            measured = remote.get_state()
            assert measured.q[0] < first_end[0]
            assert measured.gripper_width != pytest.approx(0.068)
            second_end = measured.q.copy()
            second_end[0] += 0.003
            second = strict_traj(
                measured.q,
                [second_end],
                [1],
                [JointGripperTarget(width=0.0784, force=15.0)],
            )
            second.initial_gripper_width = measured.gripper_width
            result = remote.execute_joint_trajectory(second)
            event = result.log["gripper_events"][0]
            assert result.success
            assert result.log["continuity_source"] == "measured_rebase_with_prior_ack"
            assert result.log["joint_filter_anchor"] == pytest.approx(measured.q.tolist())
            assert result.log["gripper_execution_anchor_m"] == pytest.approx(0.077)
            assert result.log["previous_acknowledged_gripper_target_m"] == pytest.approx(0.068)
            assert event["start_width_m"] == pytest.approx(0.077)
            assert event["velocity_m_s"] == pytest.approx(0.14)
    finally:
        server.shutdown()


def test_stop_and_failed_execution_clear_ack_cache():
    q0 = np.zeros(7)
    backend = FakeBackend(start_q=q0)
    robot = Robot(
        config=RobotConfig(backend="fake", control_hz=100.0),
        backend=backend,
        control_hz=100.0,
    )
    server = FlexivControlServer(
        robot=robot, host="127.0.0.1", port=0, lease_ttl=1.0, host_lock=False
    )
    server.start()
    port = server._tcp.server_address[1]
    server.serve_in_thread()
    try:
        with RemoteRobot("127.0.0.1", port, owner="prefix") as remote:
            traj = strict_traj(q0, [q0], [1])
            assert remote.execute_joint_trajectory(traj).success
            assert server._last_ack_joint_target is not None

            backend._fault = True
            backend._mode = ControlMode.IDLE
            backend.mode_log.clear()
            joint_writes_before_fault = len(backend.joint_log)
            failed = remote.execute_joint_trajectory(traj)
            assert not failed.success
            assert failed.stop_reason == "backend_fault"
            assert failed.log["prewrite_safety_rejection"] is True
            assert backend.mode_log == []
            assert len(backend.joint_log) == joint_writes_before_fault
            assert server._last_ack_joint_target is None
            backend._fault = False

            assert remote.execute_joint_trajectory(traj).success
            assert server._last_ack_joint_target is not None
            remote.stop()
            assert server._last_ack_joint_target is None
    finally:
        server.shutdown()


def test_server_start_and_shutdown_reset_continuity_cache():
    q0 = np.zeros(7)
    robot = Robot(
        config=RobotConfig(backend="fake", control_hz=100.0),
        backend=FakeBackend(start_q=q0),
        control_hz=100.0,
    )
    server = FlexivControlServer(
        robot=robot, host="127.0.0.1", port=0, lease_ttl=1.0, host_lock=False
    )
    server._last_ack_joint_target = np.ones(7)
    server._last_ack_gripper_target = 0.05
    server._joint_target_owner = "stale"
    server.start()
    assert server._last_ack_joint_target is None
    assert server._last_ack_gripper_target is None
    server.serve_in_thread()
    server._last_ack_joint_target = np.ones(7)
    server._last_ack_gripper_target = 0.05
    server._joint_target_owner = "stale"
    server.shutdown()
    assert server._last_ack_joint_target is None
    assert server._last_ack_gripper_target is None
    assert server._joint_target_owner == ""
