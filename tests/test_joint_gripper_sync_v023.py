import copy

import numpy as np
import pytest

from flexiv_control import (
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


def test_joint_v3_schema_round_trip_and_old_or_loose_payload_refusal():
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


def test_server_ack_continuity_survives_tracking_lag_and_mutation_resets_it():
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
            assert r1.log["continuity_source"] == "first_call_measured_one_tick"
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

            chained = strict_traj(first_end, [first_end], [1])
            r2 = remote.execute_joint_trajectory(chained)
            assert r2.log["continuity_source"] == "last_acknowledged_target"

            bad = strict_traj(first_end + 0.01, [first_end], [10])
            before = len(backend.joint_log)
            with pytest.raises(RemoteRobotError, match="last acknowledged"):
                remote.execute_joint_trajectory(bad)
            assert len(backend.joint_log) == before

            remote.command_gripper(GripperCommand(width=0.07))
            assert server._last_ack_joint_target is None
            with pytest.raises(RemoteRobotError, match="one-tick measured"):
                remote.execute_joint_trajectory(chained)
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
            failed = remote.execute_joint_trajectory(traj)
            assert not failed.success
            assert failed.stop_reason == "backend_fault"
            assert server._last_ack_joint_target is None
            backend._fault = False

            assert remote.execute_joint_trajectory(traj).success
            assert server._last_ack_joint_target is not None
            remote.stop()
            assert server._last_ack_joint_target is None
    finally:
        server.shutdown()
