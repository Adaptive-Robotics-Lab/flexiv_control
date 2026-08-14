from __future__ import annotations

import numpy as np
import pytest

from flexiv_control import (
    JointGripperForceTarget,
    JointTorqueTrajectory,
    JointTorqueWaypoint,
    Robot,
    RobotConfig,
)
from flexiv_control.backends.fake import FakeBackend
from flexiv_control.client import RemoteRobot
from flexiv_control.robot import TrajectoryPrevalidationError
from flexiv_control.server import FlexivControlServer
from flexiv_control.server import protocol as P


class TorqueRuntimeFake(FakeBackend):
    def runtime_info(self) -> dict:
        return {
            "joint_limits": {
                "source": "fake.RobotInfo",
                "position_min_rad": [-2.0] * 7,
                "position_max_rad": [2.0] * 7,
                "velocity_max_rad_s": [2.0] * 7,
                "torque_max_nm": [100.0] * 7,
            },
            "gripper_limits": {
                "min_force_n": 0.0,
                "max_force_n": 80.0,
            },
        }


def trajectory(value: float = 10.0) -> JointTorqueTrajectory:
    return JointTorqueTrajectory(
        initial_torques=np.zeros(7),
        waypoints=[
            JointTorqueWaypoint(
                torques=np.full(7, value),
                n_frames=2,
                gripper=JointGripperForceTarget(force=40.0),
            )
        ],
        max_joint_torque_scale=0.3,
        safety_profile="tabletop_safe",
    )


def test_protocol_v5_roundtrip_and_identity() -> None:
    original = trajectory()
    restored = P.joint_torque_trajectory_from_dict(
        P.joint_torque_trajectory_to_dict(original)
    )
    np.testing.assert_array_equal(restored.initial_torques, np.zeros(7))
    np.testing.assert_array_equal(restored.waypoints[0].torques, 10.0)
    assert P.PROTOCOL_ID == "flexiv-control.trajectory-rpc.v5"
    assert P.PROTOCOL_CONTRACT["trajectory_rpcs"][
        "execute_joint_torque_trajectory"
    ] == "traj"


def test_torque_execution_interpolates_and_uses_runtime_limits() -> None:
    backend = TorqueRuntimeFake()
    robot = Robot(
        RobotConfig(
            backend="fake",
            control_hz=1000.0,
            allow_joint_torque=True,
        ),
        backend=backend,
    )
    robot.connect()
    result = robot.execute_joint_torque_trajectory(trajectory())
    assert result.success
    np.testing.assert_allclose(backend.joint_torque_log, [[5.0] * 7, [10.0] * 7])
    assert result.log["gravity_compensation"] is True
    assert result.log["soft_limits"] is True
    assert result.log["effective_torque_max_nm"] == [30.0] * 7
    assert result.log["acknowledged_ending_gripper_force_n"] == 40.0
    assert result.final_state is not None
    assert result.final_state.gripper_force == 40.0


def test_torque_ack_reports_last_dispatched_force_not_last_waypoint_shape() -> None:
    backend = TorqueRuntimeFake()
    robot = Robot(
        RobotConfig(
            backend="fake", control_hz=1000.0, allow_joint_torque=True
        ),
        backend=backend,
    )
    robot.connect()
    traj = JointTorqueTrajectory(
        initial_torques=np.zeros(7),
        waypoints=[
            JointTorqueWaypoint(
                torques=np.zeros(7),
                n_frames=1,
                gripper=JointGripperForceTarget(force=20.0),
            ),
            JointTorqueWaypoint(torques=np.zeros(7), n_frames=1),
        ],
        max_joint_torque_scale=0.3,
        safety_profile="tabletop_safe",
    )
    result = robot.execute_joint_torque_trajectory(traj)
    assert result.success
    assert result.log["ending_gripper_force_n"] == 20.0
    assert result.log["acknowledged_ending_gripper_force_n"] == 20.0
    assert result.log["gripper_events"] == [
        {"mode": "force", "segment": 0, "force_n": 20.0}
    ]


def test_remote_torque_result_preserves_physical_ack_and_measured_end_state() -> None:
    local = Robot(
        RobotConfig(
            backend="fake", control_hz=1000.0, allow_joint_torque=True
        ),
        backend=TorqueRuntimeFake(),
    )
    server = FlexivControlServer(
        robot=local,
        host="127.0.0.1",
        port=0,
        host_lock=False,
    )
    server.start()
    assert server._tcp is not None
    port = server._tcp.server_address[1]
    server.serve_in_thread()
    try:
        with RemoteRobot("127.0.0.1", port, owner="torque-test") as remote:
            result = remote.execute_joint_torque_trajectory(trajectory())
        assert result.success
        assert result.log["acknowledged_ending_joint_torque_nm"] == [10.0] * 7
        assert result.log["acknowledged_ending_gripper_force_n"] == 40.0
        assert result.final_state is not None
        assert result.final_state.gripper_width == pytest.approx(0.08)
        assert result.final_state.gripper_force == pytest.approx(40.0)
    finally:
        server.shutdown()


def test_torque_execution_requires_and_preserves_command_continuity() -> None:
    backend = TorqueRuntimeFake()
    robot = Robot(
        RobotConfig(
            backend="fake",
            control_hz=1000.0,
            allow_joint_torque=True,
        ),
        backend=backend,
    )
    robot.connect()
    first = robot.execute_joint_torque_trajectory(trajectory())
    assert first.success
    second = JointTorqueTrajectory(
        initial_torques=np.full(7, 10.0),
        waypoints=[JointTorqueWaypoint(torques=np.zeros(7), n_frames=2)],
        max_joint_torque_scale=0.3,
        safety_profile="tabletop_safe",
    )
    assert robot.execute_joint_torque_trajectory(second).success
    np.testing.assert_allclose(
        backend.joint_torque_log[-2:],
        [[5.0] * 7, [0.0] * 7],
    )

    with pytest.raises(TrajectoryPrevalidationError, match="acknowledged"):
        robot.execute_joint_torque_trajectory(
            JointTorqueTrajectory(
                initial_torques=np.ones(7),
                waypoints=[JointTorqueWaypoint(torques=np.ones(7), n_frames=1)],
                max_joint_torque_scale=0.3,
                safety_profile="tabletop_safe",
            )
        )
    robot.stop()
    robot.clear_stop()
    assert robot.execute_joint_torque_trajectory(trajectory()).success


def test_torque_execution_fails_closed_without_opt_in_or_tau_max() -> None:
    disabled = Robot(
        RobotConfig(backend="fake", control_hz=1000.0),
        backend=TorqueRuntimeFake(),
    )
    disabled.connect()
    with pytest.raises(TrajectoryPrevalidationError, match="disabled"):
        disabled.execute_joint_torque_trajectory(trajectory())

    missing = Robot(
        RobotConfig(
            backend="fake", control_hz=1000.0, allow_joint_torque=True
        ),
        backend=FakeBackend(),
    )
    missing.connect()
    with pytest.raises(TrajectoryPrevalidationError, match="tau_max"):
        missing.execute_joint_torque_trajectory(trajectory())


def test_torque_execution_rejects_scaled_limit_before_write() -> None:
    backend = TorqueRuntimeFake()
    robot = Robot(
        RobotConfig(
            backend="fake", control_hz=1000.0, allow_joint_torque=True
        ),
        backend=backend,
    )
    robot.connect()
    with pytest.raises(TrajectoryPrevalidationError, match="tau_max"):
        robot.execute_joint_torque_trajectory(trajectory(31.0))
    assert backend.joint_torque_log == []
