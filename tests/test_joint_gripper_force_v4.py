from __future__ import annotations

import copy

import numpy as np
import pytest

from flexiv_control import (
    JointGripperForceTarget,
    JointGripperTarget,
    JointTrajectory,
    JointWaypoint,
    Robot,
    RobotConfig,
)
from flexiv_control.robot import TrajectoryPrevalidationError
from flexiv_control.server import protocol as P


def _trajectory(*forces: float, q0: np.ndarray | None = None) -> JointTrajectory:
    q0 = np.zeros(7) if q0 is None else np.asarray(q0, float)
    return JointTrajectory(
        initial_positions=q0,
        waypoints=[
            JointWaypoint(
                positions=q0.copy(),
                n_frames=2,
                gripper=JointGripperForceTarget(force=force),
            )
            for force in forces
        ],
        interpolation="linear",
        strict_timing=True,
        max_joint_speed_scale=0.3,
    )


def test_v4_force_and_move_targets_are_explicitly_tagged() -> None:
    force_payload = P.joint_trajectory_to_dict(_trajectory(20.0))
    assert force_payload["schema"] == "flexiv-control.joint-trajectory.v4"
    assert force_payload["waypoints"][0]["gripper"] == {
        "mode": "force",
        "force": 20.0,
    }
    restored = P.joint_trajectory_from_dict(force_payload)
    assert restored.initial_gripper_width is None
    assert isinstance(restored.waypoints[0].gripper, JointGripperForceTarget)

    move = JointTrajectory(
        initial_positions=np.zeros(7),
        initial_gripper_width=0.08,
        waypoints=[
            JointWaypoint(
                positions=np.zeros(7),
                n_frames=2,
                gripper=JointGripperTarget(width=0.07, force=15.0),
            )
        ],
        interpolation="linear",
        strict_timing=True,
    )
    move_payload = P.joint_trajectory_to_dict(move)
    assert move_payload["waypoints"][0]["gripper"] == {
        "mode": "move",
        "width": 0.07,
        "force_limit": 15.0,
        "velocity": None,
    }


def test_signed_effort_latent_decodes_once_to_physical_newtons() -> None:
    assert JointGripperForceTarget.from_signed_effort_latent(
        0.5, force_limit=80.0
    ).force == pytest.approx(40.0)
    assert JointGripperForceTarget.from_signed_effort_latent(
        -1.0, force_limit=80.0
    ).force == pytest.approx(-80.0)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        JointGripperForceTarget.from_signed_effort_latent(
            1.01, force_limit=80.0
        )
    with pytest.raises(ValueError, match="force_limit"):
        JointGripperForceTarget.from_signed_effort_latent(
            0.0, force_limit=0.0
        )


def test_v4_rejects_untagged_or_structurally_ambiguous_gripper() -> None:
    payload = P.joint_trajectory_to_dict(_trajectory(20.0))
    untagged = copy.deepcopy(payload)
    untagged["waypoints"][0]["gripper"].pop("mode")
    with pytest.raises(ValueError, match="mode"):
        P.joint_trajectory_from_dict(untagged)
    mixed = copy.deepcopy(payload)
    mixed["waypoints"][0]["gripper"]["width"] = 0.02
    with pytest.raises(ValueError, match="extra"):
        P.joint_trajectory_from_dict(mixed)


def test_force_trajectory_dispatches_signed_grasp_at_segment_boundaries() -> None:
    robot = Robot(RobotConfig(backend="fake", control_hz=1000.0))
    with robot:
        robot.acquire_lease("test")
        robot.start_joint_impedance()
        result = robot.execute_joint_trajectory(
            _trajectory(17.0, -9.0, q0=robot.get_state().q)
        )
    assert result.success
    assert [cmd.force for cmd in robot.backend.gripper_log] == [17.0, -9.0]
    assert all(cmd.grasp for cmd in robot.backend.gripper_log)
    assert [event["mode"] for event in result.log["gripper_events"]] == [
        "force",
        "force",
    ]
    assert result.log["ending_gripper_target_m"] == pytest.approx(0.08)
    assert result.log["ending_gripper_force_n"] == -9.0
    force_rows = [
        row for row in result.log["gripper_tracking"] if row["target_force_n"] is not None
    ]
    assert force_rows
    assert all(row["target_width_m"] is None for row in force_rows)
    assert all(row["error_m"] is None for row in force_rows)


def test_force_trajectory_fails_closed_against_runtime_limits() -> None:
    robot = Robot(RobotConfig(backend="fake", control_hz=1000.0))
    with robot:
        robot.acquire_lease("test")
        robot.start_joint_impedance()
        robot.backend.runtime_info = lambda: {
            "gripper_limits": {
                "min_width_m": 0.0,
                "max_width_m": 0.1,
                "min_velocity_m_s": 0.001,
                "max_velocity_m_s": 0.2,
                "min_force_n": -80.0,
                "max_force_n": 80.0,
            }
        }
        with pytest.raises(TrajectoryPrevalidationError, match="outside runtime"):
            robot.execute_joint_trajectory(_trajectory(81.0, q0=robot.get_state().q))
    assert robot.backend.gripper_log == []
    assert robot.backend.joint_log == []
