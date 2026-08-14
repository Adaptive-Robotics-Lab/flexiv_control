"""A dependency-free kinematic simulator backend.

This makes the *entire* stack runnable with no robot, no RDK, and no ROS:
  * CI can exercise the action contract, interpolation, safety, server, gym env,
    and LeRobot adapter end to end;
  * new lab members can learn the API before touching hardware;
  * planner / RL / MPC code can be developed and unit-tested offline.

It models the TCP as a first-order system that tracks the commanded pose, plus
trivial joint and gripper tracking. It is NOT a dynamics simulator (use
``MujocoBackend`` for contact / physics). Its real value is that it records
every setpoint it receives, so tests can assert on exact command streams.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from ..transforms import quat_normalize, quat_slerp
from ..types import (
    CART_DOF,
    ControlMode,
    ForceControlParams,
    GripperCommand,
    ImpedanceParams,
    JointImpedanceParams,
    RobotState,
    SafetyStatus,
    StopReason,
)
from .base import RobotBackend


class FakeBackend(RobotBackend):
    def __init__(
        self,
        n_joints: int = 7,
        start_pose: Optional[np.ndarray] = None,
        start_q: Optional[np.ndarray] = None,
        tracking_alpha: float = 1.0,
    ):
        self.n_joints = n_joints
        self._connected = False
        self._mode = ControlMode.IDLE
        self._alpha = float(tracking_alpha)  # 1.0 = perfect tracking
        self._pose = (
            np.asarray(start_pose, float).reshape(7)
            if start_pose is not None
            else np.array([0.45, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0])
        )
        self._q = (
            np.asarray(start_q, float)
            if start_q is not None
            else np.array([0.0, -0.7, 0.0, 1.6, 0.0, 0.9, 0.0])
        )
        self._gripper_width = 0.08
        self._gripper_force = 0.0
        self._gripper_moving = False
        self._impedance = ImpedanceParams()
        self._max_wrench = np.array([40, 40, 40, 5, 5, 5], float)
        # Test hook: set True to simulate a robot-side fault (the control loop
        # polls in_fault() every tick and must halt).
        self._fault = False

        # Recorded command log (for tests / debugging).
        self.cartesian_log: List[np.ndarray] = []
        self.joint_log: List[np.ndarray] = []
        self.joint_torque_log: List[np.ndarray] = []
        self.gripper_log: List[GripperCommand] = []
        self.mode_log: List[ControlMode] = []

    # -- lifecycle ----------------------------------------------------------
    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # -- state --------------------------------------------------------------
    def read_state(self) -> RobotState:
        return RobotState(
            stamp=time.time(),
            q=self._q.copy(),
            dq=np.zeros(self.n_joints),
            tau=np.zeros(self.n_joints),
            tcp_pose=self._pose.copy(),
            tcp_vel=np.zeros(CART_DOF),
            wrench=np.zeros(CART_DOF),
            gripper_width=self._gripper_width,
            gripper_force=self._gripper_force,
            gripper_is_moving=self._gripper_moving,
            control_mode=self._mode,
            safety_status=SafetyStatus.FAULT if self._fault else SafetyStatus.OK,
            stop_reason=StopReason.BACKEND_FAULT if self._fault else StopReason.NONE,
        )

    def in_fault(self) -> bool:
        return self._fault

    # -- mode ---------------------------------------------------------------
    def set_mode(
        self,
        mode: ControlMode,
        *,
        impedance: Optional[ImpedanceParams] = None,
        joint_impedance: Optional[JointImpedanceParams] = None,
        force_control: Optional[ForceControlParams] = None,
        nullspace_q: Optional[np.ndarray] = None,
        max_contact_wrench: Optional[np.ndarray] = None,
    ) -> None:
        self._mode = mode
        self.mode_log.append(mode)
        if impedance is not None:
            self._impedance = impedance
        if max_contact_wrench is not None:
            self._max_wrench = np.asarray(max_contact_wrench, float)

    # -- streaming ----------------------------------------------------------
    def stream_cartesian(self, pose: np.ndarray, wrench: Optional[np.ndarray] = None) -> None:
        # Enforce the same mode precondition as the real RDK backend
        # (SendCartesianMotionForce faults outside a Cartesian motion mode).
        # The fake used to accept streaming in IDLE, which made dry runs
        # structurally unable to reveal a missing start_cartesian_impedance().
        if not self._mode.is_cartesian:
            raise RuntimeError(
                f"stream_cartesian in mode {self._mode.value!r}: the real backend "
                "requires a Cartesian motion mode; call start_cartesian_impedance() "
                "first (Robot.execute_cartesian_trajectory now auto-ensures it)"
            )
        pose = np.asarray(pose, float).reshape(7)
        self.cartesian_log.append(pose.copy())
        a = self._alpha
        self._pose[:3] = (1 - a) * self._pose[:3] + a * pose[:3]
        self._pose[3:7] = quat_slerp(self._pose[3:7], quat_normalize(pose[3:7]), a)

    def stream_joint(self, q: np.ndarray) -> None:
        # Mirror the real backend: joint streaming needs a joint motion mode.
        if self._mode.is_cartesian or self._mode == ControlMode.IDLE:
            raise RuntimeError(
                f"stream_joint in mode {self._mode.value!r}: call "
                "start_joint_impedance() first (Robot.move_joint does this)"
            )
        q = np.asarray(q, float)
        self.joint_log.append(q.copy())
        a = self._alpha
        self._q = (1 - a) * self._q + a * q

    def stream_joint_torque(self, tau: np.ndarray) -> None:
        if self._mode != ControlMode.RT_JOINT_TORQUE:
            raise RuntimeError(
                "stream_joint_torque requires RT_JOINT_TORQUE mode"
            )
        command = np.asarray(tau, dtype=float).reshape(self.n_joints)
        self.joint_torque_log.append(command.copy())

    def set_contact_wrench_limit(self, wrench: np.ndarray) -> None:
        # Mirror the firmware guard so tests can assert apply/restore.
        self._max_wrench = np.asarray(wrench, float).copy()

    # -- gripper ------------------------------------------------------------
    def move_gripper(self, cmd: GripperCommand) -> None:
        self.gripper_log.append(cmd)
        self._gripper_width = float(np.clip(cmd.width, 0.0, 0.10))
        self._gripper_force = cmd.force if cmd.grasp else 0.0
        self._gripper_moving = False

    # -- safety -------------------------------------------------------------
    def stop(self) -> None:
        self._mode = ControlMode.IDLE

    def home(self, q_home: Optional[np.ndarray] = None) -> None:
        self._q = (
            np.asarray(q_home, float)
            if q_home is not None
            else np.array([0.0, -0.7, 0.0, 1.6, 0.0, 0.9, 0.0])
        )
        # Mirror the real backend: homing runs the NRT 'Home' primitive and
        # leaves the robot in NRT_PRIMITIVE mode. Tracking this in the fake lets
        # offline tests catch mode-sequencing bugs (e.g. a reset() that homes
        # last and never re-enters a motion mode before step()).
        self._mode = ControlMode.NRT_PRIMITIVE
        self.mode_log.append(self._mode)
