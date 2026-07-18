"""The backend abstraction.

A *backend* is the only place that knows about a specific robot or simulator.
Everything above it (the action contract, safety, interpolation, the Robot
facade, the server, the gym env, the LeRobot adapter) is backend-agnostic. This
is what lets the same high-level code drive:

  * ``FlexivRdkBackend``  -- the real Rizon via Flexiv RDK (primary target),
  * ``FakeBackend``       -- a kinematic sim for CI / dry-runs / onboarding,
  * ``MujocoBackend``     -- a MuJoCo sim for RL / planner real2sim2real,
  * (yours)               -- any other arm, by implementing this interface.

The streaming methods are deliberately mode-aware: a backend maps
``stream_cartesian`` to RDK ``StreamCartesianMotionForce`` in an RT mode and to
``SendCartesianMotionForce`` in an NRT mode, so the control loop above stays
identical.
"""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np

from ..safety import SafetyProfile  # noqa: F401  (re-exported convenience)
from ..types import (
    ControlMode,
    ForceControlParams,
    GripperCommand,
    ImpedanceParams,
    JointImpedanceParams,
    RobotState,
)


class RobotBackend(abc.ABC):
    n_joints: int = 7

    # -- lifecycle ----------------------------------------------------------
    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def disconnect(self) -> None: ...

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool: ...

    # -- state --------------------------------------------------------------
    @abc.abstractmethod
    def read_state(self) -> RobotState: ...

    # -- mode ---------------------------------------------------------------
    @abc.abstractmethod
    def set_mode(
        self,
        mode: ControlMode,
        *,
        impedance: Optional[ImpedanceParams] = None,
        joint_impedance: Optional[JointImpedanceParams] = None,
        force_control: Optional[ForceControlParams] = None,
        nullspace_q: Optional[np.ndarray] = None,
        max_contact_wrench: Optional[np.ndarray] = None,
    ) -> None: ...

    # -- streaming setpoints (called once per control tick) -----------------
    @abc.abstractmethod
    def stream_cartesian(
        self, pose: np.ndarray, wrench: Optional[np.ndarray] = None
    ) -> None: ...

    @abc.abstractmethod
    def stream_joint(self, q: np.ndarray) -> None: ...

    # -- gripper ------------------------------------------------------------
    @abc.abstractmethod
    def move_gripper(self, cmd: GripperCommand) -> None: ...

    # -- safety / motion control -------------------------------------------
    @abc.abstractmethod
    def stop(self) -> None: ...

    def zero_ft_sensor(self) -> None:
        """Zero (bias-calibrate) the 6-DoF force/torque sensor. Force-control
        modes fault with event 301004 until this has run in the current session.
        Concrete default is a no-op so backends without an F/T sensor (fake /
        mujoco) inherit it unchanged; the flexiv_rdk backend overrides it."""
        return None

    def set_contact_wrench_limit(self, wrench: np.ndarray) -> None:
        """Update the robot-firmware contact-wrench guard mid-mode.

        The executor calls this when a traj carries a granted
        ``contact_wrench_allowance`` (held-payload transport) and again with the
        profile cap when the traj ends, so the firmware guard tracks the same
        effective cap as the host-side guards. Default no-op: sim/fake backends
        have no firmware guard; the flexiv_rdk backend overrides it with
        ``SetMaxContactWrench`` (Cartesian modes only)."""
        return None

    def in_fault(self) -> bool:
        """Whether the robot is in a fault / protective-stop state.

        The single-writer control loop polls this EVERY tick and halts streaming
        if it returns True -- a robot-side fault (collision reflex, joint-limit
        protective stop, E-stop) is invisible to the host-side safety filter, so
        it must be surfaced here. Hardware backends override it (e.g. RDK
        ``robot.fault()``); sim/fake backends are never faulted unless a test
        injects one, so the default is False."""
        return False

    def home(self, q_home: Optional[np.ndarray] = None) -> None:
        """Default home via the joint path; backends may override with a
        vendor primitive (e.g. RDK ``NRT_PRIMITIVE_EXECUTION`` 'Home')."""
        raise NotImplementedError
