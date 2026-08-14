"""Core data types shared across the whole stack.

Conventions
-----------
* All quaternions are stored ``(w, x, y, z)`` -- the same order Flexiv RDK uses
  for TCP pose (``[x, y, z, qw, qx, qy, qz]``). Keeping one convention end to
  end avoids a whole class of silent sign/order bugs.
* SI units everywhere: metres, radians, seconds, Newtons, Newton-metres.
* Frames are named strings ("base", "tcp", "flange", "world"). The default is
  the robot base frame.

These are deliberately plain ``dataclasses`` with ``numpy`` arrays so they can
be (a) trivially serialised for the network server, (b) used from a notebook,
RL trainer, or MPC loop without dragging in ROS, and (c) type-checked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# A Rizon arm has 7 joints; kept as a module constant so backends can assert on it.
N_JOINTS_DEFAULT = 7
CART_DOF = 6  # x y z roll pitch yaw


class ControlMode(str, Enum):
    """Control modes, named to mirror Flexiv RDK ``flexiv::rdk::Mode``.

    The split between RT (1 kHz streaming, hard real time inside the robot) and
    NRT (discrete commands, robot interpolates internally) is the single most
    important distinction in this whole library. See ``docs/architecture.md``.
    """

    IDLE = "idle"

    # --- Non-real-time: robot's internal motion generator does the smoothing.
    #     Tolerant of a non-RT host and high network latency. Great default and
    #     great for cross-machine / Python-only setups.
    NRT_JOINT_POSITION = "nrt_joint_position"
    NRT_JOINT_IMPEDANCE = "nrt_joint_impedance"
    NRT_CARTESIAN_MOTION_FORCE = "nrt_cartesian_motion_force"
    NRT_PRIMITIVE = "nrt_primitive"  # go-home, calibration, vendor primitives

    # --- Real-time: 1 kHz streaming, <1 ms latency. The robot still runs the
    #     hard real-time impedance/motion-force loop internally; the host streams
    #     setpoints. Lowest latency / most reactive. Needs the Professional RDK
    #     license and (for the C++ path) a real-time scheduler.
    RT_JOINT_POSITION = "rt_joint_position"
    RT_JOINT_IMPEDANCE = "rt_joint_impedance"
    RT_CARTESIAN_MOTION_FORCE = "rt_cartesian_motion_force"

    # --- Torque-level research mode. Highest risk; gated off by default. The
    #     RDK backend raises unless built with allow_torque=True, and no facade /
    #     server / ROS path exposes it. Never a default entry for RL or a planner.
    RT_JOINT_TORQUE = "rt_joint_torque"

    @property
    def is_realtime(self) -> bool:
        return self.value.startswith("rt_")

    @property
    def is_cartesian(self) -> bool:
        return "cartesian" in self.value


class SafetyStatus(str, Enum):
    OK = "ok"
    CLIPPED = "clipped"          # last command was modified to stay in bounds
    HOLDING = "holding"          # command rejected / stale -> holding position
    STOPPED = "stopped"          # explicit/triggered stop
    FAULT = "fault"              # backend/hardware fault


class StopReason(str, Enum):
    NONE = "none"
    USER = "user"
    STALE_COMMAND = "stale_command"
    STALE_STATE = "stale_state"
    WORKSPACE = "workspace_limit"
    JOINT_LIMIT = "joint_limit"
    GRIPPER_LIMIT = "gripper_limit"
    TCP_SPEED = "tcp_speed_limit"
    POSE_JUMP = "pose_jump_limit"
    CONTACT_WRENCH = "contact_wrench"
    LEASE = "lease_violation"
    MODE = "mode_incompatible"
    BACKEND_FAULT = "backend_fault"


@dataclass
class RobotState:
    """A complete, backend-agnostic snapshot of the robot.

    Every consumer (planner, RL env, MPC loop, logger, teleop) reads *this*,
    never a raw ROS topic or RDK struct. That is what makes projects portable.
    """

    stamp: float = field(default_factory=time.time)

    # Joint space
    q: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS_DEFAULT))
    dq: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS_DEFAULT))
    tau: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS_DEFAULT))

    # Cartesian (TCP) space, in base frame
    tcp_pose: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1, 0, 0, 0], float))
    tcp_vel: np.ndarray = field(default_factory=lambda: np.zeros(CART_DOF))
    # External wrench estimated at the TCP, base frame: [fx, fy, fz, tx, ty, tz]
    wrench: np.ndarray = field(default_factory=lambda: np.zeros(CART_DOF))

    # Gripper
    gripper_width: float = 0.0
    gripper_force: float = 0.0
    gripper_is_moving: bool = False

    # Control / safety bookkeeping
    control_mode: ControlMode = ControlMode.IDLE
    safety_status: SafetyStatus = SafetyStatus.OK
    stop_reason: StopReason = StopReason.NONE
    active_owner: str = ""

    # Diagnostics (filled by the control loop / server; 0 from a bare backend)
    command_latency_ms: float = 0.0
    loop_period_ms: float = 0.0
    loop_jitter_us: float = 0.0
    missed_cycles: int = 0

    @property
    def tcp_position(self) -> np.ndarray:
        return self.tcp_pose[:3]

    @property
    def tcp_quat(self) -> np.ndarray:
        """TCP orientation quaternion (w, x, y, z)."""
        return self.tcp_pose[3:7]


@dataclass
class ImpedanceParams:
    """Cartesian impedance: stiffness K and damping ratio Z (per DoF).

    Flexiv's ``SetCartesianImpedance(K_x, Z_x)`` takes stiffness and a *damping
    ratio* (default 0.7), not an absolute damping matrix. We follow that API
    here. Order is [x, y, z, rx, ry, rz]. Linear units N/m, angular Nm/rad.
    """

    stiffness: np.ndarray = field(
        default_factory=lambda: np.array([800, 800, 800, 40, 40, 40], float)
    )
    damping_ratio: np.ndarray = field(
        default_factory=lambda: np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7], float)
    )


@dataclass
class JointImpedanceParams:
    """Joint-space impedance: stiffness (Nm/rad) and damping ratio per joint."""

    stiffness: np.ndarray = field(
        default_factory=lambda: np.array([3000, 3000, 800, 800, 200, 200, 200], float)
    )
    damping_ratio: np.ndarray = field(
        default_factory=lambda: np.full(N_JOINTS_DEFAULT, 0.7)
    )


@dataclass
class ForceControlParams:
    """Which Cartesian axes are force-controlled vs motion-controlled.

    Mirrors ``SetForceControlAxis`` / ``SetForceControlFrame``. Only meaningful
    in a *_CARTESIAN_MOTION_FORCE mode.
    """

    enabled_axes: np.ndarray = field(default_factory=lambda: np.zeros(CART_DOF, bool))
    target_wrench: np.ndarray = field(default_factory=lambda: np.zeros(CART_DOF))
    frame: str = "tcp"


@dataclass
class GripperCommand:
    """A single gripper setpoint.

    The canonical/hardware representation is a *continuous* opening ``width`` in
    metres plus ``force`` (N) and ``velocity`` (m/s) -- matching Flexiv RDK's
    ``Gripper.Move(width, velocity, force_limit)``. A parallel-jaw gripper is NOT
    binary open/close; the 0/1 you see in learning benchmarks is a normalized
    *abstraction* on top of this continuous width. ``grasp=True`` selects
    move-until-contact force control (RDK ``Gripper.Grasp(force)``) instead of a
    position move.

    .. warning:: When ``grasp=True``, the RDK backend calls ``Gripper.Grasp(force)``
       and **ignores** ``width`` entirely -- the fingers close until contact at the
       force limit, and on a MISS (no object between the fingers) they slam fully
       shut to ~0.000 m. A position-actuator MuJoCo gate that drives to a finite
       close width cannot reproduce that air-close, so a mis-positioned grasp can
       pass validation in sim yet collapse to an empty grip on hardware (an
       invisible failure). For a KNOWN-SIZE object, prefer a force-limited
       ``Gripper.Move(close_width, velocity, force)`` -- set ``grasp=False`` with
       ``width`` = the just-touch width: it stalls on the object on contact but
       FLOORS at ``close_width`` on a miss, so the settled width stays non-zero and
       a "closed on air" failure is detectable. Reserve ``grasp=True`` for blind
       closes on unknown-size objects. (The MuJoCo backend mirrors the blind close;
       both drive the fingers closed and let contact stop them.)
    """

    width: float = 0.0          # metres
    force: float = 20.0         # Newtons (clamping force)
    velocity: float = 0.1       # m/s
    # If True the command is "grasp" (move until contact/force); else "move".
    grasp: bool = False

    @classmethod
    def from_normalized(
        cls,
        opening: float,
        *,
        span: float = 0.08,
        min_width: float = 0.0,
        force: float = 20.0,
        velocity: float = 0.1,
    ) -> "GripperCommand":
        """Build from a normalized opening in ``[0, 1]`` (1 = open, 0 = closed).

        ``width = min_width + clip(opening, 0, 1) * span``. Pass your gripper's
        real stroke as ``span`` (e.g. RDK ``GripperParams.max_width``) so the
        normalized learning-layer command maps to true metres -- do not assume a
        fixed stroke.
        """
        o = float(np.clip(opening, 0.0, 1.0))
        return cls(width=min_width + o * span, force=force, velocity=velocity)

    @classmethod
    def from_signed_action(
        cls,
        action: float,
        *,
        span: float = 0.08,
        min_width: float = 0.0,
        force: float = 20.0,
        velocity: float = 0.1,
    ) -> "GripperCommand":
        """Build from a signed action in ``[-1, 1]`` (the RL/teleop convention):
        ``+1`` = open, ``-1`` = closed, and a *negative* action also selects grasp
        (move-until-contact). This is the ``[-1, 1] -> [0, 1]`` mirror of
        :meth:`from_normalized`, so the delta-action surface and the traj surface
        share one gripper encoding instead of each rolling its own remap.
        """
        a = float(np.clip(action, -1.0, 1.0))
        cmd = cls.from_normalized(
            (a + 1.0) / 2.0, span=span, min_width=min_width, force=force, velocity=velocity
        )
        cmd.grasp = a < 0.0
        return cmd
