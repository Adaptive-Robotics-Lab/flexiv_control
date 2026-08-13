"""The unified *action contract*.

This is the single most important module for cross-project reuse. Every high
level component -- a receding-horizon planner, an MPC loop, an RL policy, a
SpaceMouse teleop bridge -- emits one of these objects, and every backend knows
how to execute them. Nobody re-invents "how do I talk to the robot".

Why this exact shape
--------------------
Action-chunking and receding-horizon planners typically emit a candidate action
of the form

    u = ( (x_j, y_j, z_j, w_j, n_j) )_{j=1..H}

i.e. a short sequence of Cartesian waypoints, each with a gripper command ``w_j``
and a *number of low-level control frames* ``n_j``. That ``n_j`` is precisely a
duration once you fix a control rate:

    duration_j = n_frames_j / control_hz

``CartesianTrajectory`` below is that object, generalised so it is also exactly what
an MPC horizon, a robosuite/MuJoCo rollout, or an RL action-traj needs:
  * positions become full SE(3) poses (orientation may be *held* -> position-only
    trajs map in with zero changes),
  * per-waypoint stiffness/limits so the *same* traj can describe a free-space
    reach and a contact-rich push,
  * an explicit ``representation`` (absolute vs relative-to-traj-start) and a
    predict-vs-execute horizon split (``n_execute`` -> ``horizon_exec``), so the
    receding-horizon "predict H_pred, execute H_exec, replan" loop (Diffusion
    Policy Tp/Ta) is first-class rather than implicit,
  * an explicit safety_profile name so execution is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import List, Optional, Union

import numpy as np

from .types import CART_DOF, ForceControlParams, GripperCommand, ImpedanceParams


# ----------------------------------------------------------------------------
# Cartesian
# ----------------------------------------------------------------------------
@dataclass
class CartesianWaypoint:
    """One Cartesian target.

    ``position`` is always required. ``quaternion`` (w, x, y, z) is optional:
    leave it ``None`` to *hold the current/previous orientation*, which is what
    a position-only planner (e.g. one emitting ``(x, y, z)`` waypoints) wants.

    Either ``n_frames`` (a per-waypoint frame count) or ``duration`` may be given. The
    interpolator converts ``n_frames`` to a duration using the active control
    rate, so the same traj runs identically whether the loop is 100 Hz or
    1 kHz, as long as ``n_frames`` is interpreted at that rate.
    """

    position: np.ndarray
    quaternion: Optional[np.ndarray] = None  # (w, x, y, z); None -> hold
    gripper: Optional[GripperCommand] = None  # None -> hold gripper
    n_frames: Optional[int] = None  # number of low-level control frames
    duration: Optional[float] = None  # seconds (alternative to n_frames)
    frame: str = "base"

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, float).reshape(3)
        if self.quaternion is not None:
            q = np.asarray(self.quaternion, float).reshape(4)
            n = np.linalg.norm(q)
            if n < 1e-9:
                raise ValueError("zero-norm quaternion")
            self.quaternion = q / n
        if self.n_frames is None and self.duration is None:
            raise ValueError("CartesianWaypoint needs either n_frames or duration")
        if self.n_frames is not None and self.n_frames <= 0:
            raise ValueError("n_frames must be positive")

    def resolve_duration(self, control_hz: float) -> float:
        if self.duration is not None:
            return float(self.duration)
        return float(self.n_frames) / float(control_hz)


class TrajectoryRepresentation(str, Enum):
    """How the waypoint poses in a traj are interpreted.

    * ``ABSOLUTE`` (default): each waypoint is an absolute target in the traj's
      ``frame`` (base by default) -- the ALOHA/DROID convention.
    * ``RELATIVE_TO_START``: each waypoint is a pose *relative to the TCP pose at
      the start of the traj* (composed ``T_abs = T_start . T_rel``) -- the
      UMI-style relative-trajectory convention that is robust to base/camera
      calibration drift. It is resolved to absolute at execution time against the
      live start pose, so a traj never accumulates sequential step-to-step error.

    Which is better is task/setup dependent (UMI favours relative for
    calibration-free in-the-wild use; DROID ships absolute), so this is an
    explicit field, not a baked-in default.
    """

    ABSOLUTE = "absolute"
    RELATIVE_TO_START = "relative_to_start"


@dataclass
class CartesianTrajectory:
    """A short, bounded sequence of Cartesian waypoints -- the core action type.

    Used by a receding-horizon planner (execute first segment, replan), MPC (the first slice of a
    horizon), scripted manipulation, and high-level RL.
    """

    waypoints: List[CartesianWaypoint]

    # Compliance for the whole traj (a waypoint may override later if needed).
    impedance: ImpedanceParams = field(default_factory=ImpedanceParams)
    force_control: Optional[ForceControlParams] = None

    # Kinematic SPEED envelope. Enforced as TIGHTENING-ONLY at execution: the
    # interpolator runs at min(traj cap, active profile cap), so a traj may
    # slow itself below the profile but can never relax the profile's limits.
    max_tcp_linear_speed: float = 0.25  # m/s
    max_tcp_angular_speed: float = 0.60  # rad/s
    # Acceleration fields are ADVISORY metadata only (logged, not enforced);
    # the profile's opt-in per-tick accel cap (max_linear_accel) is what binds.
    max_tcp_linear_acc: float = 1.0  # m/s^2
    max_tcp_angular_acc: float = 2.0  # rad/s^2

    # Contact envelope (None -> use the safety profile default). Like the speed
    # caps, applied as min(traj, profile) -- tightening only.
    max_contact_wrench: Optional[np.ndarray] = None  # [fx,fy,fz,tx,ty,tz]

    # Payload wrench allowance (None -> none). Added ON TOP of the profile's
    # ``max_contact_wrench`` for this traj, so a transport of a KNOWN held
    # payload is not stopped by the object's own static wrench (weight +
    # cantilever torque read by the wrist F/T). NOT a client-side override:
    # the server clamps the request to the profile's ``max_wrench_allowance``
    # (default zero -- a deployment must explicitly grant headroom in its
    # safety YAML before any traj can relax the contact guard).
    contact_wrench_allowance: Optional[np.ndarray] = None  # [fx,fy,fz,tx,ty,tz]

    # Gripper-close tracking gate (None -> disabled). When set, a CLOSING
    # gripper command (``grasp=True`` or width below the measured width) is
    # SKIPPED if the instantaneous TCP tracking error exceeds this bound at
    # the tick where the close would fire -- a descend that stalled on
    # unmodeled contact (impedance deflection below the wrench cap) leaves
    # the tool at an unplanned height, and closing there pinches whatever is
    # under the pads instead of the planned grasp. Once a close is skipped,
    # every LATER closing command in the traj is skipped too (the grasp plan
    # is invalid; a follow-up force-grasp would blind-close on air or rim).
    # Opening commands are never gated. Recorded in ``result.log['close_aborted']``.
    grip_tracking_gate_m: Optional[float] = None

    # Expected active safety profile (reproducibility contract). Empty string
    # (default) = "execute under whatever profile is active". A non-empty name
    # is VERIFIED at execution: if it does not match the robot's active profile,
    # ``execute_cartesian_trajectory`` raises instead of silently running under a
    # different envelope. The requested and active names are always recorded in
    # ``ExecutionResult.log``.
    safety_profile: str = ""

    frame: str = "base"

    # Absolute vs relative-to-traj-start pose semantics (see TrajectoryRepresentation).
    representation: TrajectoryRepresentation = TrajectoryRepresentation.ABSOLUTE

    # Receding-horizon split: the traj *predicts* len(waypoints) (H_pred) but the
    # robot executes only the first ``n_execute`` (H_exec) before replanning.
    # None -> execute the whole traj. Diffusion-Policy reference: predict ~16,
    # execute ~8.
    n_execute: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.waypoints:
            raise ValueError("CartesianTrajectory needs at least one waypoint")
        if self.max_contact_wrench is not None:
            self.max_contact_wrench = np.asarray(self.max_contact_wrench, float).reshape(CART_DOF)
        if self.contact_wrench_allowance is not None:
            allow = np.asarray(self.contact_wrench_allowance, float).reshape(CART_DOF)
            if np.any(allow < 0):
                raise ValueError("contact_wrench_allowance must be non-negative")
            self.contact_wrench_allowance = allow
        if self.grip_tracking_gate_m is not None and self.grip_tracking_gate_m <= 0:
            raise ValueError("grip_tracking_gate_m must be positive (or None to disable)")

    @property
    def horizon(self) -> int:
        return len(self.waypoints)

    @property
    def horizon_pred(self) -> int:
        """Number of predicted waypoints (H_pred)."""
        return len(self.waypoints)

    @property
    def horizon_exec(self) -> int:
        """Number of waypoints actually executed before replanning (H_exec)."""
        n = self.n_execute if self.n_execute is not None else len(self.waypoints)
        return max(1, min(int(n), len(self.waypoints)))

    def total_duration(self, control_hz: float) -> float:
        return sum(w.resolve_duration(control_hz) for w in self.waypoints)

    def resolve_to_absolute(self, start_pose: np.ndarray) -> "CartesianTrajectory":
        """Return an ABSOLUTE-representation copy of this traj.

        Identity if already absolute. For ``RELATIVE_TO_START`` each waypoint is
        composed onto ``start_pose`` (the live TCP pose at traj start) as
        ``T_abs = T_start . T_rel`` -- so relative trajs are re-anchored to the
        current measured pose each cycle and never accumulate sequential error.
        """
        if self.representation == TrajectoryRepresentation.ABSOLUTE:
            return self
        from . import transforms as T

        sp = np.asarray(start_pose, float).reshape(7)
        s_pos, s_quat = sp[:3], sp[3:7]
        new_wps: List[CartesianWaypoint] = []
        for w in self.waypoints:
            abs_pos = s_pos + T.quat_rotate(s_quat, w.position)
            abs_quat = (
                s_quat.copy()
                if w.quaternion is None
                else T.quat_normalize(T.quat_mul(s_quat, w.quaternion))
            )
            new_wps.append(
                CartesianWaypoint(
                    position=abs_pos,
                    quaternion=abs_quat,
                    gripper=w.gripper,
                    n_frames=w.n_frames,
                    duration=w.duration,
                    frame=w.frame,
                )
            )
        return replace(self, waypoints=new_wps, representation=TrajectoryRepresentation.ABSOLUTE)

    def for_execution(self, start_pose: np.ndarray) -> "CartesianTrajectory":
        """Resolve to absolute and slice to the execution horizon H_exec.

        This is what a ``Robot`` runs: relative trajs become absolute against the
        live start pose, and only the first ``n_execute`` waypoints are kept (the
        rest are discarded and re-predicted next cycle -- receding horizon).
        """
        c = self.resolve_to_absolute(start_pose)
        he = self.horizon_exec
        if he >= len(c.waypoints):
            return c
        return replace(c, waypoints=c.waypoints[:he], n_execute=None)

    # -- Convenience constructors -------------------------------------------
    @classmethod
    def from_waypoint_array(
        cls,
        u: np.ndarray,
        *,
        gripper_force: float = 20.0,
        gripper_span: float = 0.08,
        hold_orientation: bool = True,
        frames_hz: Optional[float] = None,
        **traj_kwargs,
    ) -> "CartesianTrajectory":
        """Build a traj directly from a planner's ``(H, 5)`` action array ``u``.

        ``u`` is an ``(H, 5)`` array of rows ``(x, y, z, w, n)`` where ``w`` is a
        *normalised* gripper command in ``[0, 1]`` (1 = open, 0 = closed) and ``n``
        is the integer number of low-level control frames. ``w`` is mapped to a
        physical width ``w * gripper_span`` -- pass your gripper's real stroke as
        ``gripper_span`` (e.g. RDK ``GripperParams.max_width``); 0.08 m is a
        generic conservative default, NOT a measured stroke (the lab GN01's is
        0.10 m per its robot config). Orientation is
        held from the previous pose; use :meth:`from_pose_array` to command it.

        ``n`` is interpreted at the ROBOT's control rate by default. A planner
        that ticks at its own rate (e.g. a simulator at 12 fps) should pass that
        rate as ``frames_hz``; each ``n`` then becomes ``duration = n/frames_hz``
        so the traj runs at the speed the planner simulated, independent of the
        robot's control rate.
        """
        if frames_hz is not None and frames_hz <= 0:
            raise ValueError("frames_hz must be > 0")
        u = np.asarray(u, float)
        if u.ndim != 2 or u.shape[1] != 5:
            raise ValueError("u must have shape (H, 5): (x, y, z, w, n)")
        # An (H, 5) array carries no orientation, so every waypoint holds the
        # previous orientation (quaternion=None). ``hold_orientation`` is accepted
        # for API symmetry but is necessarily True for this position-only format;
        # use ``from_pose_array`` (H,9) if you need to command orientation.
        if not hold_orientation:
            raise ValueError(
                "from_waypoint_array ingests position-only (H,5) actions and cannot "
                "set orientation; use from_pose_array((H,9)) or build "
                "CartesianWaypoint(s) with quaternions instead."
            )
        wpts: List[CartesianWaypoint] = []
        for x, y, z, w, n in u:
            grip = GripperCommand.from_normalized(w, span=gripper_span, force=gripper_force)
            timing = (
                {"duration": max(1, int(round(n))) / float(frames_hz)}
                if frames_hz is not None
                else {"n_frames": max(1, int(round(n)))}
            )
            wpts.append(
                CartesianWaypoint(
                    position=[x, y, z],
                    quaternion=None,
                    gripper=grip,
                    **timing,
                )
            )
        return cls(waypoints=wpts, **traj_kwargs)

    @classmethod
    def from_pose_array(
        cls,
        u: np.ndarray,
        *,
        gripper_force: float = 20.0,
        gripper_span: float = 0.08,
        frames_hz: Optional[float] = None,
        **traj_kwargs,
    ) -> "CartesianTrajectory":
        """Build a traj from an ``(H, 9)`` array that carries orientation.

        Each row is ``(x, y, z, qw, qx, qy, qz, w, n)``: an SE(3) pose (quaternion
        w-first) + normalised gripper ``w`` in ``[0, 1]`` + frame count ``n``. This
        is the orientation-carrying sibling of :meth:`from_waypoint_array`, for
        policies/planners that emit per-step orientation (full SE(3) action
        trajs, e.g. ACT/openpi-style). Combine with ``representation=`` and
        ``n_execute=`` via ``traj_kwargs`` for relative / receding-horizon use.
        """
        if frames_hz is not None and frames_hz <= 0:
            raise ValueError("frames_hz must be > 0")
        u = np.asarray(u, float)
        if u.ndim != 2 or u.shape[1] != 9:
            raise ValueError("u must have shape (H, 9): (x,y,z, qw,qx,qy,qz, w, n)")
        wpts: List[CartesianWaypoint] = []
        for row in u:
            grip = GripperCommand.from_normalized(row[7], span=gripper_span, force=gripper_force)
            timing = (
                {"duration": max(1, int(round(row[8]))) / float(frames_hz)}
                if frames_hz is not None
                else {"n_frames": max(1, int(round(row[8])))}
            )
            wpts.append(
                CartesianWaypoint(
                    position=row[:3],
                    quaternion=row[3:7],
                    gripper=grip,
                    **timing,
                )
            )
        return cls(waypoints=wpts, **traj_kwargs)

    @classmethod
    def from_topdown_array(
        cls,
        u: np.ndarray,
        *,
        down_quat=(0.0, 0.0, 1.0, 0.0),
        gripper_span: float = 0.08,
        gripper_force: float = 20.0,
        frames_hz: Optional[float] = None,
        **traj_kwargs,
    ) -> "CartesianTrajectory":
        """Build a traj from ``(H, 6)`` top-down waypoints ``(x, y, z, yaw, w, n)``.

        The canonical tabletop-manipulation action: a position, a wrist yaw about
        the base z axis, a normalised gripper command ``w`` in ``[0, 1]``, and a
        frame count ``n``. Each waypoint's orientation is ``yaw`` composed onto
        ``down_quat`` -- YOUR tool-pointing-down quaternion, (w, x, y, z); the
        default ``(0, 0, 1, 0)`` is a 180-degree flip about base y. Every
        top-down planner otherwise re-derives this composition by hand (and the
        quaternion order is easy to get wrong). See :meth:`from_waypoint_array`
        for ``gripper_span`` / ``frames_hz`` semantics.
        """
        from . import transforms as T

        if frames_hz is not None and frames_hz <= 0:
            raise ValueError("frames_hz must be > 0")
        u = np.asarray(u, float)
        if u.ndim != 2 or u.shape[1] != 6:
            raise ValueError("u must have shape (H, 6): (x, y, z, yaw, w, n)")
        wpts: List[CartesianWaypoint] = []
        for x, y, z, yaw, w, n in u:
            grip = GripperCommand.from_normalized(w, span=gripper_span, force=gripper_force)
            timing = (
                {"duration": max(1, int(round(n))) / float(frames_hz)}
                if frames_hz is not None
                else {"n_frames": max(1, int(round(n)))}
            )
            wpts.append(
                CartesianWaypoint(
                    position=[x, y, z],
                    quaternion=T.top_down_quat(yaw, base_quat=down_quat),
                    gripper=grip,
                    **timing,
                )
            )
        return cls(waypoints=wpts, **traj_kwargs)


# ----------------------------------------------------------------------------
# Delta (relative) servo command -- the RL / MPC / teleop workhorse
# ----------------------------------------------------------------------------
@dataclass
class CartesianDelta:
    """A relative end-effector move, integrated on top of the current pose.

    This is the standard RL / MPC / teleop action: ``[dx, dy, dz, rx, ry, rz]``
    plus a gripper command, applied over ``duration`` seconds. The last three are
    an axis-angle *rotation vector* (not roll/pitch/yaw Euler angles), matching
    robosuite/MuJoCo ``OSC_POSE`` -- which is what makes sim->real transfer clean.
    The translation is taken in ``frame`` ("base" or "tcp").
    """

    delta: np.ndarray  # length-6, base or tcp frame
    gripper: Optional[GripperCommand] = None
    duration: float = 0.05  # 20 Hz default control step
    frame: str = "base"

    def __post_init__(self) -> None:
        self.delta = np.asarray(self.delta, float).reshape(CART_DOF)


# ----------------------------------------------------------------------------
# Joint space (reset / home / MoveIt-plan execution)
# ----------------------------------------------------------------------------
@dataclass
class JointGripperTarget:
    """Exact gripper target synchronized with a joint segment.

    ``velocity=None`` derives the velocity needed to cover the width delta in
    the segment's scheduled ticks. A supplied velocity asserts the same timing
    contract. This contract uses fire-and-forget ``Gripper.Move`` only.
    """

    width: float
    force: float = 20.0
    velocity: Optional[float] = None

    def __post_init__(self) -> None:
        self.width = float(self.width)
        self.force = float(self.force)
        if not np.isfinite(self.width):
            raise ValueError("JointGripperTarget.width must be finite")
        if not np.isfinite(self.force):
            raise ValueError("JointGripperTarget.force must be finite")
        if self.velocity is not None:
            self.velocity = float(self.velocity)
            if not np.isfinite(self.velocity) or self.velocity <= 0.0:
                raise ValueError("JointGripperTarget.velocity must be finite and > 0")


@dataclass
class JointGripperForceTarget:
    """Signed native gripper-force target synchronized with a joint segment.

    The backend dispatches ``Gripper.Grasp(force)``. Positive force closes and
    negative force opens, exactly matching Flexiv RDK. No desired width is part
    of this action; measured width remains observable robot state.
    """

    force: float

    def __post_init__(self) -> None:
        self.force = float(self.force)
        if not np.isfinite(self.force):
            raise ValueError("JointGripperForceTarget.force must be finite")


@dataclass
class JointWaypoint:
    positions: np.ndarray
    n_frames: Optional[int] = None
    duration: Optional[float] = None
    gripper: Optional[
        Union[JointGripperTarget, JointGripperForceTarget, GripperCommand]
    ] = None

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, float).reshape(-1)
        if not np.all(np.isfinite(self.positions)):
            raise ValueError("JointWaypoint.positions must be finite")
        if self.n_frames is None and self.duration is None:
            raise ValueError("JointWaypoint needs n_frames or duration")
        if self.n_frames is not None:
            if isinstance(self.n_frames, (bool, np.bool_)) or int(self.n_frames) != self.n_frames:
                raise ValueError("JointWaypoint.n_frames must be an integer")
            self.n_frames = int(self.n_frames)
            if self.n_frames <= 0:
                raise ValueError("JointWaypoint.n_frames must be positive")
        if self.gripper is not None and isinstance(self.gripper, GripperCommand):
            if self.gripper.grasp:
                raise ValueError(
                    "JointWaypoint requires an exact Gripper.Move target; "
                    "grasp=True has no exact width"
                )
            self.gripper = JointGripperTarget(
                width=self.gripper.width,
                force=self.gripper.force,
                velocity=self.gripper.velocity,
            )

    def resolve_duration(self, control_hz: float) -> float:
        if self.duration is not None:
            return float(self.duration)
        return float(self.n_frames) / float(control_hz)


@dataclass
class JointTrajectory:
    waypoints: List[JointWaypoint]
    # Preserve the legacy positional field order; v0.2.3 fields are appended.
    max_joint_speed_scale: float = 0.3
    interpolation: str = "cosine"
    safety_profile: str = ""
    # Actuator-target knot 0 is intentionally distinct from measured state.
    initial_positions: Optional[np.ndarray] = None
    initial_gripper_width: Optional[float] = None
    # Authoritative n_frames: reject rate violations, never time-stretch.
    strict_timing: bool = False

    def __post_init__(self) -> None:
        if not self.waypoints:
            raise ValueError("JointTrajectory needs at least one waypoint")
        if self.initial_positions is not None:
            self.initial_positions = np.asarray(self.initial_positions, float).reshape(-1)
            if not np.all(np.isfinite(self.initial_positions)):
                raise ValueError("initial_positions must be finite")
        if self.initial_gripper_width is not None:
            self.initial_gripper_width = float(self.initial_gripper_width)
            if not np.isfinite(self.initial_gripper_width):
                raise ValueError("initial_gripper_width must be finite")
        self.max_joint_speed_scale = float(self.max_joint_speed_scale)
        if (
            not np.isfinite(self.max_joint_speed_scale)
            or not 0.0 < self.max_joint_speed_scale <= 1.0
        ):
            raise ValueError("max_joint_speed_scale must be finite and in (0, 1]")
        if self.interpolation not in {"cosine", "linear"}:
            raise ValueError("JointTrajectory.interpolation must be 'cosine' or 'linear'")
        self.strict_timing = bool(self.strict_timing)
        if self.strict_timing:
            if self.initial_positions is None:
                raise ValueError("strict_timing requires explicit initial_positions")
            for wp in self.waypoints:
                if wp.n_frames is None or wp.duration is not None:
                    raise ValueError(
                        "strict_timing requires n_frames (and no duration) on every JointWaypoint"
                    )
            if (
                any(isinstance(wp.gripper, JointGripperTarget) for wp in self.waypoints)
                and self.initial_gripper_width is None
            ):
                raise ValueError("strict_timing gripper waypoints require initial_gripper_width")


@dataclass
class JointTorqueWaypoint:
    """One smooth arm-torque endpoint and its exact controller duration."""

    torques: np.ndarray
    n_frames: int
    gripper: Optional[
        Union[JointGripperTarget, JointGripperForceTarget, GripperCommand]
    ] = None

    def __post_init__(self) -> None:
        self.torques = np.asarray(self.torques, dtype=float).reshape(-1)
        if not np.all(np.isfinite(self.torques)):
            raise ValueError("JointTorqueWaypoint.torques must be finite")
        if isinstance(self.n_frames, (bool, np.bool_)):
            raise ValueError("JointTorqueWaypoint.n_frames must be an integer")
        self.n_frames = int(self.n_frames)
        if self.n_frames <= 0:
            raise ValueError("JointTorqueWaypoint.n_frames must be positive")
        if isinstance(self.gripper, GripperCommand):
            if self.gripper.grasp:
                self.gripper = JointGripperForceTarget(
                    force=self.gripper.force
                )
            else:
                self.gripper = JointGripperTarget(
                    width=self.gripper.width,
                    force=self.gripper.force,
                    velocity=self.gripper.velocity,
                )


@dataclass
class JointTorqueTrajectory:
    """Gravity-compensated arm torque plus synchronized gripper effort.

    ``initial_torques`` is the command immediately before the first emitted
    tick. Every segment linearly approaches its endpoint, satisfying Flexiv's
    requirement that streamed torque commands remain smooth and continuous.
    """

    waypoints: List[JointTorqueWaypoint]
    initial_torques: np.ndarray
    max_joint_torque_scale: float = 0.30
    safety_profile: str = ""

    def __post_init__(self) -> None:
        if not self.waypoints:
            raise ValueError("JointTorqueTrajectory needs at least one waypoint")
        self.initial_torques = np.asarray(
            self.initial_torques, dtype=float
        ).reshape(-1)
        if not np.all(np.isfinite(self.initial_torques)):
            raise ValueError("initial_torques must be finite")
        if any(
            waypoint.torques.shape != self.initial_torques.shape
            for waypoint in self.waypoints
        ):
            raise ValueError("all torque waypoints must match initial_torques")
        self.max_joint_torque_scale = float(self.max_joint_torque_scale)
        if (
            not np.isfinite(self.max_joint_torque_scale)
            or not 0.0 < self.max_joint_torque_scale <= 1.0
        ):
            raise ValueError("max_joint_torque_scale must be in (0, 1]")


# ----------------------------------------------------------------------------
# Execution report -- quantifies the "execution" failure category
# ----------------------------------------------------------------------------
@dataclass
class ExecutionResult:
    """Returned by every blocking execution call.

    A receding-horizon planner's failure taxonomy typically has an "execution"
    bucket. This object turns
    that bucket into something observable and quantifiable rather than a guess:
    the planner logs ``clipped``/``stop_reason``/``path_tracking_error`` and can
    attribute a bad outcome to execution vs perception/ranking with evidence.
    """

    success: bool = True
    clipped: bool = False
    stop_reason: str = "none"
    executed_duration: float = 0.0
    path_tracking_error: float = 0.0  # max ||pose_cmd - pose_meas|| over run
    max_tcp_speed: float = 0.0
    max_joint_speed: float = 0.0
    max_wrench: float = 0.0
    gripper_width_final: float = 0.0
    final_state: Optional["object"] = None  # RobotState; Optional to avoid import cycle
    log: dict = field(default_factory=dict)

    def summary(self) -> str:
        """One-line human-readable rendering, so consumers log the whole result
        instead of printing one field and silently dropping a protective stop."""
        return (
            f"{'ok' if self.success else 'STOPPED'} stop={self.stop_reason} "
            f"dur={self.executed_duration:.2f}s clip={self.clipped} "
            f"track_err={1000.0 * self.path_tracking_error:.1f}mm "
            f"grip={self.gripper_width_final:.4f}m"
        )
