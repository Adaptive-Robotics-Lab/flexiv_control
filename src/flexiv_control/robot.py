"""The :class:`Robot` facade -- the one API everything uses.

A receding-horizon planner, an MPC loop, an RL policy, and a SpaceMouse bridge all talk to this
same object. They never touch a backend, ROS topic, or RDK struct directly.

Design notes
------------
* ``execute_cartesian_trajectory`` expands the traj to setpoints, runs the fixed-rate
  loop in Python (the "NRT / modest-rate" tier), applies the safety filter every
  tick, and returns an :class:`ExecutionResult` -- which is what turns a planner's
  "execution" failure category into real numbers.
* For the lowest-latency path, point this at the C++ RT daemon via the network
  client (``flexiv_control.client.RemoteRobot``) instead; the API is identical.
* Lease + stop are here so a single process is well-behaved; the *server*
  enforces the lease across multiple processes (RL + MPC + teleop can't fight
  over the arm).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Optional

import numpy as np

from .trajectory import (
    CartesianTrajectory,
    CartesianDelta,
    CartesianWaypoint,
    ExecutionResult,
    JointGripperForceTarget,
    JointTorqueTrajectory,
    JointTrajectory,
    JointWaypoint,
)
from .backends import RobotBackend, get_backend
from .config import RobotConfig, load_safety_profile
from .interpolation import (
    CartesianTrajectoryInterpolator,
    JointTorqueTrajectoryInterpolator,
    JointTrajectoryInterpolator,
    delta_to_target_pose,
)
from .safety import SafetyFilter, SafetyProfile
from .types import (
    ControlMode,
    ForceControlParams,
    GripperCommand,
    ImpedanceParams,
    JointImpedanceParams,
    RobotState,
    SafetyStatus,
    StopReason,
)

# Settle window for the deferred gripper-close tracking gate (see
# execute_cartesian_trajectory): at a segment boundary the arm still trails its
# command by its normal dynamic lag (~15-25 mm at transit speed on a Rizon),
# so the gate must not sample there. After this long holding one commanded
# pose, a healthy arm has converged to a few mm while a contact-stalled arm
# still shows the full remaining distance. Bounded to half the close segment
# so short segments still resolve in-segment.
GRIP_GATE_SETTLE_S = 0.35


class LeaseError(RuntimeError):
    pass


class TrajectoryStoppedError(RuntimeError):
    """Raised by ``execute_*_traj(..., raise_on_stop=True)`` when the safety
    filter (or a cancel request) aborted the traj. Carries the full
    :class:`ExecutionResult` so the caller can inspect what happened."""

    def __init__(self, result: ExecutionResult):
        super().__init__(f"traj stopped: {result.summary()}")
        self.result = result


class TrajectoryPrevalidationError(ValueError):
    """A joint trajectory was rejected before any backend write."""


class Robot:
    def __init__(
        self,
        config: Optional[RobotConfig] = None,
        backend: Optional[RobotBackend] = None,
        control_hz: Optional[float] = None,
        safety_profile: Optional[str] = None,
    ):
        self.cfg = config or RobotConfig()
        self.control_hz = float(control_hz or self.cfg.control_hz)
        self.dt = 1.0 / self.control_hz
        self.backend = backend or get_backend(self.cfg.backend, **self._backend_kwargs())
        self._owner: Optional[str] = None
        self.profile: SafetyProfile = load_safety_profile(
            safety_profile or self.cfg.default_safety_profile
        )
        self.filter = SafetyFilter(self.profile, self.dt)
        self._joint_velocity_limits = np.full(
            self.cfg.n_joints,
            2.0,
            dtype=float,
        )
        self._joint_torque_limits: Optional[np.ndarray] = None
        # Torque streaming is stateful across receding-horizon RPCs.  The next
        # trajectory must begin at the last command actually acknowledged by
        # this Robot instance (zero after connect, stop, or a mode switch), so
        # a client cannot introduce an unverified first-tick torque jump.
        self._last_joint_torque_command = np.zeros(
            self.cfg.n_joints,
            dtype=float,
        )
        self._active_stream_mode: Optional[ControlMode] = None
        self._effective_joint_contract: Optional[dict] = None
        self._apply_runtime_joint_contract()
        # Cooperative cancel: another thread (e.g. the server's stop handler)
        # sets this and the executing traj loop aborts at its next tick.
        self._cancel = threading.Event()
        # Last state read from the backend -- a cheap, lock-free snapshot the
        # server can serve while a blocking traj owns the backend.
        self._last_state: Optional[RobotState] = None

    def _backend_kwargs(self) -> dict:
        """Per-backend construction kwargs drawn from the config."""
        b = self.cfg.backend.lower()
        if b in ("flexiv_rdk", "rdk", "flexiv"):
            return dict(
                robot_sn=self.cfg.robot_sn,
                gripper_name=self.cfg.gripper_name,
                allow_torque=self.cfg.allow_joint_torque,
            )
        if b in ("mujoco", "mjx"):
            return dict(
                model_path=self.cfg.model_path,
                n_joints=self.cfg.n_joints,
                # Default the sim substep to the control period so a traj plays
                # back at the same speed it is streamed; honour an explicit override.
                control_dt=self.cfg.control_dt if self.cfg.control_dt is not None else self.dt,
                tcp_site=self.cfg.mujoco_tcp_site,
                gripper_actuator=self.cfg.mujoco_gripper_actuator,
                gripper_width_scale=self.cfg.mujoco_gripper_width_scale,
                gripper_width_offset=self.cfg.mujoco_gripper_width_offset,
            )
        return {}

    # -- construction helpers ------------------------------------------------
    @classmethod
    def from_config(cls, path_or_name: str, **overrides) -> "Robot":
        cfg = RobotConfig.load(path_or_name)
        return cls(config=cfg, **overrides)

    # -- lifecycle -----------------------------------------------------------
    def connect(self) -> None:
        self.backend.connect()
        self._reset_joint_torque_continuity()
        self._apply_runtime_joint_contract()

    def _reset_joint_torque_continuity(self) -> None:
        self._last_joint_torque_command = np.zeros(
            self.cfg.n_joints,
            dtype=float,
        )
        self._active_stream_mode = None

    def _apply_runtime_joint_contract(self) -> None:
        """Intersect configured limits with cached hardware/firmware facts."""
        backend_info = dict(self.backend.runtime_info())
        runtime = backend_info.get("joint_limits")
        lower = np.asarray(self.profile.joint_lower, dtype=float).reshape(-1)
        upper = np.asarray(self.profile.joint_upper, dtype=float).reshape(-1)
        velocity = np.full(lower.shape, 2.0, dtype=float)
        torque: Optional[np.ndarray] = None
        sources = ["configured_safety_profile"]

        if runtime is not None:
            runtime_lower = np.asarray(
                runtime["position_min_rad"],
                dtype=float,
            ).reshape(-1)
            runtime_upper = np.asarray(
                runtime["position_max_rad"],
                dtype=float,
            ).reshape(-1)
            runtime_velocity = np.asarray(
                runtime["velocity_max_rad_s"],
                dtype=float,
            ).reshape(-1)
            runtime_torque_raw = runtime.get("torque_max_nm")
            runtime_torque = (
                None
                if runtime_torque_raw is None
                else np.asarray(runtime_torque_raw, dtype=float).reshape(-1)
            )
            if not (
                runtime_lower.shape == runtime_upper.shape == runtime_velocity.shape == lower.shape
            ):
                raise RuntimeError(
                    "runtime RobotInfo joint limit dimensions do not match "
                    "the configured safety profile"
                )
            lower = np.maximum(lower, runtime_lower)
            upper = np.minimum(upper, runtime_upper)
            velocity = runtime_velocity
            if runtime_torque is not None:
                if (
                    runtime_torque.shape != lower.shape
                    or not np.all(np.isfinite(runtime_torque))
                    or np.any(runtime_torque <= 0.0)
                ):
                    raise RuntimeError(
                        "runtime RobotInfo joint torque limits are invalid"
                    )
                torque = runtime_torque
            sources.append(str(runtime.get("source", "runtime_joint_limits")))
        elif self.cfg.backend.lower() in ("flexiv_rdk", "rdk", "flexiv"):
            # A hardware deployment must never fall back to the old uniform
            # 2 rad/s assumption.
            if self.backend.is_connected:
                raise RuntimeError(
                    "connected Flexiv RDK backend did not provide runtime joint limits"
                )

        current = backend_info.get("current_safety_limits")
        if current is not None:
            current_lower = np.asarray(
                current["position_min_rad"],
                dtype=float,
            ).reshape(-1)
            current_upper = np.asarray(
                current["position_max_rad"],
                dtype=float,
            ).reshape(-1)
            current_normal = np.asarray(
                current["velocity_max_normal_rad_s"],
                dtype=float,
            ).reshape(-1)
            current_reduced = np.asarray(
                current["velocity_max_reduced_rad_s"],
                dtype=float,
            ).reshape(-1)
            if not (
                current_lower.shape
                == current_upper.shape
                == current_normal.shape
                == current_reduced.shape
                == lower.shape
            ):
                raise RuntimeError(
                    "runtime SafetyLimits dimensions do not match the configured safety profile"
                )
            lower = np.maximum(lower, current_lower)
            upper = np.minimum(upper, current_upper)
            # Which firmware state is active can change asynchronously; the
            # smaller ceiling is safe in both normal and reduced operation.
            velocity = np.minimum(
                velocity,
                np.minimum(current_normal, current_reduced),
            )
            sources.append(str(current.get("source", "current_safety_limits")))

        if (
            not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower >= upper)
        ):
            raise RuntimeError("effective joint position limit intersection is invalid")
        if not np.all(np.isfinite(velocity)) or np.any(velocity <= 0.0):
            raise RuntimeError("effective per-joint velocity limits are invalid")

        self.profile.joint_lower = lower
        self.profile.joint_upper = upper
        self._joint_velocity_limits = velocity
        self._joint_torque_limits = torque
        self.filter.set_profile(self.profile)
        self.filter.set_joint_velocity_limits(velocity)
        enforced_lower = lower + self.profile.joint_margin_rad
        enforced_upper = upper - self.profile.joint_margin_rad
        if np.any(enforced_lower >= enforced_upper):
            raise RuntimeError("joint margin collapses the effective position interval")
        contract = {
            "sources": sources,
            "hard_position_min_rad": lower.tolist(),
            "hard_position_max_rad": upper.tolist(),
            "enforced_position_min_rad": enforced_lower.tolist(),
            "enforced_position_max_rad": enforced_upper.tolist(),
            "base_velocity_max_rad_s": velocity.tolist(),
            "max_joint_speed_scale": float(self.profile.max_joint_speed_scale),
            "max_joint_torque_scale": float(
                self.profile.max_joint_torque_scale
            ),
        }
        if torque is not None:
            contract["base_torque_max_nm"] = torque.tolist()
        payload = json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        contract["sha256"] = hashlib.sha256(payload).hexdigest()
        self._effective_joint_contract = contract

    def server_runtime_info(self) -> dict:
        """Return the server's cached runtime contract without a state read."""
        backend_info = dict(self.backend.runtime_info())
        info = {
            "control_hz": float(self.control_hz),
            "active_safety_profile": self.profile.name,
        }
        for key in (
            "runtime_hardware_identity",
            "gripper_limits",
            "joint_limits",
            "current_safety_limits",
        ):
            value = backend_info.get(key)
            if value is not None:
                info[key] = value
        if self._effective_joint_contract is not None:
            info["effective_joint_limits"] = dict(self._effective_joint_contract)
        return info

    def disconnect(self) -> None:
        self._reset_joint_torque_continuity()
        self.backend.disconnect()

    def __enter__(self) -> "Robot":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        try:
            self.stop()
        finally:
            self.disconnect()

    # -- lease (single-process; the server enforces cross-process) ----------
    def acquire_lease(self, owner: str) -> None:
        if self._owner is not None and self._owner != owner:
            raise LeaseError(f"robot already leased by {self._owner!r}")
        self._owner = owner

    def release_lease(self) -> None:
        self._owner = None

    def _check_lease(self) -> None:
        if self._owner is None:
            # Single-process convenience: auto-lease to "default".
            self._owner = "default"

    # -- safety profile ------------------------------------------------------
    def set_safety_profile(self, name_or_path: str) -> None:
        self.profile = load_safety_profile(name_or_path)
        # Recompute the intersection; changing a profile can tighten but never
        # expand beyond connected hardware/firmware facts.
        self._apply_runtime_joint_contract()

    # -- state ---------------------------------------------------------------
    def get_state(self) -> RobotState:
        s = self.backend.read_state()
        self._last_state = s
        return s

    def peek_state(self) -> Optional[RobotState]:
        """Latest state already read from the backend, without touching it.

        During a blocking traj the execute loop refreshes this every tick, so
        the server can answer ``get_state`` mid-traj (at most one tick stale)
        instead of blocking on the backend lock for the traj's whole duration.
        """
        return self._last_state

    # -- mode start helpers --------------------------------------------------
    def start_cartesian_impedance(
        self,
        impedance: Optional[ImpedanceParams] = None,
        *,
        realtime: bool = False,
        force_control: Optional[ForceControlParams] = None,
        nullspace_q: Optional[np.ndarray] = None,
    ) -> None:
        mode = (
            ControlMode.RT_CARTESIAN_MOTION_FORCE
            if realtime
            else ControlMode.NRT_CARTESIAN_MOTION_FORCE
        )
        self.backend.set_mode(
            mode,
            impedance=impedance or ImpedanceParams(),
            force_control=force_control,
            nullspace_q=nullspace_q if nullspace_q is not None else self.cfg.q_home,
            max_contact_wrench=self.profile.max_contact_wrench,
        )
        self._reset_joint_torque_continuity()
        self._active_stream_mode = mode

    def start_joint_impedance(
        self,
        joint_impedance: Optional[JointImpedanceParams] = None,
        *,
        realtime: bool = False,
    ) -> None:
        mode = ControlMode.RT_JOINT_IMPEDANCE if realtime else ControlMode.NRT_JOINT_IMPEDANCE
        self.backend.set_mode(mode, joint_impedance=joint_impedance or JointImpedanceParams())
        self._reset_joint_torque_continuity()
        self._active_stream_mode = mode

    def start_joint_torque(self) -> None:
        """Enter the explicitly enabled 1 kHz gravity-compensated torque mode."""
        if not self.cfg.allow_joint_torque:
            raise RuntimeError(
                "joint torque control is disabled by RobotConfig"
            )
        if abs(self.control_hz - 1000.0) > 1e-9:
            raise RuntimeError(
                "RT_JOINT_TORQUE requires control_hz=1000"
            )
        if self._active_stream_mode != ControlMode.RT_JOINT_TORQUE:
            self._last_joint_torque_command = np.zeros(
                self.cfg.n_joints,
                dtype=float,
            )
            self.backend.set_mode(ControlMode.RT_JOINT_TORQUE)
            self._active_stream_mode = ControlMode.RT_JOINT_TORQUE

    # -- the RL / MPC / teleop workhorse ------------------------------------
    def servo_cartesian_delta(
        self,
        delta,
        *,
        duration: Optional[float] = None,
        frame: str = "base",
        gripper: Optional[GripperCommand] = None,
    ) -> ExecutionResult:
        """Apply a relative ``[dx,dy,dz,drx,dry,drz]`` move over ``duration``."""
        self._check_lease()
        if not isinstance(delta, CartesianDelta):
            delta = CartesianDelta(
                delta=delta, duration=duration or self.dt, frame=frame, gripper=gripper
            )
        state = self.get_state()
        target = delta_to_target_pose(delta, state.tcp_pose)
        wp = CartesianWaypoint(
            position=target[:3],
            quaternion=target[3:7],
            gripper=delta.gripper,
            duration=delta.duration,
            frame=delta.frame,
        )
        traj = CartesianTrajectory(
            waypoints=[wp], frame=delta.frame, safety_profile=self.profile.name
        )
        return self.execute_cartesian_trajectory(traj, blocking=True)

    def servo_cartesian_pose(
        self,
        pose: np.ndarray,
        *,
        duration: float = 0.2,
        gripper: Optional[GripperCommand] = None,
    ) -> ExecutionResult:
        pose = np.asarray(pose, float).reshape(7)
        wp = CartesianWaypoint(
            position=pose[:3], quaternion=pose[3:7], gripper=gripper, duration=duration
        )
        return self.execute_cartesian_trajectory(
            CartesianTrajectory(waypoints=[wp], safety_profile=self.profile.name), blocking=True
        )

    # -- planner traj / MPC-horizon / scripted manipulation ---------------
    def _verify_trajectory_profile(self, requested: str, result: ExecutionResult) -> None:
        """Enforce the reproducibility contract on ``traj.safety_profile``.

        Empty = "run under whatever is active". A non-empty name must match the
        active profile, else we raise: silently executing under a different
        envelope than the one the traj was planned for is exactly the failure
        the field exists to prevent. Requested/active are always logged.
        """
        result.log["requested_profile"] = requested
        result.log["active_profile"] = self.profile.name
        if requested and requested != self.profile.name:
            raise ValueError(
                f"traj requests safety profile {requested!r} but the active "
                f"profile is {self.profile.name!r}; call set_safety_profile"
                f"({requested!r}) first or fix the traj"
            )

    def execute_cartesian_trajectory(
        self,
        traj: CartesianTrajectory,
        *,
        blocking: bool = True,
        raise_on_stop: bool = False,
        record: bool = False,
    ) -> ExecutionResult:
        """Execute a Cartesian traj at the control rate with per-tick safety.

        Returns an :class:`ExecutionResult` with tracking error, clipping, stop
        reason, and peak quantities -- the observable signal a planner can log
        under its "execution" failure category.

        * If the backend is not already in a Cartesian motion mode, the NRT
          Cartesian impedance mode is started automatically with the traj's
          ``impedance`` (the documented examples call
          ``start_cartesian_impedance()`` explicitly; forgetting it must not be
          a hardware-only failure).
        * The traj's kinematic/contact envelope tightens the active profile:
          the interpolator runs at ``min(traj cap, profile cap)`` and the
          contact check at the elementwise minimum wrench.
        * A non-empty ``traj.safety_profile`` must match the active profile.
        * ``blocking`` is currently always True (kept for future async parity);
          a concurrent ``stop()``/``request_stop()`` cancels mid-traj. A cancel
          that is already pending at entry aborts THIS traj immediately (a stop
          issued between trajs must not be silently erased by the next one).
        * ``raise_on_stop=True`` raises :class:`TrajectoryStoppedError` instead of
          returning a failed result, so a protective stop cannot be ignored.
        * ``record=True`` fills ``result.log["trajectory"]`` with per-tick rows
          ``[t, *pose_cmd, *pose_meas, *wrench]`` and sets
          ``result.log["stopped_at_waypoint"]`` when the run aborts -- the
          measured-vs-commanded series for sim-vs-real attribution that the
          loop otherwise measures every tick and throws away.
        """
        self._check_lease()
        result = ExecutionResult(success=True, stop_reason=StopReason.NONE.value)
        if self._cancel.is_set():
            self._cancel.clear()
            result.success = False
            result.stop_reason = StopReason.USER.value
            result.log["aborted_at_entry"] = True
            result.final_state = self.get_state()
            if raise_on_stop:
                raise TrajectoryStoppedError(result)
            return result
        self._verify_trajectory_profile(traj.safety_profile, result)

        start = self.get_state()
        if not start.control_mode.is_cartesian:
            self.start_cartesian_impedance(impedance=traj.impedance)
            result.log["mode_autostarted"] = True
            start = self.get_state()
        self.filter.reset(start)
        # Resolve relative-to-start poses against the live start pose and slice to
        # the execution horizon (receding horizon): only the first H_exec waypoints
        # run here; the rest are re-predicted by the planner next cycle.
        traj = traj.for_execution(start.tcp_pose)
        # Tightening-only envelope: a traj may slow itself below the profile's
        # caps but can never relax them.
        lin_cap = self.profile.max_linear_speed
        if traj.max_tcp_linear_speed and traj.max_tcp_linear_speed > 0:
            lin_cap = min(lin_cap, float(traj.max_tcp_linear_speed))
        ang_cap = self.profile.max_angular_speed
        if traj.max_tcp_angular_speed and traj.max_tcp_angular_speed > 0:
            ang_cap = min(ang_cap, float(traj.max_tcp_angular_speed))
        result.log["linear_speed_cap"] = lin_cap
        result.log["angular_speed_cap"] = ang_cap
        wrench_cap = self.profile.max_contact_wrench
        wrench_relaxed = False
        if traj.contact_wrench_allowance is not None:
            # Held-payload headroom: server-clamped to the profile's granted
            # ceiling (default zero), then ADDED to the profile cap. The traj
            # request is a request, never an override.
            allow = np.minimum(traj.contact_wrench_allowance, self.profile.max_wrench_allowance)
            if np.any(allow > 0):
                wrench_cap = wrench_cap + allow
                wrench_relaxed = True
                result.log["contact_wrench_allowance"] = allow.tolist()
        if traj.max_contact_wrench is not None:
            wrench_cap = np.minimum(wrench_cap, traj.max_contact_wrench)
        interp = CartesianTrajectoryInterpolator(
            traj,
            start.tcp_pose,
            self.control_hz,
            max_linear_speed=lin_cap,
            max_angular_speed=ang_cap,
        )
        if wrench_relaxed:
            # Track the firmware guard to the same effective cap (it was armed
            # with the profile value at mode start); restored in the finally.
            self.backend.set_contact_wrench_limit(wrench_cap)

        max_err = 0.0
        max_speed = 0.0
        max_wrench = 0.0
        prev_pos = start.tcp_position.copy()
        prev_cmd_pos = None  # commanded TCP position from the previous tick
        close_aborted = False
        close_issued = False  # a closing command actually reached the gripper
        pending_close: Optional[tuple] = None  # (GripperCommand, fire_at_tick)
        tick_idx = 0
        trajectory: list = [] if record else None
        t0 = time.perf_counter()
        t_loop = time.perf_counter()

        def _try_fire_close(g, state) -> None:
            """Evaluate the tracking gate at the SETTLED instant and fire/abort.

            Deferring to a settle window is what makes the gate separable: at
            a segment boundary the arm still trails its command by its normal
            dynamic lag (~15-25 mm at transit speed on a Rizon), so an
            instantaneous check there cannot tell a healthy descend from a
            stalled one. After ~0.35 s of holding the same commanded pose a
            healthy arm converges to a few mm while a contact-stalled arm
            still shows the full remaining descend distance."""
            nonlocal close_aborted, close_issued
            err_now = (
                float(np.linalg.norm(prev_cmd_pos - state.tcp_position))
                if prev_cmd_pos is not None
                else 0.0
            )
            if close_aborted or result.clipped or err_now > traj.grip_tracking_gate_m:
                if not close_aborted:
                    close_aborted = True
                    result.log["close_aborted"] = {
                        "segment": interp.current_segment,
                        "tracking_error_m": err_now,
                        "gate_m": float(traj.grip_tracking_gate_m),
                        "clipped": bool(result.clipped),
                    }
            else:
                self.backend.move_gripper(g)
                close_issued = True

        try:
            for pose, grip in interp:
                tick_idx += 1
                if self._cancel.is_set():
                    self._cancel.clear()
                    self.backend.stop()
                    result.success = False
                    result.stop_reason = StopReason.USER.value
                    break
                state = self.get_state()
                # Robot-side fault gate: a collision reflex / protective stop /
                # E-stop on the robot is invisible to the host-side geometry filter.
                if self.backend.in_fault():
                    self.backend.stop()
                    result.success = False
                    result.stop_reason = StopReason.BACKEND_FAULT.value
                    break
                # Per-traj contact envelope (tightened or payload-relaxed);
                # the filter below enforces the same effective cap.
                if np.any(np.abs(state.wrench) > wrench_cap):
                    self.backend.stop()
                    result.success = False
                    result.stop_reason = StopReason.CONTACT_WRENCH.value
                    break
                sr = self.filter.filter_cartesian(pose, state, max_contact_wrench=wrench_cap)
                if not sr.ok:
                    self.backend.stop()
                    result.success = False
                    result.stop_reason = sr.reason.value
                    break
                if sr.clipped:
                    result.clipped = True
                self.backend.stream_cartesian(sr.pose, wrench=_traj_wrench(traj))
                # A deferred close whose settle window elapsed fires (or
                # aborts) NOW, against the settled tracking error.
                if pending_close is not None and tick_idx >= pending_close[1]:
                    g = pending_close[0]
                    pending_close = None
                    _try_fire_close(g, state)
                if grip is not None:
                    # Close-intent gate: a stalled/deflected descend (impedance
                    # yield below the wrench cap) leaves the tool at an UNPLANNED
                    # height -- closing there grasps the wrong geometry (rim
                    # pinch). A CLOSING command is DEFERRED by a settle window
                    # (bounded within its segment) and then fired only if the
                    # settled tracking error is inside the gate; once one close
                    # aborts, all later closes this traj abort too (the
                    # follow-up force-grasp would blind-close mid-air). Opens
                    # always pass immediately. A garbage width read (transient
                    # gripper-bus failure returns 0.0) must not misclassify --
                    # the width comparison only counts against a sane positive
                    # reading.
                    closing = bool(grip.grasp) or (
                        float(state.gripper_width) > 1e-6
                        and grip.width < float(state.gripper_width) - 1e-3
                    )
                    # NEVER gate a grasp=True SUSTAIN once a close was actually
                    # issued this traj: the object is (potentially) between the
                    # fingers, and skipping the force-closure hand-off leaves
                    # only a stalled position-hold -- the filmed slide-out
                    # failure -- while mislabeling a physical grasp as
                    # "no close fired". The gate exists to prevent closing at an
                    # UNPLANNED height; after a close it has done its job.
                    gate_active = traj.grip_tracking_gate_m is not None and not (
                        grip.grasp and close_issued
                    )
                    if closing and gate_active:
                        if close_aborted:
                            pass  # sticky: a skipped close invalidates the rest
                        else:
                            if pending_close is not None:
                                # A second close arrived before the first
                                # resolved (very short segments): resolve the
                                # first at the current instant, then defer this
                                # one on its own window.
                                g = pending_close[0]
                                pending_close = None
                                _try_fire_close(g, state)
                            settle = min(
                                int(round(GRIP_GATE_SETTLE_S * self.control_hz)),
                                max(1, int(interp.current_segment_ticks) // 2),
                            )
                            pending_close = (grip, tick_idx + settle)
                    else:
                        self.backend.move_gripper(grip)
                        close_issued = close_issued or closing
                if trajectory is not None:
                    trajectory.append(
                        [
                            time.perf_counter() - t0,
                            *sr.pose.tolist(),
                            *state.tcp_pose.tolist(),
                            *state.wrench.tolist(),
                        ]
                    )

                # bookkeeping. path_tracking_error is the lag between the PREVIOUS
                # tick's command and the CURRENT measurement (a true residual), not
                # the size of this tick's commanded step (skipped on the first tick).
                if prev_cmd_pos is not None:
                    err = float(np.linalg.norm(prev_cmd_pos - state.tcp_position))
                    max_err = max(max_err, err)
                spd = float(np.linalg.norm(state.tcp_position - prev_pos)) / self.dt
                max_speed = max(max_speed, spd)
                max_wrench = max(max_wrench, float(np.max(np.abs(state.wrench))))
                prev_pos = state.tcp_position.copy()
                prev_cmd_pos = sr.pose[:3].copy()

                # maintain control rate
                t_loop += self.dt
                sleep = t_loop - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
            if pending_close is not None:
                # The traj ended inside a settle window (short final segment)
                # or broke out mid-traj. On a clean finish resolve the close
                # against the final settled state; on a stop the motion is
                # gone -- record the abort so the caller never mistakes the
                # untouched OPEN width for an attempted grasp.
                g = pending_close[0]
                pending_close = None
                if result.success:
                    _try_fire_close(g, self.get_state())
                elif not close_aborted:
                    close_aborted = True
                    result.log["close_aborted"] = {
                        "segment": interp.current_segment,
                        "reason": f"traj_stopped:{result.stop_reason}",
                        "gate_m": float(traj.grip_tracking_gate_m),
                        "clipped": bool(result.clipped),
                    }
        finally:
            if wrench_relaxed:
                # Re-arm the firmware guard with the profile cap no matter how
                # the traj ended -- the allowance must never outlive its traj.
                # A restore failure (robot faulted at the last tick, comms
                # hiccup) must not mask the traj's real result: the next
                # set_mode re-arms the firmware with the profile cap anyway,
                # and the host-side guards always enforce the profile values.
                try:
                    self.backend.set_contact_wrench_limit(self.profile.max_contact_wrench)
                except Exception:
                    result.log["wrench_restore_failed"] = True

        end = self.get_state()
        result.executed_duration = end.stamp - start.stamp
        result.path_tracking_error = max_err
        result.max_tcp_speed = max_speed
        result.max_wrench = max_wrench
        result.gripper_width_final = end.gripper_width
        result.final_state = end
        if trajectory is not None:
            result.log["trajectory"] = trajectory
        if not result.success:
            result.log["stopped_at_waypoint"] = interp.current_segment
        if raise_on_stop and not result.success:
            raise TrajectoryStoppedError(result)
        return result

    # -- joint space (reset / home / MoveIt-plan execution) ----------------
    @staticmethod
    def _joint_move_duration(
        q_now: np.ndarray,
        q_target: np.ndarray,
        *,
        duration: Optional[float],
        max_joint_speed: Optional[float],
        default: float = 3.0,
        floor: float = 1.0,
    ) -> float:
        """Resolve a joint move's duration from either a fixed time or a speed cap.

        A recovery move is naturally specified as "go there slowly" (a rad/s
        cap on the largest joint excursion); a fixed duration is dangerously
        fast for a large displacement and pointlessly slow for a small one.
        """
        if duration is not None:
            return float(duration)
        if max_joint_speed is not None and max_joint_speed > 0:
            dq = float(np.max(np.abs(np.asarray(q_target, float) - np.asarray(q_now, float))))
            return max(floor, dq / float(max_joint_speed))
        return default

    def move_joint(
        self,
        q_target: np.ndarray,
        *,
        duration: Optional[float] = None,
        max_joint_speed: Optional[float] = None,
        realtime: bool = False,
    ) -> ExecutionResult:
        """Interpolated joint move. Give either a fixed ``duration`` (seconds)
        or a ``max_joint_speed`` (rad/s) cap on the largest joint excursion;
        with neither, a 3 s default applies."""
        self._check_lease()
        dur = self._joint_move_duration(
            self.get_state().q, q_target, duration=duration, max_joint_speed=max_joint_speed
        )
        self.start_joint_impedance(realtime=realtime)
        traj = JointTrajectory(
            waypoints=[JointWaypoint(positions=np.asarray(q_target, float), duration=dur)],
            safety_profile=self.profile.name,
        )
        return self.execute_joint_trajectory(traj)

    def execute_joint_trajectory(
        self,
        traj: JointTrajectory,
        *,
        raise_on_stop: bool = False,
    ) -> ExecutionResult:
        self._check_lease()
        result = ExecutionResult(success=True)

        def reject_unsafe_entry(state: RobotState) -> Optional[ExecutionResult]:
            reason = StopReason.NONE
            if self.backend.in_fault() or state.safety_status != SafetyStatus.OK:
                reason = (
                    state.stop_reason
                    if state.stop_reason != StopReason.NONE
                    else StopReason.BACKEND_FAULT
                )
            elif np.any(np.abs(state.wrench) > self.profile.max_contact_wrench):
                reason = StopReason.CONTACT_WRENCH
            if reason == StopReason.NONE:
                return None
            result.success = False
            result.stop_reason = reason.value
            result.final_state = state
            result.log["prewrite_safety_rejection"] = True
            result.log["entry_safety_status"] = state.safety_status.value
            if raise_on_stop:
                raise TrajectoryStoppedError(result)
            return result

        if self._cancel.is_set():
            self._cancel.clear()
            result.success = False
            result.stop_reason = StopReason.USER.value
            result.log["aborted_at_entry"] = True
            result.final_state = self.get_state()
            if raise_on_stop:
                raise TrajectoryStoppedError(result)
            return result
        try:
            self._verify_trajectory_profile(traj.safety_profile, result)
            requested_speed_scale = float(traj.max_joint_speed_scale)
            active_speed_scale = float(self.profile.max_joint_speed_scale)
            result.log["requested_max_joint_speed_scale"] = requested_speed_scale
            result.log["active_max_joint_speed_scale"] = active_speed_scale
            if requested_speed_scale > active_speed_scale + 1e-12:
                raise ValueError(
                    "JointTrajectory max_joint_speed_scale "
                    f"{requested_speed_scale:.9g} exceeds active safety-profile "
                    f"ceiling {active_speed_scale:.9g}"
                )
            effective_speed_scale = requested_speed_scale
            result.log["effective_max_joint_speed_scale"] = effective_speed_scale
            result.log["joint_interpolation"] = traj.interpolation
            result.log["strict_timing"] = traj.strict_timing
        except ValueError as exc:
            raise TrajectoryPrevalidationError(str(exc)) from exc

        # Read measured state once, then prevalidate the entire atomic RPC before
        # any mode switch, gripper Move, or arm setpoint can reach the backend.
        start_state = self.get_state()
        entry_rejection = reject_unsafe_entry(start_state)
        if entry_rejection is not None:
            return entry_rejection
        try:
            initial_q = (
                start_state.q.copy()
                if traj.initial_positions is None
                else np.asarray(traj.initial_positions, float).reshape(-1).copy()
            )
            if initial_q.shape != start_state.q.shape:
                raise ValueError(
                    f"initial_positions must match measured q shape {start_state.q.shape}, "
                    f"got {initial_q.shape}"
                )
            joint_filter_anchor = start_state.q.copy()
            if joint_filter_anchor.shape != start_state.q.shape:
                raise ValueError(
                    "joint filter anchor must match measured q shape "
                    f"{start_state.q.shape}, got {joint_filter_anchor.shape}"
                )
            if not np.all(np.isfinite(joint_filter_anchor)):
                raise ValueError("joint filter anchor must be finite")
            effective_joint_speed = self._joint_velocity_limits * effective_speed_scale
            interp = JointTrajectoryInterpolator(
                traj,
                initial_q,
                self.control_hz,
                max_joint_speed=effective_joint_speed,
            )
            first_q = next(iter(interp)) if traj.strict_timing else None

            def first_joint_bound_error(anchor_q: np.ndarray) -> Optional[str]:
                if first_q is None:
                    return None
                first_step = np.abs(first_q - anchor_q)
                first_step_limit = effective_joint_speed * self.dt
                too_fast = first_step > first_step_limit + 1e-12
                if not np.any(too_fast):
                    return None
                joint = int(np.flatnonzero(too_fast)[0])
                return (
                    "strict_timing first emitted joint target exceeds the "
                    "effective one-tick rate bound; "
                    f"joint {joint}: step {first_step[joint]:.9g} rad > "
                    f"{first_step_limit[joint]:.9g} rad at "
                    f"control_hz={self.control_hz:.9g}"
                )

            first_error = first_joint_bound_error(joint_filter_anchor)
            if first_error is not None:
                raise ValueError(first_error)

            joint_lo = self.profile.joint_lower + self.profile.joint_margin_rad
            joint_hi = self.profile.joint_upper - self.profile.joint_margin_rad
            if traj.strict_timing:
                for label, target in [
                    ("initial_positions", initial_q),
                    *[
                        (f"waypoints[{i}].positions", wp.positions)
                        for i, wp in enumerate(traj.waypoints)
                    ],
                ]:
                    if np.any(target < joint_lo) or np.any(target > joint_hi):
                        raise ValueError(
                            f"strict_timing {label} exceeds effective joint position limits"
                        )

            initial_gripper = (
                start_state.gripper_width
                if traj.initial_gripper_width is None
                else float(traj.initial_gripper_width)
            )
            gripper_execution_anchor = float(start_state.gripper_width)
            if not np.isfinite(gripper_execution_anchor):
                raise ValueError("gripper execution anchor must be finite")
            runtime_gripper = self.backend.runtime_info().get("gripper_limits")
            if runtime_gripper is not None:
                width_lo = float(runtime_gripper["min_width_m"])
                width_hi = float(runtime_gripper["max_width_m"])
                for label, width in (
                    ("initial_gripper_width", initial_gripper),
                    ("gripper execution anchor", gripper_execution_anchor),
                ):
                    if not width_lo <= width <= width_hi:
                        raise ValueError(
                            f"{label}={width!r} outside runtime [{width_lo}, {width_hi}] "
                            "from flexivrdk.Gripper.params"
                        )
            gripper_events: list[dict] = []
            previous_width = gripper_execution_anchor
            previous_force: Optional[float] = None
            boundary_tick = 0
            for segment, (wp, requested_ticks, scheduled_ticks) in enumerate(
                zip(
                    traj.waypoints,
                    interp.requested_segment_ticks,
                    interp.scheduled_segment_ticks,
                )
            ):
                boundary_tick += scheduled_ticks
                if wp.gripper is None:
                    continue
                target = wp.gripper
                if isinstance(target, JointGripperForceTarget):
                    if runtime_gripper is not None:
                        force_lo = float(runtime_gripper["min_force_n"])
                        force_hi = float(runtime_gripper["max_force_n"])
                        if not force_lo <= target.force <= force_hi:
                            raise ValueError(
                                f"JointWaypoint {segment} gripper force={target.force!r} "
                                f"outside runtime [{force_lo}, {force_hi}] from "
                                "flexivrdk.Gripper.params"
                            )
                    gripper_events.append(
                        {
                            "mode": "force",
                            "segment": segment,
                            "requested_segment_ticks": requested_ticks,
                            "scheduled_segment_ticks": scheduled_ticks,
                            "start_tick": boundary_tick - scheduled_ticks,
                            "end_tick": boundary_tick,
                            "force_n": target.force,
                            "dispatch": previous_force != target.force,
                        }
                    )
                    previous_force = target.force
                    continue
                delta_width = abs(target.width - previous_width)
                segment_duration = scheduled_ticks * self.dt
                required_velocity = delta_width / segment_duration
                velocity = required_velocity if target.velocity is None else target.velocity
                if delta_width > 1e-12 and target.velocity is not None:
                    if not np.isclose(target.velocity, required_velocity, rtol=1e-6, atol=1e-9):
                        raise ValueError(
                            f"JointWaypoint {segment} gripper velocity {target.velocity:.9g} "
                            f"m/s does not realize width delta {delta_width:.9g} m in "
                            f"scheduled n_frames={scheduled_ticks}; required "
                            f"{required_velocity:.9g} m/s"
                        )
                if runtime_gripper is not None:
                    checks = [
                        ("width", target.width, "min_width_m", "max_width_m"),
                        ("force", target.force, "min_force_n", "max_force_n"),
                    ]
                    if delta_width > 1e-12:
                        checks.append(
                            ("velocity", velocity, "min_velocity_m_s", "max_velocity_m_s")
                        )
                    for field, value, lo_key, hi_key in checks:
                        lo = float(runtime_gripper[lo_key])
                        hi = float(runtime_gripper[hi_key])
                        if not np.isfinite(value) or not lo <= value <= hi:
                            raise ValueError(
                                f"JointWaypoint {segment} gripper {field}={value!r} "
                                f"outside runtime [{lo}, {hi}] from "
                                "flexivrdk.Gripper.params"
                            )
                gripper_events.append(
                    {
                        "mode": "move",
                        "segment": segment,
                        "requested_segment_ticks": requested_ticks,
                        "scheduled_segment_ticks": scheduled_ticks,
                        "start_tick": boundary_tick - scheduled_ticks,
                        "end_tick": boundary_tick,
                        "start_width_m": previous_width,
                        "target_width_m": target.width,
                        "velocity_m_s": velocity,
                        "requested_velocity_m_s": target.velocity,
                        "force_n": target.force,
                        "dispatch": delta_width > 1e-12,
                    }
                )
                previous_width = target.width
                previous_force = None
        except ValueError as exc:
            raise TrajectoryPrevalidationError(str(exc)) from exc

        result.log["base_joint_velocity_limits_rad_s"] = self._joint_velocity_limits.tolist()
        if self._effective_joint_contract is not None:
            result.log["effective_joint_limits_sha256"] = self._effective_joint_contract["sha256"]
        result.log["control_hz"] = float(self.control_hz)
        result.log["requested_duration_s"] = interp.requested_duration_s
        result.log["requested_segment_ticks"] = list(interp.requested_segment_ticks)
        result.log["requested_total_ticks"] = int(interp.requested_total_ticks)
        result.log["scheduled_segment_ticks"] = list(interp.scheduled_segment_ticks)
        result.log["scheduled_total_ticks"] = int(interp.scheduled_total_ticks)
        result.log["scheduled_duration_s"] = float(interp.scheduled_total_ticks * self.dt)
        result.log["initial_joint_target"] = initial_q.tolist()
        result.log["joint_filter_anchor"] = joint_filter_anchor.tolist()
        result.log["ending_joint_target"] = traj.waypoints[-1].positions.tolist()
        result.log["initial_gripper_target_m"] = initial_gripper
        result.log["gripper_execution_anchor_m"] = gripper_execution_anchor
        result.log["ending_gripper_target_m"] = previous_width
        result.log["ending_gripper_force_n"] = previous_force
        result.log["gripper_events"] = gripper_events
        result.log["gripper_tracking"] = []

        start_state = self.get_state()
        entry_rejection = reject_unsafe_entry(start_state)
        if entry_rejection is not None:
            return entry_rejection
        if start_state.control_mode.is_cartesian or start_state.control_mode == ControlMode.IDLE:
            self.start_joint_impedance()
            result.log["mode_autostarted"] = True
            start_state = self.get_state()
        # Re-anchor after a possible mode transition. The first loop snapshot
        # below refreshes this once more immediately before any actuator write.
        joint_filter_anchor = start_state.q.copy()
        gripper_execution_anchor = float(start_state.gripper_width)
        result.log["joint_filter_anchor"] = joint_filter_anchor.tolist()
        result.log["gripper_execution_anchor_m"] = gripper_execution_anchor
        self.filter.reset(start_state, q=joint_filter_anchor)

        events_by_segment = {event["segment"]: event for event in gripper_events}
        active_gripper_target: Optional[float] = gripper_execution_anchor
        active_gripper_force: Optional[float] = None
        first_gripper_event_checked = False
        t_loop = time.perf_counter()
        execution_started = t_loop
        streamed_ticks = 0
        max_joint_speed = 0.0
        max_tracking_error = 0.0
        previous_command = joint_filter_anchor
        for q in interp:
            if self._cancel.is_set():
                self._cancel.clear()
                self.backend.stop()
                result.success = False
                result.stop_reason = StopReason.USER.value
                break
            state = self.get_state()
            if self.backend.in_fault():
                self.backend.stop()
                result.success = False
                result.stop_reason = StopReason.BACKEND_FAULT.value
                break
            if np.any(np.abs(state.wrench) > self.profile.max_contact_wrench):
                self.backend.stop()
                result.success = False
                result.stop_reason = StopReason.CONTACT_WRENCH.value
                break
            if streamed_ticks == 0:
                # This is the freshest telemetry available before the first
                # Gripper.Move or joint stream. Recheck instead of trusting the
                # earlier atomic-prevalidation snapshot across a mode switch.
                joint_filter_anchor = state.q.copy()
                joint_error = first_joint_bound_error(joint_filter_anchor)
                if joint_error is not None:
                    self.backend.stop()
                    result.success = False
                    result.stop_reason = StopReason.JOINT_LIMIT.value
                    result.log["predispatch_joint_rejection"] = joint_error
                    break
                gripper_execution_anchor = float(state.gripper_width)
                result.log["joint_filter_anchor"] = joint_filter_anchor.tolist()
                result.log["gripper_execution_anchor_m"] = gripper_execution_anchor
                self.filter.reset(state, q=joint_filter_anchor)
                previous_command = joint_filter_anchor.copy()
                active_gripper_target = gripper_execution_anchor
                active_gripper_force = None
            if interp.current_segment_tick == 1:
                event = events_by_segment.get(interp.current_segment)
                if event is not None:
                    event["measured_width_at_dispatch_m"] = state.gripper_width
                    if event["mode"] == "move" and not first_gripper_event_checked:
                        first_gripper_event_checked = True
                        measured_width = float(state.gripper_width)
                        delta_width = abs(event["target_width_m"] - measured_width)
                        segment_duration = event["scheduled_segment_ticks"] * self.dt
                        required_velocity = delta_width / segment_duration
                        requested_velocity = event["requested_velocity_m_s"]
                        velocity = (
                            required_velocity if requested_velocity is None else requested_velocity
                        )
                        gripper_error = None
                        if delta_width > 1e-12 and requested_velocity is not None:
                            if not np.isclose(
                                requested_velocity,
                                required_velocity,
                                rtol=1e-6,
                                atol=1e-9,
                            ):
                                gripper_error = (
                                    "first gripper event velocity no longer realizes "
                                    "the measured width delta over the remaining "
                                    f"segment: requested {requested_velocity:.9g} m/s, "
                                    f"required {required_velocity:.9g} m/s"
                                )
                        if runtime_gripper is not None and gripper_error is None:
                            width_lo = float(runtime_gripper["min_width_m"])
                            width_hi = float(runtime_gripper["max_width_m"])
                            if not width_lo <= measured_width <= width_hi:
                                gripper_error = (
                                    f"measured gripper width {measured_width:.9g} m "
                                    f"outside runtime [{width_lo}, {width_hi}]"
                                )
                            elif delta_width > 1e-12:
                                velocity_lo = float(runtime_gripper["min_velocity_m_s"])
                                velocity_hi = float(runtime_gripper["max_velocity_m_s"])
                                if not velocity_lo <= velocity <= velocity_hi:
                                    gripper_error = (
                                        f"first gripper event velocity {velocity:.9g} m/s "
                                        f"outside runtime [{velocity_lo}, {velocity_hi}] "
                                        "after measured-width rebase"
                                    )
                        if gripper_error is not None:
                            self.backend.stop()
                            result.success = False
                            result.stop_reason = StopReason.GRIPPER_LIMIT.value
                            result.log["predispatch_gripper_rejection"] = gripper_error
                            break
                        event["prevalidated_start_width_m"] = event["start_width_m"]
                        event["start_width_m"] = measured_width
                        event["velocity_m_s"] = velocity
                        event["dispatch"] = delta_width > 1e-12
                        gripper_execution_anchor = measured_width
                        result.log["gripper_execution_anchor_m"] = measured_width
                    if event["mode"] == "force" and event["dispatch"]:
                        self.backend.move_gripper(
                            GripperCommand(
                                width=float(state.gripper_width),
                                force=event["force_n"],
                                grasp=True,
                            )
                        )
                        active_gripper_target = None
                        active_gripper_force = event["force_n"]
                    elif event["mode"] == "move" and event["dispatch"]:
                        self.backend.move_gripper(
                            GripperCommand(
                                width=event["target_width_m"],
                                force=event["force_n"],
                                velocity=event["velocity_m_s"],
                                grasp=False,
                            )
                        )
                        active_gripper_target = event["target_width_m"]
                        active_gripper_force = None
            if not result.success:
                break
            sr = self.filter.filter_joint(q, state)
            if not sr.ok or (traj.strict_timing and sr.clipped):
                self.backend.stop()
                result.success = False
                result.stop_reason = sr.reason.value if not sr.ok else StopReason.JOINT_LIMIT.value
                result.log["strict_runtime_clip_refused"] = bool(sr.clipped)
                break
            if sr.clipped:
                result.clipped = True
            self.backend.stream_joint(sr.q)
            streamed_ticks += 1
            max_joint_speed = max(
                max_joint_speed,
                float(np.max(np.abs(sr.q - previous_command)) / self.dt),
            )
            max_tracking_error = max(max_tracking_error, float(np.linalg.norm(sr.q - state.q)))
            previous_command = sr.q.copy()
            result.log["gripper_tracking"].append(
                {
                    "tick": streamed_ticks,
                    "target_width_m": active_gripper_target,
                    "target_force_n": active_gripper_force,
                    "measured_width_m": state.gripper_width,
                    "error_m": (
                        None
                        if active_gripper_target is None
                        else active_gripper_target - state.gripper_width
                    ),
                }
            )
            t_loop += self.dt
            sleep = t_loop - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
        result.executed_duration = time.perf_counter() - execution_started
        result.log["streamed_ticks"] = int(streamed_ticks)
        result.path_tracking_error = max_tracking_error
        result.max_joint_speed = max_joint_speed
        result.final_state = self.get_state()
        result.gripper_width_final = result.final_state.gripper_width
        result.log["final_gripper_tracking_error_m"] = (
            None
            if previous_force is not None
            else previous_width - result.final_state.gripper_width
        )
        if raise_on_stop and not result.success:
            raise TrajectoryStoppedError(result)
        return result

    def execute_joint_torque_trajectory(
        self,
        traj: JointTorqueTrajectory,
        *,
        raise_on_stop: bool = False,
    ) -> ExecutionResult:
        """Stream a prevalidated smooth torque trajectory at exactly 1 kHz."""
        self._check_lease()
        result = ExecutionResult(success=True)
        try:
            self._verify_trajectory_profile(traj.safety_profile, result)
        except ValueError as exc:
            raise TrajectoryPrevalidationError(str(exc)) from exc
        if not self.cfg.allow_joint_torque:
            raise TrajectoryPrevalidationError(
                "joint torque control is disabled by RobotConfig"
            )
        if abs(self.control_hz - 1000.0) > 1e-9:
            raise TrajectoryPrevalidationError(
                "JointTorqueTrajectory requires control_hz=1000"
            )
        if self._joint_torque_limits is None:
            raise TrajectoryPrevalidationError(
                "backend did not publish RobotInfo.tau_max"
            )
        requested_scale = float(traj.max_joint_torque_scale)
        active_scale = float(self.profile.max_joint_torque_scale)
        if requested_scale > active_scale + 1e-12:
            raise TrajectoryPrevalidationError(
                "JointTorqueTrajectory torque scale exceeds active profile"
            )
        command_limit = self._joint_torque_limits * requested_scale
        expected_initial = (
            self._last_joint_torque_command
            if self._active_stream_mode == ControlMode.RT_JOINT_TORQUE
            else np.zeros(self.cfg.n_joints, dtype=float)
        )
        if (
            traj.initial_torques.shape != expected_initial.shape
            or not np.array_equal(traj.initial_torques, expected_initial)
        ):
            raise TrajectoryPrevalidationError(
                "initial_torques do not match the last acknowledged torque "
                "command (zero is required after connect, stop, or mode switch)"
            )
        commands = JointTorqueTrajectoryInterpolator(traj, self.control_hz)
        segment_ticks = [waypoint.n_frames for waypoint in traj.waypoints]
        for label, value in [
            ("initial_torques", traj.initial_torques),
            *[
                (f"waypoints[{index}].torques", waypoint.torques)
                for index, waypoint in enumerate(traj.waypoints)
            ],
        ]:
            if value.shape != command_limit.shape:
                raise TrajectoryPrevalidationError(
                    f"{label} must have {command_limit.size} values"
                )
            if np.any(np.abs(value) > command_limit + 1e-12):
                raise TrajectoryPrevalidationError(
                    f"{label} exceeds scaled RobotInfo.tau_max"
                )

        runtime_gripper = self.backend.runtime_info().get("gripper_limits")
        for index, waypoint in enumerate(traj.waypoints):
            target = waypoint.gripper
            if isinstance(target, JointGripperForceTarget) and runtime_gripper:
                if not (
                    float(runtime_gripper["min_force_n"])
                    <= target.force
                    <= float(runtime_gripper["max_force_n"])
                ):
                    raise TrajectoryPrevalidationError(
                        f"waypoints[{index}] gripper force outside runtime limits"
                    )

        state = self.get_state()
        if (
            self.backend.in_fault()
            or state.safety_status != SafetyStatus.OK
            or np.any(np.abs(state.wrench) > self.profile.max_contact_wrench)
        ):
            result.success = False
            result.stop_reason = StopReason.BACKEND_FAULT.value
            result.final_state = state
            return result
        self.start_joint_torque()
        events = {
            sum(item.n_frames for item in traj.waypoints[:index]): item.gripper
            for index, item in enumerate(traj.waypoints)
            if item.gripper is not None
        }
        start = time.perf_counter()
        next_tick = start
        streamed = 0
        last_gripper_force_commanded: Optional[float] = None
        dispatched_gripper_events: list[dict] = []
        for command in commands:
            if self._cancel.is_set():
                self._cancel.clear()
                self.backend.stop()
                self._reset_joint_torque_continuity()
                result.success = False
                result.stop_reason = StopReason.USER.value
                break
            state = self.get_state()
            if self.backend.in_fault() or np.any(
                np.abs(state.wrench) > self.profile.max_contact_wrench
            ):
                self.backend.stop()
                self._reset_joint_torque_continuity()
                result.success = False
                result.stop_reason = (
                    StopReason.BACKEND_FAULT.value
                    if self.backend.in_fault()
                    else StopReason.CONTACT_WRENCH.value
                )
                break
            target = events.get(streamed)
            if isinstance(target, JointGripperForceTarget):
                self.backend.move_gripper(
                    GripperCommand(
                        width=state.gripper_width,
                        force=target.force,
                        grasp=True,
                    )
                )
                last_gripper_force_commanded = float(target.force)
                dispatched_gripper_events.append({
                    "mode": "force",
                    "segment": int(commands.current_segment),
                    "force_n": float(target.force),
                })
            elif target is not None:
                self.backend.move_gripper(
                    GripperCommand(
                        width=target.width,
                        force=target.force,
                        velocity=target.velocity or 0.1,
                        grasp=False,
                    )
                )
                # The last gripper action is now an exact-width Move, so there
                # is no longer an acknowledged direct-force endpoint.
                last_gripper_force_commanded = None
                dispatched_gripper_events.append({
                    "mode": "move",
                    "segment": int(commands.current_segment),
                    "width_m": float(target.width),
                    "force_limit_n": float(target.force),
                })
            self.backend.stream_joint_torque(command)
            self._last_joint_torque_command = np.asarray(
                command,
                dtype=float,
            ).copy()
            streamed += 1
            next_tick += self.dt
            delay = next_tick - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)
        result.executed_duration = time.perf_counter() - start
        result.log.update({
            "control_mode": ControlMode.RT_JOINT_TORQUE.value,
            "streamed_ticks": streamed,
            "gravity_compensation": True,
            "soft_limits": True,
            "base_torque_max_nm": self._joint_torque_limits.tolist(),
            "requested_max_joint_torque_scale": requested_scale,
            "effective_torque_max_nm": command_limit.tolist(),
            "requested_segment_ticks": segment_ticks,
            "scheduled_segment_ticks": segment_ticks,
            "requested_total_ticks": sum(segment_ticks),
            "scheduled_total_ticks": sum(segment_ticks),
            "strict_timing": True,
            "acknowledged_ending_joint_torque_nm": (
                self._last_joint_torque_command.tolist()
                if result.success else None
            ),
            # Both names carry a physical Newton command, never a planner
            # latent or measured aperture. Keep the historical alias while
            # exposing the acknowledgement semantics explicitly.
            "ending_gripper_force_n": (
                last_gripper_force_commanded if result.success else None
            ),
            "acknowledged_ending_gripper_force_n": (
                last_gripper_force_commanded if result.success else None
            ),
            # Report events actually dispatched before a stop, rather than all
            # events merely present in the requested trajectory.
            "gripper_events": dispatched_gripper_events,
        })
        result.final_state = self.get_state()
        result.gripper_width_final = result.final_state.gripper_width
        if raise_on_stop and not result.success:
            raise TrajectoryStoppedError(result)
        return result

    # -- gripper / home / stop ----------------------------------------------
    def command_gripper(
        self, cmd: GripperCommand, *, wait: bool = False, timeout: float = 5.0
    ) -> Optional[float]:
        """Send a gripper command; with ``wait=True``, block until the fingers
        settle and return the final width in metres (``None`` without ``wait``,
        matching ``RemoteRobot.command_gripper``). Without ``wait`` this is
        fire-and-forget (RDK ``Gripper.Move``/``Grasp`` return immediately),
        which is why every consumer that needs "open, then proceed" used to
        fabricate a do-nothing motion traj just to ride its blocking executor.

        Settle detection: a ``Move`` command must reach its commanded width.
        Only a true ``Grasp`` command may settle at an unchanged, obstructed
        width.  Treating stillness as success for ``Move`` is unsafe on GN01:
        the hardware can report ``is_moving=False`` during its actuation
        latency, which previously let an open command return while the fingers
        were still closed."""
        self._check_lease()
        initial = self.get_state().gripper_width
        self.backend.move_gripper(cmd)
        if not wait:
            return None
        t0 = time.time()
        prev_width = None
        moved = False
        state = self.get_state()
        while time.time() - t0 < timeout:
            if self._cancel.is_set():
                break  # a stop request also ends a gripper wait
            state = self.get_state()
            if state.gripper_is_moving or abs(state.gripper_width - initial) > 1e-3:
                moved = True
            settled_target = (not cmd.grasp) and abs(state.gripper_width - cmd.width) < 2e-3
            settled_grasp = (
                cmd.grasp
                and prev_width is not None
                and abs(state.gripper_width - prev_width) < 5e-4
                and not state.gripper_is_moving
                and (moved or time.time() - t0 >= 0.5)
            )
            if settled_target or settled_grasp:
                break
            prev_width = state.gripper_width
            time.sleep(0.05)
        return state.gripper_width

    def zero_ft_sensor(self) -> None:
        """Zero (bias-calibrate) the 6-DoF F/T sensor. Required after power-on
        before any force-control mode; clears event 301004. The manual zero in
        Elements' MANUAL mode does NOT carry into remote/RDK mode -- run this in
        the remote session, arm at rest with no contact. No-op on backends without
        an F/T sensor (fake/mujoco)."""
        self._check_lease()
        self.backend.zero_ft_sensor()

    def home(
        self,
        q: Optional[np.ndarray] = None,
        *,
        max_joint_speed: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> None:
        """Move to the canonical posture: ``q`` if given, else the config's
        ``q_home``; then command ``cfg.gripper_home_width`` if configured, so
        "home" means the full recorded posture (joints AND gripper), not just
        joints."""
        self._check_lease()
        q_home = np.asarray(q if q is not None else self.cfg.q_home, float)
        try:
            if q is not None or duration is not None or max_joint_speed is not None:
                # An explicit target/timing always uses the interpolated move so
                # the caller's posture (not the vendor primitive's factory home)
                # is what the arm reaches.
                raise NotImplementedError
            self.backend.home(q_home)
        except NotImplementedError:
            self.move_joint(
                q_home,
                duration=duration,
                max_joint_speed=max_joint_speed if max_joint_speed is not None else 0.3,
            )
        if self.cfg.gripper_home_width is not None:
            self.command_gripper(
                GripperCommand(width=float(self.cfg.gripper_home_width)), wait=True
            )

    def go_home_safe(
        self,
        *,
        q_home: Optional[np.ndarray] = None,
        lift_m: float = 0.10,
        open_gripper_width: Optional[float] = None,
        max_tcp_speed: float = 0.10,
        max_joint_speed: float = 0.3,
    ) -> ExecutionResult:
        """The standard end-of-session exit ritual as one resilient call:
        lift straight up -> open the gripper (and wait) -> joint-move home.

        Every robot-facing app needs exactly this block in a ``finally`` clause;
        hand-rolling it is where restore bugs live. Each stage tolerates the
        previous one failing (a recovery path must not give up halfway), and the
        returned result is the home move's, with the lift/gripper outcomes in
        ``result.log``."""
        self._check_lease()
        log: dict = {}
        contact_abort = False
        # 1) lift straight up, orientation held, capped slow.
        try:
            s = self.get_state()
            target = s.tcp_position.copy()
            target[2] = min(target[2] + float(lift_m), self.profile.ws_z[1])
            lift_traj = CartesianTrajectory(
                waypoints=[
                    CartesianWaypoint(
                        position=target,
                        quaternion=None,
                        duration=max(1.5, float(lift_m) / max(max_tcp_speed, 1e-3)),
                    )
                ],
                max_tcp_linear_speed=max_tcp_speed,
            )
            lift_result = self.execute_cartesian_trajectory(lift_traj)
            log["lift"] = lift_result.summary()
            contact_abort = lift_result.stop_reason in (
                StopReason.CONTACT_WRENCH.value,
                StopReason.BACKEND_FAULT.value,
            )
        except Exception as e:  # recovery continues even if the lift fails
            log["lift"] = f"FAILED: {type(e).__name__}: {e}"
        # 2) open the gripper and wait for it.
        width = open_gripper_width
        if width is None:
            width = self.cfg.gripper_home_width
        if width is not None:
            try:
                self.command_gripper(GripperCommand(width=float(width)), wait=True)
                log["gripper"] = f"opened to {float(width):.4f} m"
            except Exception as e:
                log["gripper"] = f"FAILED: {type(e).__name__}: {e}"
        # A stop request or a contact/fault during the lift makes the blind
        # joint-space sweep toward home the WRONG move: the arm is plausibly
        # snagged on the scene, and a joint home from there can drag the TCP
        # through the table. Hand control back to the operator instead.
        if contact_abort or self._cancel.is_set():
            result = ExecutionResult(
                success=False,
                stop_reason=(
                    StopReason.USER.value
                    if self._cancel.is_set()
                    else StopReason.CONTACT_WRENCH.value
                ),
            )
            log["home"] = "skipped: lift ended in contact/fault or a stop was requested"
            result.final_state = self.get_state()
            result.log.update(log)
            return result
        # 3) joint-move to the canonical posture, speed-capped.
        q_target = np.asarray(q_home if q_home is not None else self.cfg.q_home, float)
        result = self.move_joint(q_target, max_joint_speed=max_joint_speed)
        result.log.update(log)
        return result

    def request_stop(self) -> None:
        """Cooperative cancel: the executing traj loop aborts at its next tick
        (StopReason ``user``). Safe to call from another thread; does not touch
        the backend itself, so it cannot race the executing thread's stream."""
        self._cancel.set()

    def clear_stop(self) -> None:
        """Clear a PENDING cooperative cancel that nothing consumed.

        For session boundaries only: a dying client's disconnect handler
        requests a safety stop, and when no motion is in flight nothing
        consumes the latched flag -- it then instant-aborts the NEXT
        session's first traj (``stop=user dur=0.00``, observed live). The
        server clears it when a FRESH owner acquires the lease; never call
        this while another party's motion may be in flight."""
        self._cancel.clear()

    def stop(self) -> None:
        self._cancel.set()
        self.backend.stop()
        self._reset_joint_torque_continuity()


def _traj_wrench(traj: CartesianTrajectory):
    if traj.force_control is not None and np.any(traj.force_control.enabled_axes):
        return traj.force_control.target_wrench
    return None
