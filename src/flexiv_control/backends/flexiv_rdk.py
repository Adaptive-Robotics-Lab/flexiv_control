"""Flexiv RDK backend -- the real Rizon arm.

This wraps Flexiv RDK (Python module ``flexivrdk``). It supports both RT modes
(``Stream*`` at up to 1 kHz; the robot runs the hard real-time motion-force /
impedance loop internally) and NRT modes (``Send*``; the robot's internal motion
generator interpolates discrete commands, tolerant of a non-RT host).

IMPORTANT, please read before running on hardware
--------------------------------------------------
* RT streaming requires the **Professional** RDK license; NRT works on Standard.
* Enable "Remote/RDK" mode on the robot via Flexiv Elements first.
* RDK attribute/enum names have changed across versions (e.g. the v1.x renames
  to math-symbol members). Every spot that depends on a specific name is marked
  ``# VERIFY:``. Run ``tests/hardware_smoke.py`` and the read-only example
  before commanding motion, and fix any name that differs on your RDK version.
* The guarded import means importing this module never fails; the clear error is
  raised only if you actually try to ``connect()`` without ``flexivrdk``.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import numpy as np

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

try:  # guarded import: never crash on machines without RDK installed
    import flexivrdk  # type: ignore

    _RDK_AVAILABLE = True
    _RDK_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on environment
    flexivrdk = None  # type: ignore
    _RDK_AVAILABLE = False
    _RDK_IMPORT_ERROR = exc


# Map our ControlMode -> flexivrdk.Mode. Built lazily because the enum only
# exists once flexivrdk is importable. The RT_* modes are not exposed by every
# RDK build (e.g. flexivrdk 1.7's Mode enum has ONLY IDLE + NRT_* + UNKNOWN), so
# each entry is added ONLY if its member exists -- otherwise building the table
# AttributeError'd on the first SwitchMode (the NRT pick-place workload never
# needs the RT modes). # VERIFY enum member names per RDK version.
def _rdk_mode(mode: ControlMode):
    M = flexivrdk.Mode
    spec = {
        ControlMode.IDLE: "IDLE",
        ControlMode.NRT_JOINT_POSITION: "NRT_JOINT_POSITION",
        ControlMode.NRT_JOINT_IMPEDANCE: "NRT_JOINT_IMPEDANCE",
        ControlMode.NRT_CARTESIAN_MOTION_FORCE: "NRT_CARTESIAN_MOTION_FORCE",
        ControlMode.NRT_PRIMITIVE: "NRT_PRIMITIVE_EXECUTION",
        ControlMode.RT_JOINT_POSITION: "RT_JOINT_POSITION",
        ControlMode.RT_JOINT_IMPEDANCE: "RT_JOINT_IMPEDANCE",
        ControlMode.RT_CARTESIAN_MOTION_FORCE: "RT_CARTESIAN_MOTION_FORCE",
        ControlMode.RT_JOINT_TORQUE: "RT_JOINT_TORQUE",
    }
    table = {cm: getattr(M, name) for cm, name in spec.items() if hasattr(M, name)}
    if mode not in table:
        raise RuntimeError(
            f"control mode {mode} maps to flexivrdk.Mode.{spec[mode]}, which this "
            f"flexivrdk build does not expose (available: {[n for n in spec.values() if hasattr(M, n)]})")
    return table[mode]


def _get(obj: Any, *names: str, default=None):
    """Return the first present attribute from ``names`` (cross-version safe)."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def _required_attr(obj: Any, *names: str, context: str) -> Any:
    """Read a required cross-version RDK field without a synthetic default."""
    sentinel = object()
    value = _get(obj, *names, default=sentinel)
    if value is sentinel:
        raise RuntimeError(
            f"Flexiv RDK {context} is missing required field "
            f"{'/'.join(names)}"
        )
    return value


def _required_vector(
    obj: Any,
    *names: str,
    size: int,
    context: str,
) -> np.ndarray:
    value = np.asarray(
        _required_attr(obj, *names, context=context),
        dtype=float,
    ).reshape(-1)
    if value.shape != (size,):
        raise RuntimeError(
            f"Flexiv RDK {context}.{names[0]} has shape {value.shape}, "
            f"expected ({size},)"
        )
    if not np.all(np.isfinite(value)):
        raise RuntimeError(
            f"Flexiv RDK {context}.{names[0]} contains non-finite values"
        )
    return value


def _required_float(obj: Any, name: str, *, context: str) -> float:
    value = float(_required_attr(obj, name, context=context))
    if not np.isfinite(value):
        raise RuntimeError(
            f"Flexiv RDK {context}.{name} is not finite"
        )
    return value


def _rdk_coord(frame: str):
    """Map a frame name to ``flexivrdk.CoordType``.

    RDK's ``SetForceControlFrame`` takes a ``CoordType`` enum (``WORLD``/``TCP``),
    *not* a string. "tcp"/"flange"/"ee" -> TCP; everything else ("base"/"world")
    -> WORLD. Confirmed against RDK v1.x ``robot.hpp``.
    """
    C = flexivrdk.CoordType
    return C.TCP if str(frame).lower() in ("tcp", "flange", "ee") else C.WORLD


def _warn_rdk_version() -> None:
    """Warn if the installed flexivrdk is outside the supported v1.x range.

    Enforces (as a soft check) the version discipline docs/versions.md documents:
    this backend targets RDK v1.x; v0.x and v2.x have incompatible APIs.
    """
    v = getattr(flexivrdk, "__version__", None)
    if not v:
        return
    try:
        major = int(str(v).split(".")[0])
    except Exception:
        return
    if major != 1:
        import warnings

        warnings.warn(
            f"flexivrdk {v} detected, but this backend targets RDK v1.x. "
            "v0.x and v2.x have incompatible APIs (constructor, state fields, "
            "command form); see docs/versions.md. Pin flexivrdk>=1.5,<2.",
            RuntimeWarning,
            stacklevel=2,
        )


class FlexivRdkBackend(RobotBackend):
    def __init__(
        self,
        robot_sn: str,
        n_joints: int = 7,
        gripper_name: Optional[str] = None,
        *,
        allow_torque: bool = False,
    ):
        if not _RDK_AVAILABLE:
            raise ImportError(
                "flexivrdk is not installed. Install with "
                "`pip install flexivrdk` (match the version to your robot "
                f"software). Original import error: {_RDK_IMPORT_ERROR}"
            )
        self.robot_sn = robot_sn
        self.n_joints = n_joints
        self._gripper_name = gripper_name
        self._allow_torque = bool(allow_torque)
        self._robot = None
        self._gripper = None
        self._mode = ControlMode.IDLE
        self._connected = False
        self._runtime_info: dict = {}
        self._gripper_limits: Optional[dict[str, float | str]] = None
        # The active Cartesian force-control params, kept so stream_cartesian can
        # actually command the configured target_wrench (not just a zero default).
        self._force_control: Optional[ForceControlParams] = None

    # -- lifecycle ----------------------------------------------------------
    def connect(self) -> None:
        _warn_rdk_version()
        self._robot = flexivrdk.Robot(self.robot_sn)  # VERIFY: ctor signature

        # Clear any minor fault, then enable + wait until operational.
        fault_fn = _get(self._robot, "fault", default=None)
        if callable(fault_fn) and fault_fn():
            ok = self._robot.ClearFault()  # VERIFY: v1.x returns bool, older None
            if ok is False:
                raise RuntimeError(
                    "could not clear robot fault -- check the pendant; an engaged "
                    "E-stop or active fault must be cleared before Enable()."
                )
        self._robot.Enable()
        t0 = time.time()
        while not self._robot.operational():  # VERIFY: operational() vs Operational
            if time.time() - t0 > 10.0:
                raise TimeoutError("robot did not become operational within 10 s")
            time.sleep(0.05)

        # Adopt the true DoF from the robot rather than trusting the constructor
        # default, and assert any explicit n_joints matches (a mismatch would
        # silently mis-size every command/state vector).
        try:
            qv = _get(self._robot.states(), "q", default=None)
            n = len(list(qv)) if qv is not None else 0
            if n:
                if self.n_joints and self.n_joints != n:
                    raise ValueError(
                        f"configured n_joints={self.n_joints} but the robot reports "
                        f"{n} joints; fix the config."
                    )
                self.n_joints = n
        except ValueError:
            raise
        except Exception:
            pass

        # Cache immutable hardware facts from the already-connected RDK owner.
        # get_server_info serves only this snapshot; it never opens a second
        # robot connection or performs a state read.
        robot_info = self._robot.info()
        actual_serial = str(
            _required_attr(
                robot_info,
                "serial_num",
                context="RobotInfo",
            )
        ).strip()
        if not actual_serial:
            raise RuntimeError("Flexiv RDK RobotInfo.serial_num is empty")
        if self.robot_sn and actual_serial != self.robot_sn:
            raise RuntimeError(
                f"configured robot_sn={self.robot_sn!r}, but RDK reports "
                f"serial_num={actual_serial!r}"
            )
        dof = int(
            _required_attr(robot_info, "DoF", context="RobotInfo")
        )
        if dof != self.n_joints:
            raise RuntimeError(
                f"RDK RobotInfo.DoF={dof} disagrees with n_joints="
                f"{self.n_joints}"
            )
        robot_q_min = _required_vector(
            robot_info,
            "q_min",
            size=dof,
            context="RobotInfo",
        )
        robot_q_max = _required_vector(
            robot_info,
            "q_max",
            size=dof,
            context="RobotInfo",
        )
        robot_dq_max = _required_vector(
            robot_info,
            "dq_max",
            size=dof,
            context="RobotInfo",
        )
        if np.any(robot_q_min >= robot_q_max):
            raise RuntimeError(
                "Flexiv RDK RobotInfo joint position limits are not ordered"
            )
        if np.any(robot_dq_max <= 0.0):
            raise RuntimeError(
                "Flexiv RDK RobotInfo joint velocity limits must be > 0"
            )
        tool = flexivrdk.Tool(self._robot)
        tool_name = str(tool.name()).strip()
        if not tool_name:
            raise RuntimeError(
                "Flexiv RDK current tool profile name is empty"
            )
        self._runtime_info = {
            "runtime_hardware_identity": {
                "robot_serial": actual_serial,
                "robot_model": str(
                    _required_attr(
                        robot_info,
                        "model_name",
                        context="RobotInfo",
                    )
                ),
                "robot_software_version": str(
                    _required_attr(
                        robot_info,
                        "software_ver",
                        context="RobotInfo",
                    )
                ),
                "flexivrdk_version": str(
                    getattr(flexivrdk, "__version__", "unknown")
                ),
                "tool_profile": tool_name,
            },
            "joint_limits": {
                "source": "flexivrdk.Robot.info",
                "position_min_rad": robot_q_min.tolist(),
                "position_max_rad": robot_q_max.tolist(),
                "velocity_max_rad_s": robot_dq_max.tolist(),
            },
        }

        # Safety.current_limits() is read-only, but constructing Safety requires
        # the configured safety password. Never hard-code or expose it. If the
        # deployment supplies FLEXIV_RDK_SAFETY_PASSWORD, cache the actual
        # active firmware limits; otherwise RobotInfo limits remain available.
        safety_password = os.environ.get("FLEXIV_RDK_SAFETY_PASSWORD")
        if safety_password:
            safety = flexivrdk.Safety(self._robot, safety_password)
            current = safety.current_limits()
            current_q_min = _required_vector(
                current,
                "q_min",
                size=dof,
                context="SafetyLimits",
            )
            current_q_max = _required_vector(
                current,
                "q_max",
                size=dof,
                context="SafetyLimits",
            )
            current_dq_normal = _required_vector(
                current,
                "dq_max_normal",
                size=dof,
                context="SafetyLimits",
            )
            current_dq_reduced = _required_vector(
                current,
                "dq_max_reduced",
                size=dof,
                context="SafetyLimits",
            )
            self._runtime_info["current_safety_limits"] = {
                "source": "flexivrdk.Safety.current_limits",
                "position_min_rad": current_q_min.tolist(),
                "position_max_rad": current_q_max.tolist(),
                "velocity_max_normal_rad_s": (
                    current_dq_normal.tolist()
                ),
                "velocity_max_reduced_rad_s": (
                    current_dq_reduced.tolist()
                ),
            }

        # Force-control modes (NRT_CARTESIAN_MOTION_FORCE -- our cartesian impedance)
        # REQUIRE the 6-DoF F/T sensor to be zeroed first, else SwitchMode faults with
        # event 301004 ("FT sensor is not calibrated using primitive [ZeroFTSensor]").
        # Zero it once here, at connect, with the arm at rest (no contact) -- a no-op on
        # robots without an F/T sensor. HARDWARE-VERIFIED on Rizon4s-062626.
        try:
            M = flexivrdk.Mode
            if hasattr(M, "NRT_PRIMITIVE_EXECUTION"):
                self._robot.SwitchMode(M.NRT_PRIMITIVE_EXECUTION)
                self._robot.ExecutePrimitive("ZeroFTSensor", dict())
                t0 = time.time()
                while self._robot.busy() and time.time() - t0 < 10.0:
                    time.sleep(0.2)
                self._robot.SwitchMode(M.IDLE)
        except Exception as e:  # pragma: no cover - hardware-only path
            import warnings
            warnings.warn(
                f"ZeroFTSensor at connect failed ({e}); entering a force-control mode "
                f"may fault until the F/T sensor is zeroed.", RuntimeWarning, stacklevel=2)

        if self._gripper_name:  # non-empty device name required to enable a gripper
            self._gripper = flexivrdk.Gripper(self._robot)  # VERIFY
            # RDK v1.x gripper bring-up is TWO steps and ORDER MATTERS:
            #   1. Enable(name) -- enable the NAMED gripper DEVICE (the name is the
            #      one Flexiv Elements -> Settings -> Device reports for the gripper).
            #   2. Init()       -- home/initialize the fingers (blocking).
            # Calling Init() WITHOUT Enable(name) first fails with
            # "[flexiv::rdk::Gripper::Init] No gripper enabled" and leaves every
            # gripper command a silent no-op -- which is exactly the trap this used
            # to fall into (it tried Init() first and never reached Enable). A
            # configured gripper is part of the hardware contract, so fail connect.
            try:
                if hasattr(self._gripper, "Enable"):
                    self._gripper.Enable(self._gripper_name)
                if hasattr(self._gripper, "Init"):
                    self._gripper.Init()
                params = self._gripper.params()
                limits: dict[str, float | str] = {
                    "source": "flexivrdk.Gripper.params",
                    "device_name": str(
                        _required_attr(
                            params,
                            "name",
                            context="GripperParams",
                        )
                    ),
                    "min_width_m": _required_float(
                        params,
                        "min_width",
                        context="GripperParams",
                    ),
                    "max_width_m": _required_float(
                        params,
                        "max_width",
                        context="GripperParams",
                    ),
                    "min_velocity_m_s": _required_float(
                        params,
                        "min_vel",
                        context="GripperParams",
                    ),
                    "max_velocity_m_s": _required_float(
                        params,
                        "max_vel",
                        context="GripperParams",
                    ),
                    "min_force_n": _required_float(
                        params,
                        "min_force",
                        context="GripperParams",
                    ),
                    "max_force_n": _required_float(
                        params,
                        "max_force",
                        context="GripperParams",
                    ),
                }
                for lo, hi in (
                    ("min_width_m", "max_width_m"),
                    ("min_velocity_m_s", "max_velocity_m_s"),
                    ("min_force_n", "max_force_n"),
                ):
                    if float(limits[lo]) > float(limits[hi]):
                        raise RuntimeError(
                            f"Flexiv RDK gripper limits {lo}/{hi} are "
                            "not ordered"
                        )
                self._gripper_limits = limits
                self._runtime_info["gripper_limits"] = dict(limits)
                self._runtime_info[
                    "runtime_hardware_identity"
                ]["gripper_device"] = str(limits["device_name"])
            except Exception as e:  # pragma: no cover - hardware-only path
                self._gripper = None
                self._gripper_limits = None
                raise RuntimeError(
                    f"gripper init failed for configured device "
                    f"{self._gripper_name!r}; refusing to start without its "
                    "runtime state and limits"
                ) from e
        elif self._gripper_name is not None:
            # Explicit empty name ("") = no gripper configured; skip cleanly (no
            # scary warning) rather than attempting an init that cannot succeed.
            self._gripper = None
        self._connected = True

    def runtime_info(self) -> dict:
        """Return connect-time hardware facts without touching the robot."""
        return {
            key: (
                dict(value)
                if isinstance(value, dict)
                else value
            )
            for key, value in self._runtime_info.items()
        }

    def disconnect(self) -> None:
        if self._robot is not None:
            try:
                self._robot.Stop()
            except Exception:
                pass
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # -- state --------------------------------------------------------------
    def read_state(self) -> RobotState:
        # Truthful control mode: a robot-side fault / e-stop / power event
        # resets the ARM to IDLE while this cache still reports the last
        # commanded motion mode. The executor trusts the reported mode for its
        # auto-(re)start, so a stale cache turns every subsequent Send* into
        # '[SendCartesianMotionForce] Robot is not in an applicable control
        # mode' (observed live after an e-stop recovery). Sync from hardware.
        if self._mode != ControlMode.IDLE:
            try:
                if self._robot.mode() == _rdk_mode(ControlMode.IDLE):
                    self._mode = ControlMode.IDLE
            except Exception:  # pragma: no cover - defensive: mode() is optional info
                pass
        s = self._robot.states()  # VERIFY: states() returns RobotStates
        q = _required_vector(
            s, "q", size=self.n_joints, context="RobotStates"
        )
        dq = _required_vector(
            s,
            "dq",
            "dtheta",
            size=self.n_joints,
            context="RobotStates",
        )
        tau = _required_vector(
            s, "tau", size=self.n_joints, context="RobotStates"
        )
        tcp_pose = _required_vector(
            s,
            "tcp_pose",
            "tcpPose",
            size=7,
            context="RobotStates",
        )
        tcp_vel = _required_vector(
            s,
            "tcp_vel",
            "tcpVel",
            size=CART_DOF,
            context="RobotStates",
        )
        wrench = _required_vector(
            s,
            "ext_wrench_in_tcp",
            "ext_wrench_in_world",
            "extWrenchInTcp",
            size=CART_DOF,
            context="RobotStates",
        )

        gw, gf, gm = 0.0, 0.0, False
        if self._gripper is None and self._gripper_name:
            raise RuntimeError(
                f"configured gripper {self._gripper_name!r} is unavailable; "
                "refusing to fabricate gripper state"
            )
        if self._gripper is not None:
            gs = self._gripper.states()  # VERIFY
            gw = _required_float(gs, "width", context="GripperStates")
            gf = _required_float(gs, "force", context="GripperStates")
            gm = bool(
                _required_attr(
                    gs,
                    "is_moving",
                    "isMoving",
                    context="GripperStates",
                )
            )

        # Surface a robot-side fault through the state so the control loop (and
        # any client polling get_state) sees it -- previously hardcoded OK.
        faulted = self.in_fault()
        return RobotState(
            stamp=time.time(),
            q=q, dq=dq, tau=tau,
            tcp_pose=tcp_pose, tcp_vel=tcp_vel, wrench=wrench,
            gripper_width=gw, gripper_force=gf, gripper_is_moving=gm,
            control_mode=self._mode,
            safety_status=SafetyStatus.FAULT if faulted else SafetyStatus.OK,
            stop_reason=StopReason.BACKEND_FAULT if faulted else StopReason.NONE,
        )

    def in_fault(self) -> bool:
        """True if the robot reports a fault / protective stop (RDK ``fault()``)."""
        fault_fn = _get(self._robot, "fault", default=None)
        if not callable(fault_fn):
            return True
        try:
            return bool(fault_fn())
        except Exception:
            return True

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
        if mode == ControlMode.RT_JOINT_TORQUE and not self._allow_torque:
            # Active gate (not just gating-by-omission): direct joint torque
            # bypasses the robot's impedance safety loop, so it is opt-in only.
            raise RuntimeError(
                "RT_JOINT_TORQUE is gated off. Construct "
                "FlexivRdkBackend(..., allow_torque=True) to enable direct "
                "joint-torque control (expert/research mode)."
            )
        self._robot.SwitchMode(_rdk_mode(mode))  # VERIFY
        self._mode = mode

        if impedance is not None and mode.is_cartesian:
            self._robot.SetCartesianImpedance(  # VERIFY arg order/name
                list(impedance.stiffness), list(impedance.damping_ratio)
            )
        if joint_impedance is not None and not mode.is_cartesian:
            self._robot.SetJointImpedance(
                list(joint_impedance.stiffness), list(joint_impedance.damping_ratio)
            )
        if max_contact_wrench is not None and mode.is_cartesian:
            self._robot.SetMaxContactWrench(list(np.asarray(max_contact_wrench, float)))
        if force_control is not None and mode.is_cartesian:
            self._force_control = force_control
            self._robot.SetForceControlAxis(list(map(bool, force_control.enabled_axes)))
            # SetForceControlFrame takes a CoordType enum, not a string. (RDK v1.x)
            self._robot.SetForceControlFrame(_rdk_coord(force_control.frame))
        elif not mode.is_cartesian:
            self._force_control = None  # force control is scoped to Cartesian modes
        if nullspace_q is not None and mode.is_cartesian:
            self._robot.SetNullSpacePosture(list(np.asarray(nullspace_q, float)))

    def set_contact_wrench_limit(self, wrench: np.ndarray) -> None:
        # Only meaningful (and only accepted by the RDK) in Cartesian modes;
        # the mode-start value comes from set_mode, this updates it mid-mode.
        if self._mode.is_cartesian:
            self._robot.SetMaxContactWrench(list(np.asarray(wrench, float).reshape(CART_DOF)))

    # -- streaming ----------------------------------------------------------
    def stream_cartesian(self, pose: np.ndarray, wrench: Optional[np.ndarray] = None) -> None:
        pose = list(np.asarray(pose, float).reshape(7))
        # RDK v1.x Stream/SendCartesianMotionForce take wrench[6] as the 2nd arg;
        # it only acts on axes enabled via SetForceControlAxis, so a zero default
        # is safe for pure-motion ticks. Dropping it (pose-only) silently disabled
        # force control during traj execution. Precedence: an explicit per-tick
        # wrench wins; otherwise command the active mode's configured
        # target_wrench (so ForceControlParams.target_wrench is actually applied);
        # otherwise zero.
        if wrench is not None:
            wr = list(np.asarray(wrench, float).reshape(CART_DOF))
        elif self._force_control is not None:
            wr = list(np.asarray(self._force_control.target_wrench, float).reshape(CART_DOF))
        else:
            wr = [0.0] * CART_DOF
        if self._mode.is_realtime:
            # RT: 1 kHz streaming, robot tracks immediately.
            self._robot.StreamCartesianMotionForce(pose, wr)
        else:
            # NRT: discrete target, robot's motion generator interpolates.
            self._robot.SendCartesianMotionForce(pose, wr)

    def stream_joint(self, q: np.ndarray) -> None:
        q = [float(v) for v in np.asarray(q, float)]   # plain floats (flexivrdk rejects np.float64)
        zeros = [0.0] * self.n_joints
        if self._mode.is_realtime:
            # RT: StreamJointPosition(positions, velocities, accelerations). (RDK v1.x)
            self._robot.StreamJointPosition(q, zeros, zeros)
        else:
            # NRT: flexivrdk 1.7 SendJointPosition takes FIVE list[float] args:
            # (target_pos, target_vel, target_acc, max_vel, max_acc). The backend
            # previously passed four (dropping target_acc), which TypeError'd and
            # broke the home-restore. HARDWARE-VERIFIED signature on Rizon4s-062626.
            max_vel = [1.0] * self.n_joints
            max_acc = [1.0] * self.n_joints
            self._robot.SendJointPosition(q, zeros, zeros, max_vel, max_acc)

    # -- gripper ------------------------------------------------------------
    def move_gripper(self, cmd: GripperCommand) -> None:
        if self._gripper is None:
            if self._gripper_name:
                # A gripper was CONFIGURED but failed to enable/init at
                # connect. Silently no-op'ing its commands turns every grasp
                # into an invisible failure (the arm pantomimes a pick with
                # frozen fingers) -- fail loudly instead. Gripper-less setups
                # set gripper_name '' and skip cleanly.
                raise RuntimeError(
                    f"gripper {self._gripper_name!r} was configured but failed "
                    f"to enable/init at connect; refusing to no-op a gripper "
                    f"command. Check the device name (Flexiv Elements -> "
                    f"Settings -> Device), power, and connection -- or set "
                    f"gripper_name: '' for a gripper-less config."
                )
            return
        if self._gripper_limits is None:
            raise RuntimeError(
                "runtime gripper limits are unavailable; refusing command"
            )
        checks = [("force", cmd.force, "min_force_n", "max_force_n")]
        if not cmd.grasp:
            checks.extend(
                [
                    (
                        "width",
                        cmd.width,
                        "min_width_m",
                        "max_width_m",
                    ),
                    (
                        "velocity",
                        cmd.velocity,
                        "min_velocity_m_s",
                        "max_velocity_m_s",
                    ),
                ]
            )
        for field, raw, lo_key, hi_key in checks:
            value = float(raw)
            lo = float(self._gripper_limits[lo_key])
            hi = float(self._gripper_limits[hi_key])
            if not np.isfinite(value) or not lo <= value <= hi:
                raise ValueError(
                    f"gripper {field}={value!r} outside runtime "
                    f"[{lo}, {hi}] from flexivrdk.Gripper.params"
                )
        if cmd.grasp:
            self._gripper.Grasp(cmd.force)  # VERIFY
        else:
            self._gripper.Move(cmd.width, cmd.velocity, cmd.force)  # VERIFY

    # -- safety -------------------------------------------------------------
    def stop(self) -> None:
        self._robot.Stop()
        self._mode = ControlMode.IDLE

    def home(self, q_home: Optional[np.ndarray] = None) -> None:
        """Home via the NRT 'Home' primitive (robot-internal, safe).

        The vendor primitive goes to the FACTORY home, not a configured posture,
        so a non-None ``q_home`` raises ``NotImplementedError`` and lets
        ``Robot.home()`` fall back to a speed-capped interpolated joint move to
        the configured posture -- previously the argument was silently ignored
        and the arm ended somewhere other than the lab's canonical home.
        """
        if q_home is not None:
            raise NotImplementedError(
                "the RDK 'Home' primitive ignores a configured q_home; "
                "Robot.home falls back to an interpolated joint move"
            )
        self._robot.SwitchMode(_rdk_mode(ControlMode.NRT_PRIMITIVE))
        # ExecutePrimitive(name, input_params, block_until_started=True). (RDK v1.x)
        self._robot.ExecutePrimitive("Home", dict())
        # Block until the primitive reports completion. RDK v1.x exposes
        # primitive_states() as a dict; "reachedTarget" flips to "1" when done.
        # Fall back to busy() if that key is absent on this RDK version.
        t0 = time.time()
        while time.time() - t0 < 30.0:
            if not self._primitive_running():
                break
            time.sleep(0.05)
        self._mode = ControlMode.NRT_PRIMITIVE

    def zero_ft_sensor(self) -> None:
        """Zero the 6-DoF F/T sensor via the NRT 'ZeroFTSensor' primitive, then
        return to IDLE. Force-control modes (our Cartesian impedance) fault with
        event 301004 until this has run in the CURRENT remote/RDK session -- a
        manual zero in Elements' MANUAL mode does NOT carry over. Run with the arm
        AT REST and NO external contact. HARDWARE-VERIFIED on Rizon4s (mirrors the
        connect-time zero, callable on demand so a session need not restart)."""
        M = flexivrdk.Mode
        if not hasattr(M, "NRT_PRIMITIVE_EXECUTION"):
            return
        self._robot.SwitchMode(M.NRT_PRIMITIVE_EXECUTION)
        self._robot.ExecutePrimitive("ZeroFTSensor", dict())
        t0 = time.time()
        while self._robot.busy() and time.time() - t0 < 10.0:
            time.sleep(0.2)
        self._robot.SwitchMode(M.IDLE)
        self._mode = ControlMode.IDLE

    def _primitive_running(self) -> bool:
        """True while an NRT primitive is still executing (best-effort, cross-version)."""
        try:
            ps = self._robot.primitive_states()
            if isinstance(ps, dict) and "reachedTarget" in ps:
                return str(ps["reachedTarget"]).strip().lower() not in ("1", "true")
        except Exception:
            pass
        try:
            return bool(self._robot.busy())
        except Exception:
            return False
