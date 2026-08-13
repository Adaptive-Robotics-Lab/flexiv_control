"""Wire protocol for the control server.

A deliberately boring, dependency-free protocol: newline-delimited JSON over a
TCP socket. One request per line, one response per line.

    request  = {"id": int, "method": str, "params": {...}}
    response = {"id": int, "ok": true,  "result": {...}}
             | {"id": int, "ok": false, "error": "message"}

We keep ZMQ/gRPC out of the *core* on purpose: a plain socket means the client
``pip install``s with nothing but numpy, and an RL/MPC author on another machine
can talk to the arm without standing up a ROS workspace or a message broker. The
server design (single owner of the backend + a 1 kHz-capable loop + a lease) is
the part that matters and is transport-independent; swapping in ZMQ later is a
localized change.

This module also defines the (de)serialization for the few structured objects
that cross the wire -- :class:`RobotState`, :class:`CartesianTrajectory`,
:class:`ExecutionResult`, :class:`GripperCommand` -- so neither the server nor
the client hand-rolls it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .. import __version__
from ..trajectory import (
    CartesianTrajectory,
    CartesianWaypoint,
    TrajectoryRepresentation,
    ExecutionResult,
    JointGripperForceTarget,
    JointGripperTarget,
    JointTrajectory,
    JointWaypoint,
    JointTorqueTrajectory,
    JointTorqueWaypoint,
)
from ..types import (
    ControlMode,
    ForceControlParams,
    GripperCommand,
    ImpedanceParams,
    RobotState,
    SafetyStatus,
    StopReason,
)

DEFAULT_PORT = 8766
SERVER_INFO_SCHEMA = "flexiv-control.server-info.v5"
PROTOCOL_ID = "flexiv-control.trajectory-rpc.v5"
JOINT_TRAJECTORY_SCHEMA = "flexiv-control.joint-trajectory.v4"
JOINT_TORQUE_TRAJECTORY_SCHEMA = "flexiv-control.joint-torque-trajectory.v1"

# Canonical, path-independent description of the wire seam that must agree
# across the planner client and robot-side server.  In particular, this pins
# the trajectory RPC names and payload key that differ from the incompatible
# pre-0.2.1 ``*_chunk`` protocol.
PROTOCOL_CONTRACT = {
    "protocol_id": PROTOCOL_ID,
    "transport": "newline-delimited-json-request-response-v1",
    "identity_rpc": {
        "method": "get_server_info",
        "lease_required": False,
        "runtime_fields": {
            "required": ["control_hz", "active_safety_profile"],
            "hardware_when_available": [
                "runtime_hardware_identity",
                "gripper_limits",
                "joint_limits",
                "current_safety_limits",
                "effective_joint_limits",
            ],
        },
    },
    "trajectory_rpcs": {
        "execute_cartesian_trajectory": "traj",
        "execute_joint_trajectory": "traj",
        "execute_joint_torque_trajectory": "traj",
    },
    "joint_trajectory_contract": {
        "schema": JOINT_TRAJECTORY_SCHEMA,
        "trajectory_fields": [
            "schema",
            "waypoints",
            "initial_positions",
            "initial_gripper_width",
            "max_joint_speed_scale",
            "interpolation",
            "strict_timing",
            "safety_profile",
        ],
        "waypoint_fields": ["positions", "n_frames", "duration", "gripper"],
        "gripper_target_variants": {
            "move": ["mode", "width", "force_limit", "velocity"],
            "force": ["mode", "force"],
        },
        "rpc_identity_fields": ["protocol_id", "protocol_fingerprint_sha256"],
        "explicit_initial_target": ["initial_positions", "initial_gripper_width"],
        "strict_timing": "authoritative-n_frames-reject-no-clip-or-time-stretch",
        "continuity": "measured-rebase-every-rpc-prior-ack-provenance-only",
        "first_emitted_joint_bound": "effective-runtime-rate-times-control-period",
        "gripper_execution_anchor": "current-measured-width-every-rpc",
        "predispatch_revalidation": [
            "first-joint-setpoint-after-mode-transition",
            "first-gripper-event-from-current-measured-width",
        ],
        "numeric_json_types": "numbers-and-arrays-only-no-strings-or-booleans",
        "gripper": "explicit-Move-or-signed-Grasp-concurrent-at-segment-boundary",
        "interpolation": ["cosine", "linear"],
        "max_joint_speed_scale": "finite-(0,1]-active-profile-ceiling",
    },
    "joint_torque_trajectory_contract": {
        "schema": JOINT_TORQUE_TRAJECTORY_SCHEMA,
        "action": "gravity-compensated-joint-torque-nm",
        "rate_hz": 1000,
        "interpolation": "linear",
        "limits": "RobotInfo.tau_max-times-active-profile-scale",
        "firmware_soft_limits": True,
        "gripper": "synchronized-signed-force-at-segment-boundary",
    },
}


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_fingerprint_sha256() -> str:
    """Hash installed package source bytes without depending on install path."""
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


PROTOCOL_FINGERPRINT_SHA256 = _canonical_sha256(PROTOCOL_CONTRACT)
SOURCE_FINGERPRINT_SHA256 = _source_fingerprint_sha256()


def server_info(**runtime: Any) -> dict[str, Any]:
    """Return process identity plus cached, lease-free runtime facts."""
    required = {"control_hz", "active_safety_profile"}
    missing = sorted(required.difference(runtime))
    if missing:
        raise ValueError("server_info missing required runtime fields: " + ", ".join(missing))
    control_hz = float(runtime["control_hz"])
    if not np.isfinite(control_hz) or control_hz <= 0.0:
        raise ValueError("server_info control_hz must be finite and > 0")
    active_profile = str(runtime["active_safety_profile"]).strip()
    if not active_profile:
        raise ValueError("server_info active_safety_profile must be non-empty")
    info: dict[str, Any] = {
        "schema": SERVER_INFO_SCHEMA,
        "package": "flexiv-control",
        "package_version": __version__,
        "protocol_id": PROTOCOL_ID,
        "protocol_fingerprint_sha256": PROTOCOL_FINGERPRINT_SHA256,
        "source_fingerprint_sha256": SOURCE_FINGERPRINT_SHA256,
        "control_hz": control_hz,
        "active_safety_profile": active_profile,
    }
    for key in (
        "runtime_hardware_identity",
        "gripper_limits",
        "joint_limits",
        "current_safety_limits",
        "effective_joint_limits",
    ):
        if key in runtime:
            info[key] = runtime[key]
    return info


# ---------------------------------------------------------------------------
# JSON helpers (numpy-aware)
# ---------------------------------------------------------------------------
def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(f"not JSON serializable: {type(x)!r}")


def dumps(obj: dict) -> bytes:
    return (json.dumps(obj, default=_jsonable) + "\n").encode("utf-8")


def loads(line: bytes) -> dict:
    return json.loads(line.decode("utf-8"))


# ---------------------------------------------------------------------------
# GripperCommand
# ---------------------------------------------------------------------------
def gripper_to_dict(g: Optional[GripperCommand]) -> Optional[dict]:
    if g is None:
        return None
    return {"width": g.width, "force": g.force, "velocity": g.velocity, "grasp": g.grasp}


def gripper_from_dict(d: Optional[dict]) -> Optional[GripperCommand]:
    if d is None:
        return None
    return GripperCommand(
        width=float(d.get("width", 0.0)),
        force=float(d.get("force", 20.0)),
        velocity=float(d.get("velocity", 0.1)),
        grasp=bool(d.get("grasp", False)),
    )


# ---------------------------------------------------------------------------
# RobotState
# ---------------------------------------------------------------------------
def state_to_dict(s: RobotState) -> dict:
    return {
        "stamp": s.stamp,
        "q": s.q.tolist(),
        "dq": s.dq.tolist(),
        "tau": s.tau.tolist(),
        "tcp_pose": s.tcp_pose.tolist(),
        "tcp_vel": s.tcp_vel.tolist(),
        "wrench": s.wrench.tolist(),
        "gripper_width": s.gripper_width,
        "gripper_force": s.gripper_force,
        "gripper_is_moving": s.gripper_is_moving,
        "control_mode": s.control_mode.value,
        "safety_status": s.safety_status.value,
        "stop_reason": s.stop_reason.value,
        "active_owner": s.active_owner,
        "command_latency_ms": s.command_latency_ms,
        "loop_period_ms": s.loop_period_ms,
        "loop_jitter_us": s.loop_jitter_us,
        "missed_cycles": s.missed_cycles,
    }


def state_from_dict(d: dict) -> RobotState:
    return RobotState(
        stamp=float(d["stamp"]),
        q=np.asarray(d["q"], float),
        dq=np.asarray(d["dq"], float),
        tau=np.asarray(d["tau"], float),
        tcp_pose=np.asarray(d["tcp_pose"], float),
        tcp_vel=np.asarray(d["tcp_vel"], float),
        wrench=np.asarray(d["wrench"], float),
        gripper_width=float(d["gripper_width"]),
        gripper_force=float(d["gripper_force"]),
        gripper_is_moving=bool(d["gripper_is_moving"]),
        control_mode=ControlMode(d["control_mode"]),
        safety_status=SafetyStatus(d["safety_status"]),
        stop_reason=StopReason(d["stop_reason"]),
        active_owner=d.get("active_owner", ""),
        command_latency_ms=float(d.get("command_latency_ms", 0.0)),
        loop_period_ms=float(d.get("loop_period_ms", 0.0)),
        loop_jitter_us=float(d.get("loop_jitter_us", 0.0)),
        missed_cycles=int(d.get("missed_cycles", 0)),
    )


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------
def result_to_dict(r: ExecutionResult) -> dict:
    return {
        "success": r.success,
        "clipped": r.clipped,
        "stop_reason": r.stop_reason,
        "executed_duration": r.executed_duration,
        "path_tracking_error": r.path_tracking_error,
        "max_tcp_speed": r.max_tcp_speed,
        "max_joint_speed": r.max_joint_speed,
        "max_wrench": r.max_wrench,
        "gripper_width_final": r.gripper_width_final,
        "final_state": state_to_dict(r.final_state) if r.final_state is not None else None,
        "log": r.log,
    }


def result_from_dict(d: dict) -> ExecutionResult:
    fs = d.get("final_state")
    return ExecutionResult(
        success=bool(d["success"]),
        clipped=bool(d.get("clipped", False)),
        stop_reason=d.get("stop_reason", "none"),
        executed_duration=float(d.get("executed_duration", 0.0)),
        path_tracking_error=float(d.get("path_tracking_error", 0.0)),
        max_tcp_speed=float(d.get("max_tcp_speed", 0.0)),
        max_joint_speed=float(d.get("max_joint_speed", 0.0)),
        max_wrench=float(d.get("max_wrench", 0.0)),
        gripper_width_final=float(d.get("gripper_width_final", 0.0)),
        final_state=state_from_dict(fs) if fs is not None else None,
        log=d.get("log", {}),
    )


# ---------------------------------------------------------------------------
# CartesianTrajectory
# ---------------------------------------------------------------------------
def trajectory_to_dict(c: CartesianTrajectory) -> dict:
    return {
        "waypoints": [
            {
                "position": w.position.tolist(),
                "quaternion": None if w.quaternion is None else w.quaternion.tolist(),
                "gripper": gripper_to_dict(w.gripper),
                "n_frames": w.n_frames,
                "duration": w.duration,
                "frame": w.frame,
            }
            for w in c.waypoints
        ],
        "impedance": {
            "stiffness": c.impedance.stiffness.tolist(),
            "damping_ratio": c.impedance.damping_ratio.tolist(),
        },
        "force_control": (
            None
            if c.force_control is None
            else {
                "enabled_axes": c.force_control.enabled_axes.tolist(),
                "target_wrench": c.force_control.target_wrench.tolist(),
                "frame": c.force_control.frame,
            }
        ),
        "max_tcp_linear_speed": c.max_tcp_linear_speed,
        "max_tcp_angular_speed": c.max_tcp_angular_speed,
        "max_tcp_linear_acc": c.max_tcp_linear_acc,
        "max_tcp_angular_acc": c.max_tcp_angular_acc,
        "max_contact_wrench": None
        if c.max_contact_wrench is None
        else c.max_contact_wrench.tolist(),
        "contact_wrench_allowance": None
        if c.contact_wrench_allowance is None
        else c.contact_wrench_allowance.tolist(),
        "grip_tracking_gate_m": c.grip_tracking_gate_m,
        "safety_profile": c.safety_profile,
        "frame": c.frame,
        "representation": c.representation.value,
        "n_execute": c.n_execute,
    }


def trajectory_from_dict(d: dict) -> CartesianTrajectory:
    wpts = [
        CartesianWaypoint(
            position=np.asarray(w["position"], float),
            quaternion=None if w.get("quaternion") is None else np.asarray(w["quaternion"], float),
            gripper=gripper_from_dict(w.get("gripper")),
            n_frames=w.get("n_frames"),
            duration=w.get("duration"),
            frame=w.get("frame", "base"),
        )
        for w in d["waypoints"]
    ]
    imp = d.get("impedance")
    impedance = (
        ImpedanceParams(
            stiffness=np.asarray(imp["stiffness"], float),
            damping_ratio=np.asarray(imp["damping_ratio"], float),
        )
        if imp
        else ImpedanceParams()
    )
    fc = d.get("force_control")
    force_control = (
        ForceControlParams(
            enabled_axes=np.asarray(fc["enabled_axes"], bool),
            target_wrench=np.asarray(fc["target_wrench"], float),
            frame=fc.get("frame", "tcp"),
        )
        if fc
        else None
    )
    return CartesianTrajectory(
        waypoints=wpts,
        impedance=impedance,
        force_control=force_control,
        max_tcp_linear_speed=float(d.get("max_tcp_linear_speed", 0.25)),
        max_tcp_angular_speed=float(d.get("max_tcp_angular_speed", 0.60)),
        max_tcp_linear_acc=float(d.get("max_tcp_linear_acc", 1.0)),
        max_tcp_angular_acc=float(d.get("max_tcp_angular_acc", 2.0)),
        max_contact_wrench=None
        if d.get("max_contact_wrench") is None
        else np.asarray(d["max_contact_wrench"], float),
        contact_wrench_allowance=None
        if d.get("contact_wrench_allowance") is None
        else np.asarray(d["contact_wrench_allowance"], float),
        grip_tracking_gate_m=d.get("grip_tracking_gate_m"),
        safety_profile=d.get("safety_profile", ""),
        frame=d.get("frame", "base"),
        representation=TrajectoryRepresentation(d.get("representation", "absolute")),
        n_execute=d.get("n_execute"),
    )


# ---------------------------------------------------------------------------
# JointTrajectory
# ---------------------------------------------------------------------------
def _require_exact_keys(value: dict, expected: set[str], *, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{context} keys do not match {JOINT_TRAJECTORY_SCHEMA}: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _require_json_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a JSON number (not a string or boolean)")
    return float(value)


def _require_optional_json_number(value: Any, *, context: str) -> Optional[float]:
    if value is None:
        return None
    return _require_json_number(value, context=context)


def _require_json_number_array(value: Any, *, context: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON array")
    return [
        _require_json_number(item, context=f"{context}[{index}]")
        for index, item in enumerate(value)
    ]


def joint_gripper_target_to_dict(
    g: Optional[JointGripperTarget | JointGripperForceTarget],
) -> Optional[dict]:
    if g is None:
        return None
    if isinstance(g, JointGripperForceTarget):
        return {"mode": "force", "force": g.force}
    return {
        "mode": "move",
        "width": g.width,
        "force_limit": g.force,
        "velocity": g.velocity,
    }


def joint_gripper_target_from_dict(
    d: Optional[dict],
) -> Optional[JointGripperTarget | JointGripperForceTarget]:
    if d is None:
        return None
    if not isinstance(d, dict):
        raise ValueError("JointWaypoint.gripper must be an object or null")
    mode = d.get("mode")
    if mode == "force":
        _require_exact_keys(d, {"mode", "force"}, context="JointGripperForceTarget")
        return JointGripperForceTarget(
            force=_require_json_number(
                d["force"], context="JointGripperForceTarget.force"
            )
        )
    if mode != "move":
        raise ValueError("JointWaypoint.gripper.mode must be 'move' or 'force'")
    _require_exact_keys(
        d,
        {"mode", "width", "force_limit", "velocity"},
        context="JointGripperTarget",
    )
    return JointGripperTarget(
        width=_require_json_number(d["width"], context="JointGripperTarget.width"),
        force=_require_json_number(
            d["force_limit"], context="JointGripperTarget.force_limit"
        ),
        velocity=_require_optional_json_number(
            d["velocity"], context="JointGripperTarget.velocity"
        ),
    )


def joint_trajectory_to_dict(c: JointTrajectory) -> dict:
    return {
        "schema": JOINT_TRAJECTORY_SCHEMA,
        "waypoints": [
            {
                "positions": w.positions.tolist(),
                "n_frames": w.n_frames,
                "duration": w.duration,
                "gripper": joint_gripper_target_to_dict(w.gripper),
            }
            for w in c.waypoints
        ],
        "initial_positions": (
            None if c.initial_positions is None else c.initial_positions.tolist()
        ),
        "initial_gripper_width": c.initial_gripper_width,
        "max_joint_speed_scale": c.max_joint_speed_scale,
        "interpolation": c.interpolation,
        "strict_timing": c.strict_timing,
        "safety_profile": c.safety_profile,
    }


def joint_trajectory_from_dict(d: dict) -> JointTrajectory:
    _require_exact_keys(
        d,
        {
            "schema",
            "waypoints",
            "initial_positions",
            "initial_gripper_width",
            "max_joint_speed_scale",
            "interpolation",
            "strict_timing",
            "safety_profile",
        },
        context="JointTrajectory",
    )
    if d["schema"] != JOINT_TRAJECTORY_SCHEMA:
        raise ValueError(
            f"unsupported JointTrajectory schema {d['schema']!r}; "
            f"expected {JOINT_TRAJECTORY_SCHEMA!r}"
        )
    if not isinstance(d["waypoints"], list):
        raise ValueError("JointTrajectory.waypoints must be a list")
    if not isinstance(d["strict_timing"], bool):
        raise ValueError("JointTrajectory.strict_timing must be a boolean")
    if not isinstance(d["interpolation"], str):
        raise ValueError("JointTrajectory.interpolation must be a string")
    if not isinstance(d["safety_profile"], str):
        raise ValueError("JointTrajectory.safety_profile must be a string")
    initial_positions = (
        None
        if d["initial_positions"] is None
        else _require_json_number_array(
            d["initial_positions"], context="JointTrajectory.initial_positions"
        )
    )
    initial_gripper_width = _require_optional_json_number(
        d["initial_gripper_width"], context="JointTrajectory.initial_gripper_width"
    )
    max_joint_speed_scale = _require_json_number(
        d["max_joint_speed_scale"], context="JointTrajectory.max_joint_speed_scale"
    )
    wpts = []
    for index, w in enumerate(d["waypoints"]):
        _require_exact_keys(
            w,
            {"positions", "n_frames", "duration", "gripper"},
            context=f"JointWaypoint[{index}]",
        )
        positions = _require_json_number_array(
            w["positions"], context=f"JointWaypoint[{index}].positions"
        )
        n_frames = w["n_frames"]
        if n_frames is not None and (isinstance(n_frames, bool) or not isinstance(n_frames, int)):
            raise ValueError(f"JointWaypoint[{index}].n_frames must be a JSON integer or null")
        duration = _require_optional_json_number(
            w["duration"], context=f"JointWaypoint[{index}].duration"
        )
        wpts.append(
            JointWaypoint(
                positions=np.asarray(positions, float),
                n_frames=n_frames,
                duration=duration,
                gripper=joint_gripper_target_from_dict(w["gripper"]),
            )
        )
    return JointTrajectory(
        waypoints=wpts,
        initial_positions=(
            None if initial_positions is None else np.asarray(initial_positions, float)
        ),
        initial_gripper_width=initial_gripper_width,
        max_joint_speed_scale=max_joint_speed_scale,
        interpolation=d["interpolation"],
        strict_timing=d["strict_timing"],
        safety_profile=d["safety_profile"],
    )


def joint_torque_trajectory_to_dict(c: JointTorqueTrajectory) -> dict:
    return {
        "schema": JOINT_TORQUE_TRAJECTORY_SCHEMA,
        "initial_torques": c.initial_torques.tolist(),
        "waypoints": [
            {
                "torques": waypoint.torques.tolist(),
                "n_frames": waypoint.n_frames,
                "gripper": joint_gripper_target_to_dict(waypoint.gripper),
            }
            for waypoint in c.waypoints
        ],
        "max_joint_torque_scale": c.max_joint_torque_scale,
        "safety_profile": c.safety_profile,
    }


def joint_torque_trajectory_from_dict(d: dict) -> JointTorqueTrajectory:
    _require_exact_keys(
        d,
        {
            "schema",
            "initial_torques",
            "waypoints",
            "max_joint_torque_scale",
            "safety_profile",
        },
        context="JointTorqueTrajectory",
    )
    if d["schema"] != JOINT_TORQUE_TRAJECTORY_SCHEMA:
        raise ValueError("unsupported JointTorqueTrajectory schema")
    if not isinstance(d["waypoints"], list) or not d["waypoints"]:
        raise ValueError("JointTorqueTrajectory.waypoints must be non-empty")
    waypoints = []
    for index, waypoint in enumerate(d["waypoints"]):
        _require_exact_keys(
            waypoint,
            {"torques", "n_frames", "gripper"},
            context=f"JointTorqueTrajectory.waypoints[{index}]",
        )
        frames = waypoint["n_frames"]
        if isinstance(frames, bool) or not isinstance(frames, int):
            raise ValueError("JointTorqueWaypoint.n_frames must be an integer")
        waypoints.append(
            JointTorqueWaypoint(
                torques=np.asarray(
                    _require_json_number_array(
                        waypoint["torques"],
                        context=f"waypoints[{index}].torques",
                    ),
                    dtype=float,
                ),
                n_frames=frames,
                gripper=joint_gripper_target_from_dict(waypoint["gripper"]),
            )
        )
    return JointTorqueTrajectory(
        waypoints=waypoints,
        initial_torques=np.asarray(
            _require_json_number_array(
                d["initial_torques"], context="initial_torques"
            ),
            dtype=float,
        ),
        max_joint_torque_scale=_require_json_number(
            d["max_joint_torque_scale"], context="max_joint_torque_scale"
        ),
        safety_profile=str(d["safety_profile"]),
    )
