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

import json
from typing import Any, Optional

import numpy as np

from ..trajectory import (
    CartesianTrajectory,
    CartesianWaypoint,
    TrajectoryRepresentation,
    ExecutionResult,
    JointTrajectory,
    JointWaypoint,
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
def joint_trajectory_to_dict(c: JointTrajectory) -> dict:
    return {
        "waypoints": [
            {
                "positions": w.positions.tolist(),
                "n_frames": w.n_frames,
                "duration": w.duration,
            }
            for w in c.waypoints
        ],
        "max_joint_speed_scale": c.max_joint_speed_scale,
        "safety_profile": c.safety_profile,
    }


def joint_trajectory_from_dict(d: dict) -> JointTrajectory:
    wpts = [
        JointWaypoint(
            positions=np.asarray(w["positions"], float),
            n_frames=w.get("n_frames"),
            duration=w.get("duration"),
        )
        for w in d["waypoints"]
    ]
    return JointTrajectory(
        waypoints=wpts,
        max_joint_speed_scale=float(d.get("max_joint_speed_scale", 0.3)),
        safety_profile=d.get("safety_profile", ""),
    )
