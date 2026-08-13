"""Turn high-level actions into a stream of fixed-rate setpoints.

The control loop runs at a fixed rate (e.g. 100 Hz NRT or 1 kHz RT). The job of
this module is to expand a :class:`CartesianTrajectory` / :class:`JointTrajectory` /
:class:`CartesianDelta` into one TCP pose (or joint target) per tick, with
smooth interpolation, so the loop just reads "the next setpoint" each cycle.

Position uses linear interpolation; orientation uses SLERP. A "hold orientation"
waypoint (``quaternion=None``) is handled by carrying the previous orientation
forward, so position-only plans need no special casing.
"""

from __future__ import annotations

import math
from typing import Iterator, List, Optional, Tuple

import numpy as np

from . import transforms as T
from .trajectory import (
    CartesianTrajectory,
    CartesianDelta,
    JointTrajectory,
)
from .types import GripperCommand

# Peak instantaneous speed of the cosine ease is (pi/2)x its average speed
# (the forward difference maxes at sin(pi/2n) <= pi/2n at the segment midpoint).
# Scaling the velocity-cap tick count by this factor guarantees the peak tick
# stays under the cap, so the safety filter never has to clip in-spec motion.
_BLEND_PEAK = math.pi / 2.0


def _cosine_blend(s: float) -> float:
    """Smooth 0->1 easing so velocity is zero at segment ends (less jerk)."""
    return 0.5 - 0.5 * np.cos(np.pi * float(np.clip(s, 0.0, 1.0)))


class CartesianTrajectoryInterpolator:
    """Iterate a traj into ``(tcp_pose, gripper_or_None)`` per control tick.

    If ``max_linear_speed`` / ``max_angular_speed`` are given, a segment that
    would exceed them is *time-stretched* (more ticks) so it still reaches the
    waypoint, just no faster than the cap. This keeps the safety filter from
    having to spatially clip in-spec motion (which would stall the path), while
    honouring the traj's requested ``n_frames`` whenever it is already slow
    enough.
    """

    def __init__(
        self,
        traj: CartesianTrajectory,
        start_pose: np.ndarray,
        control_hz: float,
        *,
        max_linear_speed: Optional[float] = None,
        max_angular_speed: Optional[float] = None,
    ):
        self.traj = traj
        self.hz = float(control_hz)
        self.dt = 1.0 / self.hz
        self.start_pose = np.asarray(start_pose, float).reshape(7).copy()
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        # Index of the waypoint currently being interpolated (updated during
        # iteration), so an aborting executor can report WHERE the run stopped.
        self.current_segment = 0
        # Tick count of that segment (updated per segment): the executor's
        # deferred gripper-close gate sizes its settle window within it.
        self.current_segment_ticks = 1

    def _segment_ticks(self, prev_pos, tgt_pos, prev_quat, tgt_quat, wp) -> int:
        n = max(1, int(round(wp.resolve_duration(self.hz) * self.hz)))
        if self.max_linear_speed and self.max_linear_speed > 0:
            dist = float(np.linalg.norm(tgt_pos - prev_pos))
            n = max(n, int(np.ceil(_BLEND_PEAK * dist / (self.max_linear_speed * self.dt))))
        if self.max_angular_speed and self.max_angular_speed > 0:
            ang = float(T.quat_angle(prev_quat, tgt_quat))
            n = max(n, int(np.ceil(_BLEND_PEAK * ang / (self.max_angular_speed * self.dt))))
        return max(1, n)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Optional[GripperCommand]]]:
        prev_pos = self.start_pose[:3].copy()
        prev_quat = self.start_pose[3:7].copy()
        for seg_idx, wp in enumerate(self.traj.waypoints):
            self.current_segment = seg_idx
            tgt_pos = wp.position
            tgt_quat = prev_quat if wp.quaternion is None else wp.quaternion
            n = self._segment_ticks(prev_pos, tgt_pos, prev_quat, tgt_quat, wp)
            self.current_segment_ticks = n
            for k in range(1, n + 1):
                s = _cosine_blend(k / n)
                pos = prev_pos + s * (tgt_pos - prev_pos)
                quat = T.quat_slerp(prev_quat, tgt_quat, s)
                pose = np.concatenate([pos, quat])
                # Emit the gripper command on the first tick of the segment;
                # the control loop latches it.
                grip = wp.gripper if k == 1 else None
                yield pose, grip
            prev_pos = tgt_pos.copy()
            prev_quat = tgt_quat.copy()

    def setpoints(self) -> List[Tuple[np.ndarray, Optional[GripperCommand]]]:
        return list(iter(self))


class JointTrajectoryInterpolator:
    def __init__(
        self,
        traj: JointTrajectory,
        start_q: np.ndarray,
        control_hz: float,
        *,
        max_joint_speed: Optional[np.ndarray | float] = None,
    ):
        self.traj = traj
        self.hz = float(control_hz)
        self.dt = 1.0 / self.hz
        self.start_q = np.asarray(start_q, float).reshape(-1).copy()
        self.max_joint_speed: Optional[np.ndarray]
        if max_joint_speed is None:
            self.max_joint_speed = None
        else:
            speed = np.asarray(max_joint_speed, dtype=float)
            if speed.ndim == 0:
                speed = np.full(self.start_q.shape, float(speed))
            else:
                speed = speed.reshape(-1)
            if speed.shape != self.start_q.shape:
                raise ValueError(
                    "max_joint_speed must be scalar or match start_q shape "
                    f"{self.start_q.shape}, got {speed.shape}"
                )
            if not np.all(np.isfinite(speed)) or np.any(speed <= 0.0):
                raise ValueError("max_joint_speed values must be finite and > 0")
            self.max_joint_speed = speed

        cumulative_requested_ticks = 0.0
        cumulative_scheduled_ticks = 0
        self.requested_duration_s = 0.0
        self.requested_segment_ticks: list[int] = []
        for wp in self.traj.waypoints:
            duration_s = float(wp.resolve_duration(self.hz))
            if not np.isfinite(duration_s) or duration_s <= 0.0:
                raise ValueError("JointWaypoint duration must be finite and > 0")
            self.requested_duration_s += duration_s
            cumulative_requested_ticks += duration_s * self.hz
            boundary = max(
                cumulative_scheduled_ticks + 1,
                int(np.floor(cumulative_requested_ticks + 0.5)),
            )
            self.requested_segment_ticks.append(boundary - cumulative_scheduled_ticks)
            cumulative_scheduled_ticks = boundary
        self.requested_total_ticks = cumulative_scheduled_ticks
        # Backward-compatible names used by callers/tests.
        self.nominal_segment_ticks = list(self.requested_segment_ticks)
        self.nominal_total_ticks = self.requested_total_ticks

        peak_factor = 1.0 if traj.interpolation == "linear" else _BLEND_PEAK
        self.scheduled_segment_ticks: list[int] = []
        prev = self.start_q
        for segment_index, (wp, requested_n) in enumerate(
            zip(self.traj.waypoints, self.requested_segment_ticks)
        ):
            tgt = wp.positions
            if tgt.shape != self.start_q.shape:
                raise ValueError(
                    f"JointWaypoint {segment_index} positions must match "
                    f"start_q shape {self.start_q.shape}, got {tgt.shape}"
                )
            n = int(requested_n)
            if self.max_joint_speed is not None:
                dq = np.abs(tgt - prev)
                required = peak_factor * dq / (n * self.dt)
                too_fast = required > self.max_joint_speed + 1e-12
                if traj.strict_timing and np.any(too_fast):
                    joint = int(np.flatnonzero(too_fast)[0])
                    raise ValueError(
                        "strict_timing joint rate exceeds effective runtime "
                        f"limit at segment {segment_index}, joint {joint}: "
                        f"required {required[joint]:.9g} rad/s > "
                        f"{self.max_joint_speed[joint]:.9g} rad/s; "
                        f"requested n_frames={n} is authoritative"
                    )
                if not traj.strict_timing:
                    n = max(
                        n,
                        int(np.max(np.ceil(peak_factor * dq / (self.max_joint_speed * self.dt)))),
                    )
            self.scheduled_segment_ticks.append(max(1, n))
            prev = tgt
        self.scheduled_total_ticks = int(sum(self.scheduled_segment_ticks))
        self.current_segment = 0
        self.current_segment_tick = 0

    def __iter__(self) -> Iterator[np.ndarray]:
        prev = self.start_q.copy()
        for segment_index, (wp, n) in enumerate(
            zip(self.traj.waypoints, self.scheduled_segment_ticks)
        ):
            self.current_segment = segment_index
            tgt = wp.positions
            for k in range(1, n + 1):
                self.current_segment_tick = k
                phase = k / n
                s = phase if self.traj.interpolation == "linear" else _cosine_blend(phase)
                yield prev + s * (tgt - prev)
            prev = tgt.copy()

    def setpoints(self) -> List[np.ndarray]:
        return list(iter(self))


class JointTorqueTrajectoryInterpolator:
    """Linear interpolation of direct torque endpoints at controller rate."""

    def __init__(self, traj, control_hz: float):
        del control_hz  # n_frames is authoritative by contract.
        self.traj = traj
        self.current_segment = 0
        self.current_segment_tick = 0
        self.scheduled_total_ticks = int(
            sum(waypoint.n_frames for waypoint in traj.waypoints)
        )

    def __iter__(self):
        previous = self.traj.initial_torques.copy()
        for segment, waypoint in enumerate(self.traj.waypoints):
            self.current_segment = segment
            for tick in range(1, waypoint.n_frames + 1):
                self.current_segment_tick = tick
                yield previous + (tick / waypoint.n_frames) * (
                    waypoint.torques - previous
                )
            previous = waypoint.torques.copy()


def delta_to_target_pose(delta: CartesianDelta, current_pose: np.ndarray) -> np.ndarray:
    """Integrate a relative delta onto the current pose -> absolute target.

    Honours ``delta.frame`` ("base" or "tcp") so end-effector-relative servoing
    (the natural SpaceMouse / "move forward along the gripper" mode) is correct.
    """
    return T.integrate_pose(current_pose, delta.delta, frame=delta.frame)
