"""Numpy-only planning of what a traj WILL command -- the intended motion.

This module deliberately imports no visualization library so the preview math
is unit-tested in the core (numpy-only) CI job and can never drift silently
behind a missing optional dependency.

The one rule that makes the preview trustworthy: it must run the SAME code the
executor runs. ``plan_trajectory_preview`` mirrors ``Robot.execute_cartesian_trajectory``
exactly -- ``traj.for_execution(start_pose)`` (relative-traj resolution +
horizon slicing), tightening-only caps ``min(traj, active profile)``, and the
real :class:`~flexiv_control.interpolation.CartesianTrajectoryInterpolator` --
so the rendered path includes time-stretching and is the true per-tick command
stream, not a naive waypoint lerp. A regression test asserts the preview
equals the executed command stream on the fake backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .. import transforms as T
from ..trajectory import CartesianTrajectory
from ..interpolation import CartesianTrajectoryInterpolator
from ..safety import SafetyProfile
from ..types import GripperCommand


@dataclass
class GripperEvent:
    """A gripper actuation inside the planned stream (latched at the first
    tick of its segment and running concurrently with the motion)."""

    tick: int                 # index into TrajectoryPreview.setpoints
    position: np.ndarray      # TCP position where it fires
    command: GripperCommand
    closing: bool             # True if narrower than the previous width


@dataclass
class TrajectoryPreview:
    """Everything a viewer (or a go/no-go gate) needs about an intended traj."""

    setpoints: np.ndarray                 # (N, 7) per-tick poses, w-first quats
    gripper_events: List[GripperEvent] = field(default_factory=list)
    waypoints: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    warnings: List[str] = field(default_factory=list)
    duration_s: float = 0.0               # true wall-clock incl. time-stretch
    nominal_duration_s: float = 0.0       # sum of requested waypoint durations
    linear_speed_cap: float = 0.0         # the cap that will actually bind
    angular_speed_cap: float = 0.0
    start_pose: np.ndarray = field(default_factory=lambda: np.zeros(7))

    @property
    def time_stretched(self) -> bool:
        return self.duration_s > self.nominal_duration_s + 1e-6

    @property
    def terminal_pose(self) -> np.ndarray:
        return self.setpoints[-1] if len(self.setpoints) else self.start_pose


def effective_caps(
    traj: CartesianTrajectory, profile: Optional[SafetyProfile]
) -> Tuple[float, float]:
    """The tightening-only speed caps the executor will enforce:
    ``min(traj cap, active profile cap)`` per axis (matching
    ``Robot.execute_cartesian_trajectory``)."""
    lin = float(traj.max_tcp_linear_speed) if traj.max_tcp_linear_speed else float("inf")
    ang = float(traj.max_tcp_angular_speed) if traj.max_tcp_angular_speed else float("inf")
    if profile is not None:
        lin = min(lin, float(profile.max_linear_speed))
        ang = min(ang, float(profile.max_angular_speed))
    return lin, ang


def plan_trajectory_preview(
    traj: CartesianTrajectory,
    start_pose: np.ndarray,
    profile: Optional[SafetyProfile] = None,
    *,
    control_hz: float = 100.0,
) -> TrajectoryPreview:
    """Dry-run the traj into the exact per-tick command stream the executor
    would send, plus the annotations an operator needs for a go/no-go call."""
    start_pose = np.asarray(start_pose, float).reshape(7)
    resolved = traj.for_execution(start_pose)
    lin_cap, ang_cap = effective_caps(resolved, profile)
    interp = CartesianTrajectoryInterpolator(
        resolved,
        start_pose,
        control_hz,
        max_linear_speed=None if np.isinf(lin_cap) else lin_cap,
        max_angular_speed=None if np.isinf(ang_cap) else ang_cap,
    )

    poses: List[np.ndarray] = []
    events: List[GripperEvent] = []
    prev_width: Optional[float] = None
    for pose, grip in interp:
        if grip is not None:
            # A grasp (close-until-contact) is always a closing event; a width
            # move is closing iff it narrows the previous commanded width.
            width = 0.0 if grip.grasp else float(grip.width)
            closing = bool(grip.grasp) or (
                prev_width is not None and width < prev_width - 1e-6
            )
            events.append(
                GripperEvent(
                    tick=len(poses),
                    position=pose[:3].copy(),
                    command=grip,
                    closing=closing,
                )
            )
            prev_width = width
        poses.append(pose)

    setpoints = np.asarray(poses, float).reshape(-1, 7)
    warnings = profile.validate_trajectory(resolved) if profile is not None else []
    return TrajectoryPreview(
        setpoints=setpoints,
        gripper_events=events,
        waypoints=np.asarray([w.position for w in resolved.waypoints], float).reshape(-1, 3),
        warnings=list(warnings),
        duration_s=len(setpoints) / float(control_hz),
        nominal_duration_s=resolved.total_duration(control_hz),
        linear_speed_cap=lin_cap,
        angular_speed_cap=ang_cap,
        start_pose=start_pose,
    )


def pose_distance(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """(linear metres, angular radians) between two 7-vector poses -- used for
    the preview-staleness check (the preview was planned FROM a pose; if the
    live TCP has moved since, the rendered path no longer starts where the
    robot is and the gate must refuse)."""
    a = np.asarray(a, float).reshape(7)
    b = np.asarray(b, float).reshape(7)
    return (
        float(np.linalg.norm(a[:3] - b[:3])),
        float(T.quat_angle(a[3:7], b[3:7])),
    )


def workspace_box_edges(profile: SafetyProfile) -> np.ndarray:
    """The profile's TCP workspace box as 12 line segments, shape (12, 2, 3),
    ready for a polyline/line-segments primitive."""
    x0, x1 = profile.ws_x
    y0, y1 = profile.ws_y
    z0, z1 = profile.ws_z
    c = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ],
        float,
    )
    pairs = [
        (0, 1), (1, 2), (2, 3), (3, 0),   # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),   # top
        (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
    ]
    return np.asarray([[c[i], c[j]] for i, j in pairs], float)


def time_colors(n: int) -> np.ndarray:
    """An (n, 3) uint8 start->end gradient (cool blue -> hot red) so a path's
    color encodes time: the operator reads direction and pacing at a glance."""
    t = np.linspace(0.0, 1.0, max(n, 2))[:n, None]
    start = np.array([60.0, 140.0, 255.0])
    end = np.array([255.0, 70.0, 50.0])
    return ((1.0 - t) * start + t * end).astype(np.uint8)


def trail_segments(points: np.ndarray) -> np.ndarray:
    """Consecutive points -> (N-1, 2, 3) segments for a line-segments handle."""
    pts = np.asarray(points, float).reshape(-1, 3)
    if len(pts) < 2:
        return np.zeros((0, 2, 3))
    return np.stack([pts[:-1], pts[1:]], axis=1)
