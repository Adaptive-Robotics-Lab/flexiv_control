"""flexiv_control.viz -- live browser visualization + intended-motion preview.

Install the optional extra::

    pip install "flexiv-control[viz]"

then either monitor a running server from any machine on the LAN::

    flexiv-control viz --connect <robot-pc>

or embed in a planner process::

    from flexiv_control.viz import RobotViz
    viz = RobotViz();  viz.attach(robot, allow_lease=True)
    viz.preview_trajectory(traj)            # the intended motion, before executing
    result = robot.execute_cartesian_trajectory(traj)
    viz.on_step(i, traj, result)

The preview math (``flexiv_control.viz.preview``) is numpy-only and importable
without viser; only :class:`RobotViz` needs the extra.
"""

from __future__ import annotations

# numpy-only pieces: always importable (unit-tested in the core CI job).
from .preview import (  # noqa: F401
    TrajectoryPreview,
    GripperEvent,
    effective_caps,
    plan_trajectory_preview,
    pose_distance,
    time_colors,
    workspace_box_edges,
)

__all__ = [
    "RobotViz",
    "TrajectoryPreview",
    "GripperEvent",
    "effective_caps",
    "plan_trajectory_preview",
    "pose_distance",
    "time_colors",
    "workspace_box_edges",
]


def __getattr__(name: str):
    if name == "RobotViz":
        try:
            from .app import RobotViz
        except ImportError as e:
            raise ImportError(
                "RobotViz needs the optional viz dependencies "
                '(pip install "flexiv-control[viz]"): ' + str(e)
            ) from e
        return RobotViz
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
