"""Live browser visualization: robot mirror + intended-motion preview.

Runs entirely offline on the FakeBackend -- no hardware, no robot model
assets (frames mode). Open the printed URL in a browser and you will see the
TCP frame + gripper jaws move, the measured trail accumulate, the safety
profile's workspace box, and -- before each traj executes -- the INTENDED
motion: the true per-tick command path (time-colored start->end), waypoint
knots, gripper open/close glyphs, the terminal pose, and an animated ghost.

    pip install "flexiv-control[viz]"
    python examples/08_live_viz.py

Against a real server, the same view is one command (read-only, no lease):

    flexiv-control viz --connect <robot-pc>

In a planner loop, the viz doubles as a go/no-go gate -- see
``RobotViz.gate(require_click=True)`` and docs/visualization.md.
"""

from __future__ import annotations

import time


from flexiv_control import (
    CartesianTrajectory,
    CartesianWaypoint,
    GripperCommand,
    Robot,
    RobotConfig,
)
from flexiv_control.viz import RobotViz


def main() -> None:
    robot = Robot(RobotConfig(backend="fake"))
    robot.connect()
    robot.acquire_lease("viz-demo")

    viz = RobotViz(port=8080, state_hz=20.0)
    viz.attach(robot, allow_lease=True)  # we ARE the controlling process here
    print(f"open {viz.url} in a browser; running a scripted pick-ish loop...")

    try:
        for cycle in range(100):
            s = robot.get_state()
            p = s.tcp_pose[:3]
            traj = CartesianTrajectory(
                waypoints=[
                    CartesianWaypoint(position=p + [0.06, 0.04, -0.05], quaternion=None,
                                      gripper=GripperCommand(width=0.02, grasp=True),
                                      duration=1.5),
                    CartesianWaypoint(position=p + [0.06, -0.06, 0.04], quaternion=None,
                                      duration=1.5),
                    CartesianWaypoint(position=p, quaternion=None,
                                      gripper=GripperCommand(width=0.08),
                                      duration=1.5),
                ],
                max_tcp_linear_speed=0.12,
            )
            # 1) show the INTENDED motion (and let the ghost animate a moment)
            pv = viz.preview_trajectory(traj, s, robot.profile, traj_id=str(cycle))
            print(f"traj {cycle}: {len(pv.setpoints)} setpoints, "
                  f"{pv.duration_s:.1f}s planned"
                  + (" (time-stretched)" if pv.time_stretched else ""))
            time.sleep(2.0)
            # 2) execute, then overlay commanded-vs-measured for debugging
            result = robot.execute_cartesian_trajectory(traj, record=True)
            viz.on_step(cycle, traj, result)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        viz.stop()
        robot.disconnect()


if __name__ == "__main__":
    main()
