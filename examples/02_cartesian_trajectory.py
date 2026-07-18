#!/usr/bin/env python3
"""02 - Cartesian action trajs (receding-horizon planner style).

A receding-horizon planner emits a short traj of Cartesian waypoints

    u = ((x, y, z, w, n), ...)            # w = gripper in [0, 1], n = #frames

and runs receding-horizon: it executes only the FIRST segment, observes, then
replans. This example shows exactly that loop against the fake backend.

    python examples/02_cartesian_chunk.py

The key call is ``robot.execute_cartesian_trajectory(traj)`` which:
  * expands the traj to per-tick setpoints at ``control_hz`` (SLERP on
    orientation, gripper latched on the first tick of a segment),
  * time-stretches any segment that would exceed the safety profile's velocity
    limit (so it still reaches the waypoint, just no faster than allowed),
  * runs every setpoint through the safety filter, and
  * returns an :class:`ExecutionResult` with tracking error, peak speed/wrench,
    whether anything was clipped, and a stop reason -- the exact signal a planner
    logs under its "execution" failure category.
"""

import numpy as np

from flexiv_control import CartesianTrajectory, Robot, RobotConfig


def fake_policy(obs_tcp_xyz: np.ndarray, step: int) -> np.ndarray:
    """Stand-in for a receding-horizon planner: returns a (H, 5) traj.

    Here we just nudge the TCP along a little square in front of the robot and
    toggle the gripper. A real policy would build a MuJoCo scene from RGB-D,
    CEM-sample trajs, score "future videos" with a VLM, and return the best.
    """
    x, y, z = obs_tcp_xyz
    targets = [
        (x + 0.04, y + 0.00, z, 1.0, 20),
        (x + 0.04, y + 0.04, z, 1.0, 20),
        (x + 0.00, y + 0.04, z, 0.0, 20),
    ]
    return np.array(targets, dtype=float)


def main() -> None:
    robot = Robot(RobotConfig(backend="fake", control_hz=100.0))
    with robot:
        robot.acquire_lease("planner")
        # Cartesian impedance is the right mode for contact-rich manipulation.
        robot.start_cartesian_impedance(realtime=False)

        for step in range(4):  # 4 replans
            tcp_xyz = robot.get_state().tcp_position
            u = fake_policy(tcp_xyz, step)
            traj = CartesianTrajectory.from_waypoint_array(u)

            # Receding horizon: execute the FIRST segment only, then replan.
            first_segment = CartesianTrajectory(
                waypoints=traj.waypoints[:1],
                safety_profile=traj.safety_profile,
            )
            result = robot.execute_cartesian_trajectory(first_segment)

            print(
                f"step {step}: reached {np.round(result.final_state.tcp_position, 3)} "
                f"| success={result.success} clipped={result.clipped} "
                f"| track_err={result.path_tracking_error*1000:.1f} mm "
                f"| peak_speed={result.max_tcp_speed:.3f} m/s "
                f"| stop={result.stop_reason}"
            )

            if not result.success:
                print("  execution stopped by safety -> planner would replan/abort")
                break


if __name__ == "__main__":
    main()
