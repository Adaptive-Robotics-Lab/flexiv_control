#!/usr/bin/env python3
"""07 - Receding-horizon execution + the VLA / MPC policy-server seam.

A policy maps an observation to an action CHUNK; the robot executes only the
first ``horizon_exec`` waypoints, then re-observes and re-plans (the Diffusion
Policy / openpi pattern). The ``policy`` callable is the seam -- wrap an openpi /
OpenVLA websocket policy server (``infer(obs) -> traj``), an MPC solver, or a
local sampler + simulated-future ranker (ActAhead) behind it.

Two modes:
  (A) blocking      -- observe -> traj -> execute H_exec -> replan
  (B) streaming     -- enqueue the next traj while the single-writer loop streams
                       the current one (infer-ahead; the loop holds-on-stale)

Runs on the fake backend, no hardware.
"""

import numpy as np

from flexiv_control import CartesianTrajectory, RecedingHorizonRunner, Robot, RobotConfig
from flexiv_control.server import ReactiveServoLoop


def make_policy(goal_x: float):
    """A trivial stand-in 'policy': step the TCP toward a goal x and stop there.

    Replace the body with your real policy -- e.g. an openpi
    ``websocket_client_policy.infer(obs)`` returning an action traj, an MPC
    solve, or an ActAhead sample+rank. Return ``None`` to end the run.
    """

    def policy(obs):
        x = float(obs.tcp_pose[0])
        if x >= goal_x - 0.005:
            return None
        tgt = obs.tcp_pose.copy()
        tgt[0] = min(x + 0.02, goal_x)
        # predict a short traj; execute only its first waypoint (receding
        # horizon). Naming a safety_profile pins the envelope the traj was
        # planned for -- the executor VERIFIES it against the robot's active
        # profile (mismatch raises), so set the robot's profile to match (see
        # main()) or leave the field "" to run under whatever is active.
        return CartesianTrajectory.from_pose_array(
            np.concatenate([tgt, [1.0, 40]])[None, :],
            n_execute=1,
            safety_profile="free_space_fast",
        )

    return policy


def main() -> None:
    # (A) blocking receding horizon
    robot = Robot(RobotConfig(backend="fake"))
    robot.connect()
    robot.set_safety_profile("free_space_fast")  # match the trajs' named profile
    robot.start_cartesian_impedance()
    goal = float(robot.get_state().tcp_pose[0]) + 0.08
    print(f"(A) blocking receding-horizon toward x={goal:.3f}")
    n = RecedingHorizonRunner(robot, make_policy(goal), max_steps=20).run(
        on_step=lambda i, c, r: print(f"    step {i}: x={r.final_state.tcp_pose[0]:.3f}")
    )
    print(f"    done in {n} steps, x={robot.get_state().tcp_pose[0]:.3f}")
    robot.disconnect()

    # (B) streaming / infer-ahead over the single-writer loop
    robot2 = Robot(RobotConfig(backend="fake"))
    robot2.connect()
    robot2.start_cartesian_impedance()
    goal2 = float(robot2.get_state().tcp_pose[0]) + 0.08
    print(f"(B) streaming (infer-ahead) receding-horizon toward x={goal2:.3f}")
    with ReactiveServoLoop(robot2) as loop:
        m = RecedingHorizonRunner(robot2, make_policy(goal2), max_steps=20).run_streaming(
            loop, replan_hz=10.0
        )
    print(f"    enqueued {m} trajs, x={robot2.get_state().tcp_pose[0]:.3f}")
    robot2.disconnect()


if __name__ == "__main__":
    main()
