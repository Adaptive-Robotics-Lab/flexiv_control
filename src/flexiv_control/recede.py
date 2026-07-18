"""Receding-horizon execution: predict a traj, execute H_exec, replan.

The canonical action-chunking loop (Diffusion Policy / openpi / ACT): a policy
maps an observation to an action *traj*; the robot executes only the first
``horizon_exec`` waypoints, then re-observes and re-plans. The ``policy`` is any
callable ``RobotState -> CartesianTrajectory | None`` -- it is the **seam between a VLA
/ planner / MPC and the controller**. It can wrap:

* an openpi / OpenVLA websocket *policy server* (``infer(obs) -> action traj``),
* an MPC solver returning the first slice of its horizon,
* a local sampler + simulated-future ranker (ActAhead),

and ``None`` ends the run. Set ``n_execute`` on the returned traj to control the
execution horizon (Diffusion-Policy reference: predict ~16, execute ~8).
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from .trajectory import CartesianTrajectory, ExecutionResult
from .types import RobotState

Policy = Callable[[RobotState], Optional[CartesianTrajectory]]
OnStep = Callable[[int, CartesianTrajectory, ExecutionResult], Optional[bool]]
OnPropose = Callable[[int, CartesianTrajectory], bool]


def console_confirm(step: int, traj: CartesianTrajectory) -> bool:
    """A ready-made ``on_propose`` gate: summarize the proposed traj and ask
    the operator for y/N before any real motion. This is the per-traj
    confirmation pattern every careful lab loop re-implements with a bare
    ``input()`` buried in planner code."""
    grips = [w.gripper for w in traj.waypoints if w.gripper is not None]
    first = traj.waypoints[0].position
    last = traj.waypoints[-1].position
    print(
        f"[confirm] traj {step}: {traj.horizon} waypoints, "
        f"{first.round(3).tolist()} -> {last.round(3).tolist()}, "
        f"{len(grips)} gripper command(s), "
        f"caps {traj.max_tcp_linear_speed:.2f} m/s / {traj.max_tcp_angular_speed:.2f} rad/s"
    )
    return input(f"execute traj {step} on the robot? [y/N] ").strip().lower() == "y"


class RecedingHorizonRunner:
    """Drive a robot with a policy in a receding-horizon loop.

    ``robot`` is anything with ``get_state()`` and ``execute_cartesian_trajectory()``
    (a :class:`~flexiv_control.Robot` or a
    :class:`~flexiv_control.RemoteRobot`).

    ``observe`` lets the policy consume something richer than the robot's
    proprioceptive state -- e.g. an external camera observation -- without
    bypassing the runner: it is called once per cycle and its return value is
    passed to ``policy``. Default: ``robot.get_state``.
    """

    def __init__(
        self,
        robot,
        policy: Policy,
        *,
        max_steps: Optional[int] = None,
        observe: Optional[Callable[[], object]] = None,
    ):
        self.robot = robot
        self.policy = policy
        self.max_steps = max_steps
        self.observe = observe

    def run(
        self,
        *,
        on_step: Optional[OnStep] = None,
        on_propose: Optional[OnPropose] = None,
    ) -> int:
        """Blocking loop: observe -> policy -> (confirm) -> execute the traj's
        first ``horizon_exec`` waypoints -> repeat. Returns the number of
        executed trajs; stops when the policy returns ``None``, ``on_propose``
        returns ``False`` (the pre-execution gate -- pass
        :func:`console_confirm` for an interactive y/N), ``on_step`` returns
        ``False``, or ``max_steps`` is reached."""
        steps = 0
        while self.max_steps is None or steps < self.max_steps:
            obs = self.observe() if self.observe is not None else self.robot.get_state()
            traj = self.policy(obs)
            if traj is None:
                break
            if on_propose is not None and not on_propose(steps + 1, traj):
                break
            result = self.robot.execute_cartesian_trajectory(traj)
            steps += 1
            if on_step is not None and on_step(steps, traj, result) is False:
                break
        return steps

    def run_streaming(self, loop, *, replan_hz: float = 5.0) -> int:
        """Async infer-ahead loop over a :class:`ReactiveServoLoop`: enqueue the
        next traj while the loop streams the current one. The loop is the single
        writer and holds-on-stale if the policy stalls. Note this is infer-ahead,
        not true real-time chunking (RTC): a newly enqueued traj PREEMPTS the
        current one at the next tick (a hard swap), it is not velocity-blended
        across the boundary.

        ``loop`` must already be started. Returns the number of trajs enqueued.
        """
        steps = 0
        period = 1.0 / float(replan_hz)
        next_t = time.perf_counter()
        while self.max_steps is None or steps < self.max_steps:
            obs = loop.get_state()
            traj = self.policy(obs)
            if traj is None:
                break
            loop.enqueue_trajectory(traj)
            steps += 1
            next_t += period
            sl = next_t - time.perf_counter()
            if sl > 0:
                time.sleep(sl)
        return steps
