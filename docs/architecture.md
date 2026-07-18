# Architecture

> Community project, **NOT affiliated with Flexiv Robotics**.

`flexiv_control` is a thin, fast execution-and-safety layer between a high-level
decision maker (a policy, an MPC planner, a teleoperator) and a Flexiv Rizon
arm. Its one job: take a high-level action, turn it into a safe stream of
setpoints at the control rate, and report back what actually happened.

## The stack

```
            ┌─────────────────────────────────────────────────────────────┐
   high     │  Language policy   MPC planner   RL trainer   SpaceMouse      │
   level    └───────┬───────────────┬─────────────┬─────────────┬──────────┘
                    │               │             │             │
                    ▼               ▼             ▼             ▼
            ┌─────────────────────────────────────────────────────────────┐
   contract │      one action contract:  CartesianTrajectory / CartesianDelta    │
            │                            JointTrajectory / GripperCommand        │
            └───────────────────────────────┬─────────────────────────────┘
                                             ▼
            ┌─────────────────────────────────────────────────────────────┐
   control  │  Robot facade  →  SafetyFilter (per-tick)  →  Interpolator     │
            │                    + lease (single writer)   (fixed-rate)      │
            └───────────────────────────────┬─────────────────────────────┘
                                             ▼
            ┌─────────────────────────────────────────────────────────────┐
   backend  │   FlexivRdkBackend     FakeBackend      MujocoBackend          │
            └───────────────────────────────┬─────────────────────────────┘
                                             ▼
                                        ┌─────────┐
   robot                                │  Rizon  │  (hard RT impedance loop
                                        └─────────┘   runs *inside* the robot)
```

Everything above the backend line is pure-Python + numpy. The backend line is
the only place hardware-specific code lives.

## Two tiers of real-time

The single most important hardware fact driving this design: **the Rizon runs
its hard real-time impedance/motion loop inside the robot controller.** The host
streams setpoints; the robot does the 1 kHz servoing. This is more forgiving
than a libfranka-style system where the host must close the torque loop itself.

That gives two tiers that share *everything* above the backend:

- **Tier A — Python, ships first, no root, no C++.** The Python control loop
  streams setpoints through the RDK's **non-real-time** modes
  (`SendCartesianMotionForce`, `SendJointPosition`, …) at 100–500 Hz. The robot
  interpolates internally, so this is reactive enough for chunked policy
  execution, MPC, and RL. Works on the **Standard** RDK license.
- **Tier B — C++ 1 kHz daemon, optional upgrade.** When you need a true 1 kHz
  *host* loop (high-rate streaming MPC, torque research, tight contact), the C++
  `rt_server` (`cpp/`) uses `flexiv::rdk::Scheduler` and the RDK **real-time**
  modes. It speaks the **same wire protocol** as the Python server (today only
  the streaming subset — `set_cartesian_target`/`get_state`/`stop`; the rest is a
  purely-additive roadmap item). Needs a real-time kernel, root, and the
  **Professional** RDK license.

You start on Tier A and only reach for Tier B if a measurement says you must.
See [design_rationale.md](design_rationale.md) for why this beats a
"Python can't do 1 kHz, so everything must be C++/ROS" framing.

## Components

| Component | File | Responsibility |
|---|---|---|
| Action contract | `trajectory.py`, `types.py` | `CartesianTrajectory`/`CartesianDelta`/`JointTrajectory`, `GripperCommand`, `ExecutionResult`. `CartesianTrajectory.from_waypoint_array(u)` ingests a planner's `(H,5)` array directly. |
| `Robot` facade | `robot.py` | The one object you use. `connect`, lease, mode start, `servo_cartesian_delta`, `execute_cartesian_trajectory`, `move_joint`, `home`, `stop`. Owns the per-tick loop. |
| Safety supervisor | `safety.py` | Named YAML `SafetyProfile` + cheap per-tick `SafetyFilter` (workspace box, speed cap, pose-jump cap, joint limits, contact-wrench stop, watchdog). Clips or rejects, and *reports*. |
| Interpolator | `interpolation.py` | Expands a trajectory into one setpoint per control tick (linear position + SLERP). **Velocity-aware**: time-stretches a segment that would exceed the profile's speed cap so it still reaches the waypoint instead of being spatially clipped short. |
| Backends | `backends/` | `RobotBackend` ABC; `FakeBackend` (dependency-free sim), `FlexivRdkBackend` (real), `MujocoBackend` (stub for real2sim2real). `get_backend(name)`. |
| Server | `server/` | `FlexivControlServer`: one owner of the backend, a `Lease` (single writer, TTL + heartbeat), newline-JSON over TCP. `ReactiveServoLoop` is the always-on single-writer setpoint loop. |
| Client | `client/` | `RemoteRobot`: mirrors the `Robot` API over the wire, with lease heartbeat. Lets an RL/MPC author on another machine drive the arm with a `pip install` and no ROS. |
| Gym env | `envs/gym_env.py` | `FlexivRealEnv` (Gymnasium): 7-dim `[dx,dy,dz,drx,dry,drz,gripper]` action (rotation = axis-angle rotvec), 28-dim observation. Works against a local `Robot` or a `RemoteRobot`. |
| LeRobot adapter | `adapters/lerobot_robot.py` | `LeRobotFlexivAdapter` implementing LeRobot's `Robot` interface — the single biggest community lever (free data collection / training / visualization). |
| Teleop | `teleop/spacemouse.py` | `SpaceMouseTeleop` + sources. The SpaceMouse doubles as teleoperation **and** an RL intervention device (HIL-SERL style). |
| CLI | `cli.py` | `flexiv-control serve | home | state | demo`. |
| C++ Tier-B | `cpp/` | `rt_server` 1 kHz daemon (see above). |
| ROS 2 overlay | `ros2/` | Optional. `flexiv_control_msgs` + a bringup node bridging the contract to ROS topics/services/actions, including a MoveIt-Servo-compatible jog input. |

## Why these boundaries

The contract is the spine. Every consumer (policy, MPC, RL, teleop) speaks the
same `CartesianTrajectory` / `CartesianDelta`, every backend consumes the same
filtered setpoint stream, and the network/ROS layers are pure pass-through. That
is what makes the same controller reusable across a receding-horizon planning
project, a lab mate's RL work, and the wider community without forks.

This convergent shape — **C++/host RT loop + thin Python client + a single
action interface + sim and real behind one backend switch** — is what Deoxys,
Polymetis, frankapy, SERL/HIL-SERL, and LeRobot all settled on. See
[design_rationale.md](design_rationale.md) for the field survey.
