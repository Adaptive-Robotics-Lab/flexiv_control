# Integration: MPC / high-rate control

> Community project, **NOT affiliated with Flexiv Robotics**.

An MPC controller solves for an action from the latest state and streams a fresh
setpoint every tick. `flexiv_control` supports that directly: read state →
compute a target → send it → repeat, at whatever rate your solver runs, with the
safety filter on every setpoint. See `examples/04_mpc_loop.py`.

## The basic loop

```python
import time, numpy as np
from flexiv_control import Robot, RobotConfig

robot = Robot(RobotConfig(backend="fake", control_hz=100.0))  # or "rizon4s_lab"
robot.connect()
robot.start_cartesian_impedance()
dt = robot.dt                                  # 1 / control_hz

while running:
    state = robot.get_state()                  # latest measured state
    pose_or_delta = solver.step(state, dt)     # YOUR MPC
    robot.servo_cartesian_pose(pose_or_delta, duration=dt)
    # (time pacing handled by the blocking call; add sleep only if your
    #  solver is faster than dt)
```

Two equivalent ways to command a single setpoint per tick, both filtered:

- `robot.servo_cartesian_pose(pose, duration=dt)` — absolute TCP pose
  `[x,y,z,qw,qx,qy,qz]`.
- `robot.servo_cartesian_delta(delta, duration=dt)` — relative
  `[dx,dy,dz,drx,dry,drz]` integrated on the current pose (maps 1:1 to
  robosuite/MuJoCo OSC actions). See [action_contract.md](action_contract.md).

## The always-on streaming loop: `ReactiveServoLoop`

For the lowest-latency Python path — a single-writer loop that runs at
`control_hz` and to which your solver just publishes the **latest target**
(rather than calling a blocking execute each tick) — use
`flexiv_control.server.ReactiveServoLoop`. This is the Python analogue of the
C++ 1 kHz daemon and decouples your solver's compute time from the control rate:

```python
from flexiv_control import Robot, RobotConfig
from flexiv_control.server import ReactiveServoLoop

robot = Robot(RobotConfig(backend="fake", control_hz=200.0))
robot.connect(); robot.start_cartesian_impedance()

with ReactiveServoLoop(robot) as loop:         # background fixed-rate writer
    while running:
        state = loop.get_state()
        target_pose = solver.step(state)       # may be slower than the loop
        loop.set_cartesian_target(target_pose) # newest target wins; loop streams it
    # watchdog holds the last pose if you stop publishing
```

The loop owns the single writer to the backend; the safety watchdog holds the
last commanded pose if targets go stale, so a slow or stalled solver fails safe
instead of jerking. (`loop.set_joint_target(q)` is the joint-space equivalent.)

## How fast can you go?

This is the Tier-A / Tier-B decision (see [architecture.md](architecture.md) and
[flexiv_setup.md](flexiv_setup.md)):

- **Tier A — Python, 100–500 Hz, Standard license.** The host streams setpoints
  through RDK's non-real-time modes and the **robot interpolates internally** at
  its own 1 kHz. For most MPC (50–200 Hz solve rates) this is plenty reactive
  and needs no real-time kernel, root, or C++. Start here.
- **Tier B — C++ 1 kHz host loop, Professional license.** When you need a true
  1 kHz *host* loop (very high-rate streaming MPC, tight contact, torque-level
  research), run the optional `cpp/rt_server` (`flexiv::rdk::Scheduler`, RT
  modes). It speaks the **same wire protocol**, but today implements only the
  streaming subset; see [`cpp/README.md`](../cpp/README.md) for the exact method
  set and how to drive it (a stock `RemoteRobot` needs the lease handlers first).

Practical guidance: profile first. If your end-to-end latency budget is met at
Tier A (it usually is for vision-driven MPC), stay there. Reach for Tier B only
when a measurement shows the *host* command rate — not perception or solve time
— is the bottleneck.

## Cross-machine MPC

Run the server next to the robot and the solver on your compute box:

```python
from flexiv_control import RemoteRobot
with RemoteRobot("ROBOT_HOST_IP", 8766, owner="mpc") as r:
    r.acquire_lease()
    r.start_cartesian_impedance()
    while running:
        s = r.get_state()
        r.servo_cartesian_pose(solver.step(s), duration=dt)
```

`RemoteRobot` mirrors the `Robot` API over newline-JSON/TCP, holds the lease with
a heartbeat, and installs with only numpy.

## Strict receding-horizon joint prefixes

For physics-sampling MPC whose action is generalized effort, use
`JointTorqueTrajectory` instead of the position-target example below. It streams
seven additional joint torques at 1 kHz with RDK nonlinear-dynamics compensation
and firmware soft limits enabled, and dispatches an optional
`JointGripperForceTarget` at the same segment boundaries. This API is disabled by
default and is admitted only when the deployment, live `RobotInfo.tau_max`,
safety profile, continuity anchor, and gripper force limits all agree. See
[`joint_torque_mpc.md`](joint_torque_mpc.md).

Use one atomic `execute_joint_trajectory` call for a multi-segment actuator
prefix. Knot 0 is the previous commanded target, not a fresh measured-state
sample; normal physical tracking lag therefore does not bend the next spline.

```python
import numpy as np
from flexiv_control import JointGripperTarget, JointTrajectory, JointWaypoint

knot0 = np.asarray(previous_ack_target, float)  # [q0..q6, gripper_width]
knots = np.asarray(prefix_targets, float)       # shape (N, 8)
frames = [32, 1]

traj = JointTrajectory(
    initial_positions=knot0[:7],
    initial_gripper_width=float(knot0[7]),
    waypoints=[
        JointWaypoint(
            positions=knot[:7],
            gripper=JointGripperTarget(
                width=float(knot[7]),
                force=20.0,
                velocity=None,  # derive abs(delta_width) / segment duration
            ),
            n_frames=n_frames,
        )
        for knot, n_frames in zip(knots, frames)
    ],
    interpolation="linear",
    strict_timing=True,
    max_joint_speed_scale=0.3,
)
result = robot.execute_joint_trajectory(traj)
```

Every strict segment is prevalidated against the effective runtime joint rates
and cached `Gripper.params` before any backend write. Requested `n_frames` are
authoritative: a violation raises instead of clipping or time-stretching. A
provided gripper velocity must equal the velocity that realizes its width delta
in the same segment; `None` derives it. Position-mode `Gripper.Move` is issued
fire-and-forget at the segment boundary, then the arm streams concurrently.

Knot 0 need not equal the prior commanded endpoint because it is not streamed.
Every feedback-MPC call rebases its joint safety filter and first gripper ramp to
current measured state. The actual first interpolated joint target must be
reachable from that measurement in one controller tick under the effective
per-joint rate limits; gripper velocity is likewise derived and validated from
current measured width. This permits the next plan's knot 0 to reflect ordinary
tracking error without hiding a command jump. The previous acknowledged joint
and gripper targets remain in the result log as provenance, not an admission
gate. The executor repeats the first-setpoint check on the freshest snapshot
after any mode transition, immediately before actuator writes, and recomputes
the first gripper-event ramp from width measured at dispatch. Stop, fault, lease
changes, mode changes, and other mutating RPCs clear
that provenance cache. A payload rejected completely during prevalidation
leaves it intact for diagnosis. The result log also includes requested/scheduled
segment and total ticks, interpolation and execution anchors, gripper events,
and measured gripper tracking.

## Safety notes for MPC

- Keep the safety filter on (it is, by default). A misbehaving solver that
  commands a huge step is **rejected → hold**, and an out-of-box target is
  **clipped**, with the reason reported in `result.stop_reason` /
  `state.stop_reason`.
- The interpolator is velocity-aware: if a per-tick target exceeds the profile
  speed cap it is slowed (time-stretched), not truncated — so the loop stays
  smooth under an aggressive command. ([safety.md](safety.md))
- Use `free_space_fast` only after validating there is no contact in the
  workspace; use `contact_manipulation` when you intend contact.
