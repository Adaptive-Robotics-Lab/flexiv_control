# The action contract

> Community project, **NOT affiliated with Flexiv Robotics**.

The action contract is the spine of `flexiv_control`. Every high-level component
— a receding-horizon planner, an MPC horizon, an RL policy, a SpaceMouse bridge —
emits one of a tiny set of objects, and every backend (real Rizon, `fake`,
MuJoCo) knows how to execute them. Nobody re-invents "how do I talk to the
robot," which is exactly what lets one controller serve many projects.

All of these live in `flexiv_control.action_chunk` and are re-exported from the
top level:

```python
from flexiv_control import (
    CartesianTrajectory, CartesianWaypoint, CartesianDelta,
    JointTrajectory, JointWaypoint, GripperCommand, ExecutionResult,
)
```

## Conventions (read this once)

These hold everywhere in the library, end to end:

- **Units are SI**: metres, radians, seconds, Newtons, Newton-metres.
- **Quaternions are `(w, x, y, z)`** — the same order Flexiv RDK uses for TCP
  pose `[x, y, z, qw, qx, qy, qz]`. One convention end to end avoids a whole
  class of silent sign/order bugs.
- **A TCP pose is a length-7 array** `[x, y, z, qw, qx, qy, qz]` in the robot
  **base** frame by default. Frames are named strings (`"base"`, `"tcp"`,
  `"flange"`, `"world"`); the default is `"base"`.
- **`n_frames` and `duration` are interchangeable.** A waypoint may specify
  either. `n_frames` (a per-waypoint frame count) is converted to seconds at the active
  control rate: `duration = n_frames / control_hz`. The same trajectory therefore
  runs identically whether the loop is 100 Hz or 1 kHz, as long as `n_frames`
  is interpreted at that rate.

## `CartesianWaypoint` — one Cartesian target

```python
CartesianWaypoint(
    position,              # required, length-3 [x, y, z]
    quaternion=None,       # (w, x, y, z); None -> HOLD previous/current orientation
    gripper=None,          # GripperCommand; None -> HOLD gripper
    n_frames=None,         # frame count (int > 0)           } give exactly
    duration=None,         # seconds (alternative to n_frames) } one of these
    frame="base",
)
```

`quaternion=None` means *hold orientation*, which is exactly what a
position-only planner (one emitting `(x, y, z)` waypoints) wants — it maps in with
zero changes. A waypoint must specify either `n_frames` or `duration`.

## `CartesianTrajectory` — the core action type

A short, bounded sequence of waypoints. Used by a receding-horizon planner (execute the first
segment, then replan), MPC (the first slice of a horizon), scripted
manipulation, and high-level RL.

```python
CartesianTrajectory(
    waypoints,                       # List[CartesianWaypoint], >= 1
    impedance=ImpedanceParams(),     # compliance (applied when the executor
                                     #   auto-starts the Cartesian mode)
    force_control=None,              # ForceControlParams or None
    max_tcp_linear_speed=0.25,       # m/s    } TIGHTENING-ONLY: the executor runs
    max_tcp_angular_speed=0.60,      # rad/s  } at min(trajectory cap, profile cap)
    max_tcp_linear_acc=1.0,          # m/s^2  } ADVISORY metadata only (the
    max_tcp_angular_acc=2.0,         # rad/s^2} profile's max_linear_accel binds)
    max_contact_wrench=None,         # [fx,..]; min(trajectory, profile); None -> profile
    safety_profile="",               # "" = run under the active profile; a
                                     #   non-empty name is VERIFIED against the
                                     #   active profile (mismatch raises) and both
                                     #   are recorded in ExecutionResult.log
    frame="base",
)
```

Execution notes:

- `execute_cartesian_trajectory` **auto-ensures the NRT Cartesian impedance mode**
  (with the trajectory's `impedance`) when the backend is not already in a Cartesian
  mode, so a missing `start_cartesian_impedance()` is no longer a
  hardware-only failure.
- Segments whose implied speed exceeds the effective cap are **time-stretched**
  by the interpolator (the trajectory still reaches its waypoints, just no faster
  than the cap) -- wall-clock execution can exceed the nominal
  `total_duration`. Use `ExecutionResult.executed_duration` for the truth, or
  estimate ahead with the active profile's caps
  (`RemoteRobot.get_safety_profile()`).
- Preflight a trajectory against the active envelope with
  `profile.validate_chunk(trajectory)` instead of duplicating workspace constants
  client-side.

Helpful attributes: `trajectory.horizon` (number of waypoints) and
`trajectory.total_duration(control_hz)`.

### `CartesianTrajectory.from_waypoint_array(u)` — ingest a planner's action array

A common receding-horizon / action-chunking action has the form `u = ((x_j, y_j, z_j, w_j, n_j))_{j=1..H}`:
a short sequence of Cartesian positions, each with a normalised gripper command
`w_j ∈ [0, 1]` (1 = open, 0 = closed) and an integer number of low-level control
frames `n_j`. This constructor turns that `(H, 5)` array straight into a trajectory:

```python
import numpy as np
from flexiv_control import CartesianTrajectory

u = np.array([
    [0.45, 0.00, 0.30, 1.0, 20],   # reach, gripper open, 20 frames
    [0.50, 0.00, 0.22, 1.0, 15],   # descend
    [0.50, 0.00, 0.22, 0.0, 10],   # close gripper
])
trajectory = CartesianTrajectory.from_waypoint_array(u)            # orientation held throughout
```

The normalised gripper maps to a width with `width = clip(w, 0, 1) * 0.08`
(≈ the 80 mm stroke of the lab's gripper). Override `gripper_force=` and pass any
`CartesianTrajectory` field through `**chunk_kwargs` (e.g.
`safety_profile="contact_manipulation"`). `from_waypoint_array` is position-only,
so orientation is always held; use `from_pose_array` (below) to command it.

### Canonical trajectory extensions

`CartesianTrajectory` is a *canonical* action trajectory: beyond the per-waypoint SE(3)
pose + gripper + duration and the per-trajectory impedance/force/limits, it carries:

- **`representation`** (`TrajectoryRepresentation.ABSOLUTE` default, or
  `RELATIVE_TO_START`). Absolute = waypoints are targets in `frame` (the
  ALOHA/DROID convention). Relative = each waypoint is a pose *relative to the
  TCP pose at trajectory start* (`T_abs = T_start · T_rel`, the UMI relative-trajectory
  convention), re-anchored to the live pose at execution so it never accumulates
  sequential step-to-step error. Which is better is task/calibration dependent —
  it is an explicit field, not a baked-in default.
- **Predict-vs-execute horizon.** `horizon_pred` = predicted waypoints;
  `n_execute` → `horizon_exec` = how many actually run before replanning. A
  `Robot` executes only the first `horizon_exec` waypoints (receding horizon;
  Diffusion-Policy reference: predict ~16, execute ~8).
- **Orientation-carrying array:** `from_pose_array(u)` ingests an `(H, 9)` array
  of `(x, y, z, qw, qx, qy, qz, w, n)` rows — the orientation-bearing sibling of
  the position-only `(H, 5)` `from_waypoint_array`.

```python
from flexiv_control import CartesianTrajectory, TrajectoryRepresentation
trajectory = CartesianTrajectory.from_pose_array(u_h9,         # (H, 9) SE(3) + gripper + n
                                       representation=TrajectoryRepresentation.RELATIVE_TO_START,
                                       n_execute=8)   # predict H, execute 8, replan
```

## `CartesianDelta` — the RL / MPC / teleop workhorse

A *relative* end-effector move integrated on top of the current pose. This is
the standard RL/MPC/teleop action and maps 1:1 onto robosuite/MuJoCo OSC-style
actions, which is what keeps sim→real transfer clean.

```python
CartesianDelta(
    delta,             # length-6 [dx, dy, dz, drx, dry, drz] (rotation = axis-angle rotvec)
    gripper=None,
    duration=0.05,     # 20 Hz default control step
    frame="base",
)
```

You rarely build this by hand — `Robot.servo_cartesian_delta([...])` and the Gym
env do it for you.

## Joint space — `JointWaypoint` / `JointTrajectory`

For resets, homing, and executing a planned joint trajectory (e.g. from MoveIt):

```python
JointWaypoint(positions, n_frames=None, duration=None)   # give one of the two

JointTrajectory(
    waypoints,                      # List[JointWaypoint], >= 1
    max_joint_speed_scale=0.3,      # fraction of nominal joint-velocity limits
    safety_profile="",              # same semantics as CartesianTrajectory: "" = active
)
```

## `GripperCommand` and direct effort MPC

The library exposes two explicit, non-interchangeable gripper contracts. Legacy
position trajectories use opening `width` through Flexiv RDK
`Gripper.Move(width, velocity, force_limit)`. Direct-effort MPC uses
`JointGripperForceTarget(force)` through `Gripper.Grasp(force)`: positive Newtons
close, negative Newtons open, and measured width is state rather than an action.
The runtime `GripperParams.min_force/max_force` values are authoritative; a
device that does not support a requested signed force is rejected before any
trajectory write. Neither contract is an implicit binary open/close flag.

```python
GripperCommand(
    width=0.0,        # metres
    force=20.0,       # Newtons (clamping force)
    velocity=0.1,     # m/s
    grasp=False,      # True -> move-until-contact (RDK Grasp); False -> position move
)

# normalized learning-layer command [0,1] -> physical width (pass your stroke):
GripperCommand.from_normalized(0.7, span=0.08)   # width = 0.7 * 0.08 m
```

For an effort trajectory, use the separately tagged action:

```python
JointGripperForceTarget(force=-20.0)  # direct opening force, if live limits allow
JointGripperForceTarget(force=35.0)   # direct closing force
```

If a planner samples a dimensionless signed-effort latent, decode it exactly
once before constructing the RPC payload:

```python
JointGripperForceTarget.from_signed_effort_latent(z, force_limit=80.0)
# z in [-1,1]: positive closes, negative opens; payload is physical Newtons
```

This is intentionally different from the legacy positional
`GripperCommand.from_signed_action`, where positive means open and the decoded
quantity is a width in metres. Neither latent convention appears on the wire.

**`grasp=True` ignores `width` on hardware**: the RDK backend calls
`Gripper.Grasp(force)` and the fingers close until contact at the force limit
-- a planner that encodes close-intent by thresholding width must not expect
the commanded width to be tracked. The MuJoCo backend mirrors this (a grasp
command drives the fingers closed and contact physics stops them), so sim and
real agree. Gripper RPCs are fire-and-forget by default; use
`command_gripper(cmd, wait=True)` to block until the fingers settle (no more
do-nothing hold-pose trajectories as a wait workaround).

A gripper command embedded in a trajectory waypoint is emitted on the FIRST tick of
that waypoint's segment and runs concurrently with the motion toward the
waypoint -- give the gripper its own (near-)stationary waypoint when you need
it to finish before the arm moves on.

## `ExecutionResult` — quantifies the "execution" failure bucket

Every blocking execution call returns this. A receding-horizon planner's failure taxonomy typically has an
"execution" category; this object turns that category into something observable
and quantifiable rather than a guess. The planner can attribute a bad outcome to
execution vs perception/ranking *with evidence*.

```python
result = robot.execute_cartesian_trajectory(trajectory)
result.success              # bool
result.clipped              # was any setpoint modified to stay in bounds?
result.stop_reason          # "none" | "workspace_limit" | "contact_wrench" | ...
result.executed_duration    # seconds actually run
result.path_tracking_error  # max ||pose_cmd - pose_meas|| over the run (m)
result.max_tcp_speed        # m/s
result.max_joint_speed      # rad/s
result.max_wrench           # N
result.gripper_width_final  # m
result.final_state          # RobotState snapshot at the end
result.log                  # dict of extra diagnostics
```

A planner that logs `clipped` / `stop_reason` / `path_tracking_error` per trajectory
gets a clean signal for *why* a rollout failed — see
[integration_planner.md](integration_planner.md).

## How each consumer uses the contract

| Consumer | Emits | Path |
|---|---|---|
| Receding-horizon planner | `CartesianTrajectory.from_waypoint_array(u)` | execute first segment → replan |
| MPC loop | `CartesianDelta` per tick, or first slice of a horizon as a `CartesianTrajectory` | [integration_mpc.md](integration_mpc.md) |
| RL policy | `[dx,dy,dz,drx,dry,drz,gripper]` → `CartesianDelta` (via `FlexivRealEnv`) | [integration_rl.md](integration_rl.md) |
| SpaceMouse teleop / RL intervention | `CartesianDelta` per tick | [integration_teleop.md](integration_teleop.md) |
| Reset / home / MoveIt plan | `JointTrajectory` | `Robot.move_joint` / `execute_joint_trajectory` |

Because the contract is the only thing crossing the boundary, the network server
([architecture.md](architecture.md)), the Gym env, and the ROS 2 overlay are all
pure pass-through: they serialize and forward these exact objects.
