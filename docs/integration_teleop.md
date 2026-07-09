# Integration: teleoperation

> Community project, **NOT affiliated with Flexiv Robotics**.

Teleop runs a reactive servo loop: every tick it reads the SpaceMouse and
streams a Cartesian delta at `control_hz`, with the safety watchdog holding
position if input goes stale. The same device is the RL **intervention** source
(HIL-SERL, see [integration_rl.md](integration_rl.md)). See
`examples/05_spacemouse_teleop.py`.

## Quick start (no device)

The scripted source traces a small circle so the whole path runs on the `fake`
backend with no hardware at all. `teleop.run()` acquires the lease, starts
Cartesian impedance, then reads the source and streams a delta every tick until
the time budget is up:

```python
from flexiv_control import Robot, RobotConfig
from flexiv_control.teleop import SpaceMouseTeleop, ScriptedSpaceMouseSource

robot = Robot(RobotConfig(backend="fake", control_hz=200.0))

with robot:
    teleop = SpaceMouseTeleop(robot=robot, source=ScriptedSpaceMouseSource())
    teleop.run(duration=3.0)        # or run(max_ticks=600)
```

```bash
python examples/05_spacemouse_teleop.py            # scripted, no device
python examples/05_spacemouse_teleop.py --device   # real SpaceMouse
```

## Real device

```bash
pip install "flexiv-control[teleop]"     # pyspacemouse / hidapi
```

```python
from flexiv_control import Robot, RobotConfig
from flexiv_control.teleop import SpaceMouseTeleop, PySpaceMouseSource

robot = Robot(RobotConfig(backend="rizon4s_lab"))   # your real config
with robot:
    teleop = SpaceMouseTeleop(
        robot=robot,
        source=PySpaceMouseSource(),    # raises if driver/device missing
        deadband=0.05,                  # ignore tiny jitter
        deadman_button=0,               # motion only while this button is held
        gripper_button=1,               # toggles the gripper
        pos_scale=0.05,                 # m/s at full deflection
        rot_scale=0.6,                  # rad/s at full deflection
    )
    teleop.run(duration=30.0)
```

On Linux a SpaceMouse is usually read via `pyspacemouse`/`hidapi` (the extra
above). If you prefer the system `spacenavd` daemon you can write a small custom
`SpaceMouseSource` subclass (implement `open` / `read` / `close` returning a
`SpaceMouseState`); the teleop logic is source-agnostic.

## Key behaviours

- **Deadman gate.** With `deadman_button` set, deltas are zero unless that button
  is held — release it and the arm holds. Set it to `None` to disable gating.
- **Deadband.** Per-axis input below `deadband` is zeroed so the arm does not
  drift on a noisy neutral.
- **Stale-input watchdog.** If the loop stops sending fresh deltas, the safety
  watchdog holds the last pose (see [safety.md](safety.md)) — teleop fails safe.
- **Same path as everything else.** Teleop emits `CartesianDelta` through the
  same `Robot` → safety filter → backend path as MPC and RL. Nothing about the
  device is special to the robot.

## RL intervention (HIL-SERL)

`teleop.intervention(policy_action)` returns `(action, intervened)`: while the
deadman is held and the device moves, the human delta overrides the policy and
`intervened=True`. That flag is the supervision signal HIL-SERL trains on — see
[integration_rl.md](integration_rl.md).

## Relationship to your existing teleop repo

Your standalone `flexiv-spacemouse-teleop` project drives the arm through
**MoveIt Servo** Cartesian twists (`/servo_node/delta_twist_cmds`,
`TwistStamped`) → `flexiv_ros2` / `ros2_control` → Flexiv RDK. That is a fine,
ROS-native teleop stack and you do not need to replace it.

`flexiv_control` offers two ways to relate to it:

- **Use this library's teleop directly** (no ROS): the Python path above drives
  the arm through the unified contract and safety filter, and the *same*
  SpaceMouse doubles as an RL intervention device — which the ROS/MoveIt-Servo
  teleop does not give you.
- **Keep MoveIt Servo, gain a uniform backend.** The optional ROS 2 overlay
  (`ros2/flexiv_control_bringup`) exposes a `~/delta_twist_cmds` `TwistStamped`
  input that is **MoveIt-Servo-compatible**: point your existing Servo pipeline
  at the bringup node and the twist flows through the same safety filter, state
  reporting, and action server as the rest of the stack. This lets your teleop
  repo and your RL/MPC code share one controller and one safety profile instead
  of two parallel control paths.

  Mind the **unit semantics** when you rewire: a MoveIt-Servo-style joystick
  pipeline publishes *unitless* values in `[-1, 1]` that Servo scales to real
  speeds. The bringup node's default (`twist_in_type: unitless`,
  `twist_scale_linear: 0.4` m/s, `twist_scale_rotational: 0.8` rad/s) matches
  the flexiv_ros2 Servo config, so the same command stream moves at the same
  speed. If your publisher sends real m/s / rad/s instead, set
  `twist_in_type: speed_units`. Stale commands older than `twist_max_age`
  (default 0.25 s) are dropped. Re-verify speeds on the real arm after any
  rewiring.

Either way, the win is a single execution-and-safety layer underneath teleop,
RL, MPC, and a high-level planner — not three different ways to command the same arm.

## ROS 2 overlay

If you live in ROS, the overlay publishes `RobotState`/`JointState`, offers
`set_mode` / `home` / `stop` services and an `ExecuteCartesianChunk` action, and
accepts the MoveIt-Servo-compatible jog input. Build it as a normal ament
package; see `ros2/` and [architecture.md](architecture.md). It is **optional** —
RL/MPC users who do not want a ROS workspace never touch it.
