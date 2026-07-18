# Live visualization & intended-motion preview

`flexiv_control.viz` mirrors a running robot in any browser on the LAN and --
the part that matters for safety -- shows the **intended motion** of the next
trajectory *before* it executes: the true per-tick command path (including
time-stretching), waypoint knots, gripper open/close glyphs, the terminal pose,
and an animated ghost TCP, with the active safety profile's workspace box and
a wrench/status HUD alongside.

```bash
pip install "flexiv-control[viz]"     # viser (browser viewer) + yourdfpy (URDF)
```

## Two topologies

**Standalone mirror** -- watch a robot that something else is controlling
(read-only; the monitor never owns the arm):

```bash
flexiv-control viz --connect <robot-pc>        # then open http://localhost:8080
```

**Embedded in a planner** -- the process that *proposes* trajectories also previews
them (this is what ActAhead's `--viz` flag does):

```python
from flexiv_control.viz import RobotViz

viz = RobotViz()                       # frames mode; pass model=<urdf> for the mesh arm
viz.attach(robot, allow_lease=True)    # we ARE the controlling process
print(viz.url)

pv = viz.preview_chunk(trajectory)          # render the intended motion
result = robot.execute_cartesian_trajectory(trajectory, record=True)
viz.on_step(i, trajectory, result)          # outcome flash + commanded-vs-measured overlay
```

With `RecedingHorizonRunner`, the viz doubles as a go/no-go gate:

```python
runner.run(
    on_propose=viz.gate(require_click=True),   # browser Approve/Reject per trajectory
    on_step=viz.on_step,
)
```

`gate()` also auto-refuses a **stale** preview: if the live TCP has moved more
than 5 mm / 2 degrees from the pose the trajectory was planned from, the rendered
path no longer starts where the robot is, and executing it would be a lie.

## Design rules (why it is trustworthy)

1. **The preview IS the executor.** `preview_chunk` runs the exact code
   `execute_cartesian_trajectory` runs -- `trajectory.for_execution()` resolution,
   tightening-only `min(trajectory, profile)` caps, the real
   `CartesianTrajectoryInterpolator` -- so the rendered path includes
   time-stretching and is the true command stream, never a waypoint lerp.
   A regression test asserts preview == executed command stream.
2. **TCP from the robot, never local FK.** The TCP marker, trail, and preview
   anchor always use the streamed `tcp_pose`. Flexiv's published URDFs are
   accurate for link rendering but flexiv_rdk issue #82 documented a >4 cm
   URDF-vs-robot TCP discrepancy -- the URDF mesh arm is *cosmetic*. If the
   mesh fingertip and the TCP frame visibly diverge, that is correct behavior.
3. **A monitor never owns the arm.** `flexiv-control viz --connect` uses a bare
   `RemoteRobot.connect()` -- no lease (`get_state`/`get_safety_profile` are
   lease-free, served from per-tick snapshots, and never block a running
   trajectory). `attach()` refuses a lease-holding connection unless you say
   `allow_lease=True` (the embedded-planner case). **Never** use
   `with RemoteRobot(...)` for monitoring -- `__enter__` acquires the lease.
4. **The workspace box is the server's truth.** It is drawn from
   `get_safety_profile()` (re-polled every 2 s; amber = clip on violation,
   red = reject) -- not from client-side constants that drift.

## The robot model (optional)

The mesh mirror renders the REAL Rizon arm **and the articulated GN01
gripper**: flexiv_description ships only xacro sources, so the library
generates a self-contained URDF (standalone `xacro`, no ROS) with two
post-processing steps that make it render correctly:

* **mesh paths absolutized** (the vendor mixes `package://` and
  package-relative paths, which break outside a ROS workspace);
* **nested mimic chains flattened** -- the GN01's 4-bar fingers mimic a joint
  that itself mimics the drive joint, which yourdfpy resolves only one level
  deep (five of six finger joints would freeze). After flattening, the whole
  gripper articulates, driven directly by the streamed `gripper_width` in
  metres (the vendor's `finger_width_joint` mimic coefficients, 9.404/-0.155,
  are the same calibration our MJCF uses).

The arm joints are mapped **by name** (`joint1..joint7`) -- the URDF lists the
gripper drive first among actuated joints, so positional feeding would twist
the gripper with joint 1.

Asset resolution order (first hit wins):

1. a previously generated URDF in `~/.cache/flexiv_control/generated/`;
2. any `*.urdf` you provide under `FLEXIV_DESCRIPTION_DIR`;
3. generation from the xacro sources in `FLEXIV_DESCRIPTION_DIR` or the cache;
4. `flexiv-control viz --fetch-assets` downloads the pinned
   [flexiv_description](https://github.com/flexivrobotics/flexiv_description)
   `humble-v1` branch first (explicit consent; the `*-v2` branches describe a
   different robot, and the library never downloads silently);
5. nothing found -> **frames mode**: TCP frame + parametric gripper jaws +
   trail + workspace box + full intended-motion preview. Frames mode is fully
   functional for the safety/debugging use case; assets are never
   load-bearing. (The fake backend should run frames mode: its `q` and
   `tcp_pose` are not FK-consistent, so a mesh arm would lie about the TCP.)

## What you see

| Element | Source |
|---|---|
| mesh arm + articulated GN01 (optional) | generated URDF; `q` by joint name + `gripper_width` -> `finger_width_joint` (cosmetic) |
| TCP frame (+ jaws in frames mode) | streamed `tcp_pose` + `gripper_width` (authoritative) |
| blue trail | last ~25 s of measured TCP positions |
| amber/red wireframe box | active safety profile workspace (clip/reject) |
| time-colored path (blue->red) | the trajectory's true per-tick command stream |
| white knots | trajectory waypoints |
| green/blue spheres | gripper close / open events |
| small frame at path end | terminal pose |
| yellow ghost sphere | animated playback (scrub with the *preview %* slider) |
| gray + magenta overlay | commanded vs measured path after execution (`record=True`) |
| HUD | mode, safety status, stop reason, lease owner, gripper width, loop health, wrench bar |

## Notes

* Poll rate defaults to 20 Hz (`--hz` / `state_hz=`); the server serves states
  from per-tick snapshots, so polling never contends with the control loop.
* Flexiv Elements can stay connected read-only during RDK control and is the
  zero-code fallback monitor; it cannot show intended motions, which is the
  point of this module.
* `examples/08_live_viz.py` runs the whole experience offline on the fake
  backend (no hardware, no assets).
