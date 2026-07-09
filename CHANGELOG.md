# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-07-09

Fixes from the July 2026 cross-repo audit against `flexiv-spacemouse-teleop`.

### Fixed
- **SpaceMouse gripper toggle was level-triggered**: holding the button flipped
  the gripper open/closed once per control tick (~100 Hz on the default loop).
  It now toggles only on the button's rising edge, tracked every tick (also
  while the deadman is released and inside `intervention()`), matching the ROS
  teleop bridge's behaviour.
- **ROS bridge `~/delta_twist_cmds` treated unitless commands as m/s**: the
  node advertises itself as a drop-in target for a MoveIt-Servo teleop
  pipeline, but such pipelines publish joystick-style values in `[-1, 1]` that
  Servo scales down (0.4 m/s / 0.8 rad/s in the flexiv_ros2 config). Feeding
  them straight in executed ~2.5x faster than the Servo path. The node now
  defaults to `twist_in_type: unitless` with matching `twist_scale_linear` /
  `twist_scale_rotational` parameters (`speed_units` restores the old
  behaviour), and drops commands older than `twist_max_age` (default 0.25 s).

### Changed
- Gripper toggle conventions now match the lab's ROS teleop bridge: GN01
  widths 0.09 m open / 0.01 m close (was 0.08 / 0.0), and the first press
  OPENS (was: first press closed). Configure with `gripper_open_width`,
  `gripper_close_width`, `initial_open`.
- `SpaceMouseTeleop` accepts `signs` (six +/-1 values) to calibrate device
  axes to the robot frame without a custom source subclass.

### Housekeeping
- Versions 0.1.6-0.1.8 were lab-internal iterations installed on the control
  PC without changelog entries or tags; their changes are folded into this
  release.

## [0.1.5] - 2026-06-19

### Fixed: 3 more flexiv_rdk-backend bugs from the first real-hardware session

The `flexiv_rdk` backend had never touched a real robot; the first live Rizon4s-062626
session surfaced three RDK-1.7 API/state mismatches that each blocked motion:

- **`Mode` enum**: `_rdk_mode` referenced `Mode.RT_*` members that flexivrdk 1.7 does not
  expose (its `Mode` has only `IDLE` + `NRT_*` + `UNKNOWN`), so building the map
  `AttributeError`'d on the first `SwitchMode`. Each entry is now added only if the member
  exists; the NRT pick-place workload is unaffected.
- **F/T sensor zeroing**: force-control modes (`NRT_CARTESIAN_MOTION_FORCE`, our cartesian
  impedance) fault with event 301004 unless the 6-DoF F/T sensor is zeroed first. `connect()`
  now runs the `ZeroFTSensor` primitive once (arm at rest) — a no-op on robots without an
  F/T sensor.
- **`SendJointPosition`**: flexivrdk 1.7 takes FIVE `list[float]` args
  (pos, vel, acc, max_vel, max_acc); the backend passed four (no acc) → `TypeError` that
  broke the home-restore. Added the acceleration vector and plain-`float` conversion
  (np.float64 is rejected).

All HARDWARE-VERIFIED on Rizon4s-062626 (RDK-Professional, flexivrdk 1.7): enable →
F/T-zero → cartesian-impedance → NRT joint chunks → `go_home_safe` all succeed.

## [0.1.4] - 2026-06-18

### Fixed: GN01 gripper "No gripper enabled" no-op on real hardware

The `flexiv_rdk` backend created the `Gripper` but called `Gripper.Init()` WITHOUT
first calling `Gripper.Enable(device_name)`. On RDK v1.x the gripper device must be
enabled by name before `Init()`, so on real hardware this failed with
`[flexiv::rdk::Gripper::Init] No gripper enabled` and every gripper command silently
became a no-op. The backend now does `Enable(gripper_name)` → `Init()`, guarded on a
non-empty name (an empty `gripper_name` means "no gripper configured" and is skipped
cleanly), with a clearer failure message pointing at Flexiv Elements → Settings →
Device. The shipped `rizon4s_actahead_lab` config now sets `gripper_name: "Flexiv-GN01"`.

Hardware-verified on Rizon4s-062626 (GN01): the server enables the gripper without a
warning and a move to 0.085 m settles open (0.0855 m), aligning the real gripper with
the simulation home.

## [0.1.3] - 2026-06-12

### Added: REAL mesh arm + articulated GN01 gripper in the live mirror

The mesh mirror now shows the actual Rizon 4s and a fully articulating GN01
gripper (the advisor's ask: see the real arm and gripper move, not stand-ins):

- **URDF generation from the vendor xacro sources** (`flexiv_description`
  ships no committed URDF): standalone `xacro` (no ROS) with
  `$(find ...)` shadow substitution, cached self-contained output.
- **Nested mimic chains flattened**: the GN01 4-bar chains mimics two levels
  deep; yourdfpy resolves one level only, freezing five of six finger joints
  (and warning at 20 Hz). Transitive composition of multiplier/offset points
  every mimic at the actuated `finger_width_joint`, so BOTH fingers
  articulate -- driven directly by the streamed gripper width in metres (the
  vendor mimic coefficients 9.404/-0.155 equal our MJCF calibration).
- **Mesh paths absolutized** (vendor mixes `package://` and package-relative
  paths; both break outside a ROS workspace).
- **Joint mapping by NAME** (`joint1..joint7` + `finger_width_joint`): the
  URDF lists the gripper drive FIRST among actuated joints, so positionally
  feeding `state.q` would twist the gripper with joint 1. Parametric jaw
  glyphs auto-hide when the mesh gripper is present.
- `xacro` added to the `[viz]` extra; asset tests incl. a both-fingers
  articulation regression; verified live in a browser (arm sweeping, gripper
  opening/closing, intended-motion preview overlaid on the mesh robot).

## [0.1.2] - 2026-06-11

### Added: live visualization + intended-motion preview (`[viz]` extra)

`flexiv_control.viz` -- a browser-based live mirror (viser) for safety and
debugging, viewable from any machine on the LAN:

- **Live robot mirror**: optional URDF mesh arm (`flexiv_description`, with a
  consented asset fetcher + frames-mode fallback), an authoritative TCP frame
  drawn from the STREAMED `tcp_pose` (never local FK -- flexiv_rdk #82
  documented >4 cm URDF/TCP drift), parametric gripper jaws, a measured TCP
  trail, the active safety profile's workspace box (amber=clip / red=reject,
  re-polled live), and a wrench bar + mode/stop/lease/loop-health HUD.
- **Intended-motion preview** (the headline): `RobotViz.preview_chunk` renders
  the chunk's TRUE per-tick command stream -- the executor's own
  `for_execution` resolution, tightening-only `min(chunk, profile)` caps, and
  the real interpolator, so time-stretching is visible -- as a time-colored
  path with waypoint knots, gripper open/close glyphs, the terminal pose, an
  animated scrubbable ghost TCP, and `validate_chunk` warnings. A regression
  test pins preview == executed command stream.
- **Go/no-go gate**: `viz.gate()` plugs into `RecedingHorizonRunner(on_propose=)`;
  refuses stale previews (live TCP moved > 5 mm / 2 deg since planning) and,
  with `require_click=True`, blocks on browser Approve/Reject buttons.
  `viz.on_step` flashes the outcome and overlays commanded-vs-measured paths
  from `record=True` trajectories.
- **CLI**: `flexiv-control viz --connect <robot-pc>` -- a read-only standalone
  mirror that NEVER takes the lease (monitoring must not own the arm;
  `attach()` enforces this).
- numpy-only preview math (`flexiv_control.viz.preview`) stays importable and
  tested without viser; `examples/08_live_viz.py`; `docs/visualization.md`;
  a `viz` CI job.

## [0.1.1] - 2026-06-11

Driven by a consumer friction audit (the ActAhead real2sim2real runner): four
latent hardware-only failures plus a set of "accepted but not enforced" fields,
all masked by the permissive fake backend.

### Fixed / behavior changes
- **`execute_cartesian_chunk` auto-ensures the NRT Cartesian impedance mode**
  (with the chunk's `impedance`); `FakeBackend.stream_*` now ENFORCE the real
  backend's mode preconditions so dry runs reveal mode-sequencing bugs.
  `ReactiveServoLoop` re-ensures the mode before streaming.
- **The chunk kinematic/contact envelope is enforced, tightening-only**:
  execution runs at `min(chunk caps, profile caps)`; previously
  `max_tcp_linear_speed` etc. were serialized but ignored (dead config that
  consumers visibly relied on).
- **`CartesianChunk.safety_profile` defaults to `""` and is verified**: a
  non-empty name must match the active profile or execution raises; requested
  and active names are recorded in `ExecutionResult.log`.
- **`FlexivRdkBackend.home()` no longer silently ignores a configured
  `q_home`** (the vendor primitive goes to the factory home): `Robot.home()`
  falls back to a speed-capped interpolated joint move to the configured
  posture, then restores `RobotConfig.gripper_home_width` (new field).
- `actahead_lab` profile re-derived from the pick-place runner's envelope (the
  old transcription was the push envelope: 0.05 m/s, x <= 0.85) and set to
  `workspace_action: reject`; `rizon4s_actahead_lab` `q_home` j7 sign fixed
  against the recorded probe and `gripper_home_width: 0.085` added.

### Added
- `move_joint(..., max_joint_speed=)` — speed-parameterized joint moves (the
  natural "go home slowly" shape consumers guessed at and got TypeErrors for).
- `Robot.go_home_safe()` / `RemoteRobot.go_home_safe()` — the end-of-session
  exit ritual (lift -> open gripper -> joint-home) as one resilient call.
- `command_gripper(cmd, wait=True, timeout=)` — blocking gripper commands.
- `get_safety_profile` RPC + `SafetyProfile.to_config_dict()` /
  `validate_chunk()` — clients preflight against the server's ACTIVE envelope
  instead of duplicating workspace constants.
- `SafetyProfile.workspace_action: clip|reject` — opt-in protective stop on
  workspace violation instead of a silent position clip.
- Cooperative cancel: `Robot.request_stop()`; the server's `stop` RPC now
  aborts an in-flight chunk within one control tick (no lease required), and
  `get_state` is served from a per-tick snapshot instead of blocking behind a
  running chunk.
- Lease: an in-flight RPC from the owner counts as liveness (`Lease.hold`);
  any authenticated RPC refreshes the TTL; the client heartbeat attempts one
  re-acquire after an expiry instead of dying.
- `CartesianChunk.from_topdown_array((x, y, z, yaw, w, n))` +
  `transforms.top_down_quat` / `mat_to_quat`; `frames_hz=` on the array
  constructors for planners that tick at their own rate.
- `ExecutionResult.summary()`; `execute_*_chunk(raise_on_stop=True)` raising
  `ChunkStoppedError`; `RecedingHorizonRunner(observe=...)` +
  `run(on_propose=...)` pre-execution gate with a shipped `console_confirm`.
- MuJoCo backend honors `grasp=True` (close-until-contact) so sim and RDK
  gripper semantics agree; documented that `Grasp` ignores `width` on hardware.
- `flexiv-control serve` prints the active safety profile and workspace box.
- `execute_cartesian_chunk(record=True)` fills `result.log["trajectory"]` with
  per-tick `[t, pose_cmd, pose_meas, wrench]` rows, and a failed run records
  `log["stopped_at_waypoint"]` -- the measured-vs-commanded series for
  sim-vs-real attribution that used to be measured every tick and discarded.

### Hardening from the post-fix adversarial review
- **RemoteRobot wire robustness**: motion RPCs (execute/move/home/go_home_safe/
  gripper-wait) read with a long `motion_timeout` (default 600 s) instead of
  the 10 s default that used to raise client-side WHILE THE ARM KEPT MOVING;
  every response's id is validated against its request (a mismatch raises a
  hard desync error instead of silently mis-attributing results).
- **Stops cannot be lost**: a cancel pending at chunk entry aborts that chunk
  (consume-on-abort) instead of being cleared; the server's stop handler
  re-arms the cancel when a chunk races it; `command_gripper(wait=True)`
  honors a stop request; a `force=True` lease steal cancels the displaced
  owner's in-flight motion.
- **Joint moves get the same guards as Cartesian**: per-tick backend-fault and
  contact-wrench gates in both chunk executors, and joint-mode auto-ensure in
  `execute_joint_chunk`.
- `go_home_safe` refuses the blind joint-home when the lift stage ended in
  contact/fault (the arm is plausibly snagged; the operator recovers manually).
- Gripper-wait settle detection requires observed motion or a 0.5 s dwell
  (real hardware has an actuation-latency window in which the unchanged old
  width read as "settled"); `Robot.command_gripper` now returns the final
  width like `RemoteRobot`.
- Servo loop: the writer thread can no longer die silently (faults stop the
  backend, mark FAULT, and de-register liveness; `servo_stream` raises on a
  dead loop); mid-chunk `get_state` prefers the loop's own snapshot.
- `validate_chunk` uses the interpolator's PEAK-speed threshold (pi/2 factor)
  and also checks the angular cap; `frames_hz<=0` raises; acceleration fields
  documented as advisory (they were never enforced); known deferred item:
  `blocking=False` async execution remains a 0.2 candidate.

### Also in this release: post-0.1.0 hardening (conformance audit, batches 1-5)

From a conformance audit against the industry real-robot
stack (Polymetis / Deoxys / frankapy / SERL / LeRobot / ros2_control + the Flexiv
RDK). These changes are backward compatible; new safety features are opt-in.

#### Added
- Per-tick robot-fault gate: `RobotBackend.in_fault()` (RDK `robot.fault()`),
  surfaced as `FAULT`/`BACKEND_FAULT`; the control loop halts streaming on a fault.
- Safety filter: opt-in per-tick acceleration cap (`max_linear_accel`), the
  `STALE_STATE` state-age watchdog (`stop_on_state_stale`), bounded-hold
  escalation (`hold_timeout_ms`), and attributable clip reasons.
- `GripperCommand.from_signed_action` — the unified `[-1, 1]` gripper encoding.
- Mesh-free MuJoCo DLS-IK regression test, run in a new CI `sim` job.
- `py.typed` (PEP 561); single-sourced version; `CHANGELOG.md`; `SECURITY.md`;
  `CODE_OF_CONDUCT.md`; issue/PR templates.

#### Changed
- `FlexivRealEnv.reset()` re-enters Cartesian mode after homing (the first
  `step()` would otherwise be rejected on hardware).
- MuJoCo `control_dt` defaults to the control period so chunks play back at the
  right speed in sim; `move_gripper()` now steps the sim.
- RDK backend: gripper `Init()`/`moving()`, `ClearFault()` return checked, DoF
  adopted from the robot, configured `target_wrench` actually commanded.
- Honest labeling: `RemotePolicyClient` speaks flexiv_control's own JSON (not the
  openpi/pi0 wire format); rotation deltas named axis-angle (`drx/dry/drz`);
  "real-time chunking" relabeled "async infer-ahead".
- GitHub Actions bumped to Node-24 runtimes; PyPI publish is idempotent
  (`skip-existing`) and bound to a gated `pypi` environment.

## [0.1.0] - 2026-06-08

First public release.

### Added
- Unified action contract: `CartesianChunk` / `CartesianDelta` / `JointChunk` +
  `GripperCommand` and a quantified `ExecutionResult`.
- First-class safety: named, version-controlled `SafetyProfile`s and a
  microsecond per-tick `SafetyFilter` (workspace box, speed/jump caps, joint
  limits, contact-wrench stop, command watchdog).
- Backends behind one config switch: real Rizon (Flexiv RDK v1.x), a
  dependency-free fake, and MuJoCo (Jacobian DLS IK, gravity compensation, GN01
  gripper).
- Cross-process server + `RemoteRobot` with an in-process lease and a host-wide
  lock; single-writer, hold-on-stale `ReactiveServoLoop`.
- Optional C++ 1 kHz RT daemon and a ROS 2 overlay.
- Gymnasium env + HIL-SERL intervention wrapper; LeRobot adapter; SpaceMouse
  teleop; `RecedingHorizonRunner` + `RemotePolicyClient` (the policy-server seam).
- PyPI Trusted Publishing release workflow.

[Unreleased]: https://github.com/ZihaoLu001/flexiv_control/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/ZihaoLu001/flexiv_control/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/ZihaoLu001/flexiv_control/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/ZihaoLu001/flexiv_control/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ZihaoLu001/flexiv_control/releases/tag/v0.1.0
