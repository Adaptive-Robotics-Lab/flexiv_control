# Integration: receding-horizon & action-chunking planners

> Community project, **NOT affiliated with Flexiv Robotics**.

Many modern manipulation systems are **receding-horizon planners**: at each step
they perceive the scene, sample or optimize a short sequence of future actions (a
"trajectory"), pick the best one, execute only the **first** segment, then replan from
the new state. Action-chunking policies (ACT-style) and sampling/optimization
planners (CEM, MPPI, MPC) all fit this mold. `flexiv_control` is built to be the
execution-and-safety layer underneath such a loop, with three things aligned on
purpose: the action shape, the sim/real path, and a quantified "execution"
failure signal.

## 1. The action shape

These planners typically emit a candidate action of the form
`u = ((x_j, y_j, z_j, w_j, n_j))_{j=1..H}` — a short sequence of Cartesian
positions, each with a normalised gripper command `w_j ∈ [0,1]` (1 = open,
0 = closed) and an integer number of low-level control frames `n_j`. That is
exactly what `CartesianTrajectory.from_waypoint_array(u)` ingests:

```python
from flexiv_control import Robot, CartesianTrajectory

robot = Robot.from_config("rizon4s_lab")    # "fake" while developing offline
robot.connect()
robot.start_cartesian_impedance()

u = planner.best_chunk()                     # (H, 5): (x, y, z, w, n)
# Naming a safety_profile pins the envelope the trajectory was planned for; it is
# VERIFIED against the robot's active profile (a mismatch raises). Leave it ""
# (the default) to run under whatever profile is active.
trajectory = CartesianTrajectory.from_waypoint_array(u, safety_profile="tabletop_safe")
result = robot.execute_cartesian_trajectory(trajectory)
```

`n_j` becomes a duration at the active control rate (`duration = n_j /
control_hz`), so the same `u` runs identically whether you execute at 100 Hz
(Tier A) or 1 kHz (Tier B). `from_waypoint_array` is position-only, so
orientation is always held; if your planner emits orientation, build full-SE(3)
`CartesianWaypoint`s with quaternions and pass them to `CartesianTrajectory(...)`
directly. See [action_contract.md](action_contract.md).

## 2. Receding horizon: execute the first segment, then replan

Commit only the first segment of the chosen trajectory and replan from the new state.
Two clean ways to express "first segment only":

- **Build a one-waypoint trajectory** from the first row of `u` and execute it
  (simplest, fully blocking):
  ```python
  first = CartesianTrajectory.from_waypoint_array(u[:1])
  result = robot.execute_cartesian_trajectory(first)
  state = robot.get_state()        # replan from here
  ```
- **Or run a fixed wall-clock slice** and re-issue. Because `execute_*` is
  blocking and returns a populated `ExecutionResult`, a synchronous
  perceive → plan → execute-first → repeat loop needs no callback plumbing.

```python
while not done:
    obs   = perceive(camera)                  # build a local scene / features
    u     = planner.plan_and_select(obs)      # (H, 5)
    res   = robot.execute_cartesian_trajectory(CartesianTrajectory.from_waypoint_array(u[:1]))
    log_execution(res)                         # see section 4
    done  = goal_reached(robot.get_state())
```

If perception/solve runs on a different machine than the arm, swap `Robot` for
`RemoteRobot` with no other change ([flexiv_setup.md](flexiv_setup.md)). For a
high-rate streaming variant (commit a setpoint every tick instead of
per-segment) see [integration_mpc.md](integration_mpc.md) and
`flexiv_control.server.ReactiveServoLoop`.

## 3. real2sim2real: one contract for sim and real

The same trajectory drives the `mujoco` backend (sim) and a real Rizon:

- **Plan in sim, execute on real** — a planner's internal rollouts and the real
  execution speak the identical contract, so there is no separate "sim action
  format" vs "real action format" to keep in sync.
- **Develop offline on `fake`** — the dependency-free backend runs the whole
  loop (plan → trajectory → execution report) with no hardware and no MuJoCo, useful
  for planner/integration tests in CI.
- Point the env/robot at `backend="mujoco"` for sim, `"fake"` for offline,
  `"flexiv_rdk"` for the arm — the planner code does not change.

> The shipped `MujocoBackend` is a thin stub/seam; wire it to *your* MuJoCo scene
> and IK / operational-space mapping (it is intentionally left as the integration
> point rather than guessing your model). The contract and safety layers around
> it are complete.

## 4. Close the "execution" failure bucket with evidence

Receding-horizon planners usually have an "execution" entry in their failure
taxonomy. Today that is often a guess; `ExecutionResult` makes it measurable, so
a bad outcome can be attributed to execution vs perception vs ranking with data:

```python
res = robot.execute_cartesian_trajectory(trajectory)
record = {
    "success": res.success,
    "clipped": res.clipped,                       # safety filter modified a setpoint
    "stop_reason": res.stop_reason,               # workspace_limit / contact_wrench / ...
    "path_tracking_error": res.path_tracking_error,  # cmd vs measured drift (m)
    "max_tcp_speed": res.max_tcp_speed,
    "max_wrench": res.max_wrench,
}
```

Useful signals:

- `clipped == True` or `stop_reason == "workspace_limit"` → the sampled trajectory
  left the safe set; bias sampling inward or widen the validated workspace.
- `stop_reason == "contact_wrench"` → unexpected contact; an execution failure,
  not a perception miss.
- large `path_tracking_error` with `success == True` → the arm could not track
  the commanded speed; lower `max_tcp_linear_speed` on the trajectory or slow `n_j`.

Logging these per executed segment gives a clean per-rollout label separating
"the plan was bad" from "the plan was fine but execution clipped/stopped."

## Recommended defaults

- **Tier A (Python NRT) to start.** Reactive enough for trajectory execution; no
  real-time kernel or Professional license. Move to Tier B only if a high-rate
  streaming variant is genuinely host-rate-limited.
- **Safety profile**: `tabletop_safe` for general tabletop work;
  `contact_manipulation` for deliberate-contact subtasks. Validate the workspace
  box on your cell first ([safety.md](safety.md)).
- **Pin** `flexiv-control==0.1.0` in your planner's repo so the planner and the
  controller version together ([versions.md](versions.md)).
