# Bringing up a real Rizon

> Community project, **NOT affiliated with Flexiv Robotics**. Flexiv, Rizon, and
> RDK are trademarks of their owner. Use the official Flexiv documentation as
> the authority on your robot, firmware, and license; the notes here are how
> `flexiv_control` plugs into it.

The whole stack runs with **no hardware** on the `fake` backend (that is how the
examples and CI work). This page is the extra steps to drive a physical arm.

## Prerequisites

- A **Flexiv Rizon** arm whose firmware is compatible with the RDK version you
  install. *RDK and robot firmware are version-matched* — check Flexiv's
  compatibility table for your pair.
- The **Flexiv RDK** installed, including the Python module. The import name is
  `flexivrdk` (not `flexiv_rdk`):
  ```bash
  python -c "import flexivrdk; print('flexivrdk OK')"
  ```
  If that fails, install RDK per Flexiv's instructions (typically a wheel or a
  built Python binding) before continuing.
- A **license** appropriate to the modes you want (see below).
- A wired network connection to the robot, low latency, ideally a dedicated
  NIC. Note the robot's IP and your workstation's IP.
- An **E-stop within reach** and a validated safety profile (see
  [safety.md](safety.md)).

## Licenses: Standard vs Professional

This is the single most important planning decision.

- **Standard license → Tier A (Python, non-real-time modes).** The Python loop
  streams setpoints through RDK's NRT modes (`SendCartesianMotionForce`,
  `SendJointPosition`, …) at 100–500 Hz; the robot interpolates internally. No
  root, no real-time kernel, no C++. This is the default and is reactive enough
  for chunked policy execution, MPC, and RL.
- **Professional license → Tier B (C++ 1 kHz real-time streaming).** The
  real-time modes (`StreamCartesianMotionForce`, `StreamJointPosition`) and the
  `flexiv::rdk::Scheduler` need the Professional license, a `PREEMPT_RT` kernel,
  and root. Use it only when a measurement says Tier A's host rate is the
  bottleneck. See [`cpp/README.md`](../cpp/README.md).

Start on Tier A. You can upgrade later without changing any high-level code —
both tiers speak the same action contract and the same wire protocol.

## On the robot / pendant

1. Power up and clear any faults in **Flexiv Elements**.
2. Enable **Remote / RDK mode** so external control is permitted (the exact
   label depends on your software version).
3. Note the **serial number** (e.g. `Rizon4s-XXXXXX`) — you need it to connect.
4. If you have a gripper, note its model/name as RDK expects it.

## Configure `flexiv_control`

Copy the shipped template and edit it for your arm. `src/flexiv_control/configs/robots/rizon4s_lab.yaml`:

```yaml
robot_id: rizon4s_lab
backend: flexiv_rdk
robot_sn: "Rizon4s-XXXXXX"        # <-- your arm's serial number
gripper_name: Flexiv-GraspG2      # <-- your gripper name, or remove the line
n_joints: 7
control_hz: 100.0
default_safety_profile: tabletop_safe
q_home: [0.0, -0.7, 0.0, 1.6, 0.0, 0.9, 0.0]
```

The fields map directly to `RobotConfig`: `backend`, `robot_sn` (note: *sn*, not
`serial`), `gripper_name`, `control_hz`, `default_safety_profile`, `q_home`.

Then connect by config name (the shipped templates are searched under
`src/flexiv_control/configs/robots/`; set `FLEXIV_CONTROL_CONFIGS` to point at
your own config directory) or by path:

```python
from flexiv_control import Robot
robot = Robot.from_config("rizon4s_lab")     # or Robot.from_config("/path/to/your.yaml")
```

Point the `FLEXIV_CONTROL_CONFIGS` environment variable at your own config
directory to keep your real-robot YAMLs outside this repo.

## First-run checklist (do this slowly)

1. **Read state only** — no motion:
   ```bash
   flexiv-control state --config rizon4s_lab
   ```
   Confirm joint angles and TCP pose look sane.
2. **Validate the safety box.** Hand-guide or jog to the corners of your allowed
   region, read `robot.get_state().tcp_position`, and set the `workspace` bounds
   a few cm inside that. Lower `max_linear_speed` to ≤ 0.1 m/s for now.
   ([safety.md](safety.md))
3. **Home carefully**, ready on the E-stop:
   ```bash
   flexiv-control home --config rizon4s_lab
   ```
4. **One small motion.** Start Cartesian impedance and command a short, slow
   trajectory a few centimetres from the current pose; check `result.success`,
   `result.clipped`, and `result.path_tracking_error`.
5. **Then** raise speeds and hand control to your policy / MPC / RL env.

```python
import numpy as np
from flexiv_control import Robot, CartesianTrajectory, CartesianWaypoint

robot = Robot.from_config("rizon4s_lab")
robot.connect()
robot.start_cartesian_impedance()                    # NRT by default (Tier A)

start = robot.get_state().tcp_position
near = start + np.array([0.0, 0.0, -0.03])           # 3 cm down
trajectory = CartesianTrajectory(
    waypoints=[CartesianWaypoint(position=near, duration=2.0)]  # slow: 2 s
)
result = robot.execute_cartesian_trajectory(trajectory)
print(result.success, result.clipped, result.path_tracking_error)

robot.stop()
robot.disconnect()
```

## Networked / cross-machine use

If your policy, MPC, or RL trainer runs on a different machine (common — the GPU
box is rarely the robot box), run the server next to the robot and drive it from
anywhere with `RemoteRobot`:

```bash
# on the robot's workstation
flexiv-control serve --config rizon4s_lab --host 0.0.0.0 --port 8766
```

```python
# on the GPU/dev machine
from flexiv_control import RemoteRobot
with RemoteRobot("ROBOT_HOST_IP", 8766, owner="trainer") as r:
    # __enter__ already connected AND acquired the lease -- no second acquire.
    r.start_cartesian_impedance()
    ...

# without a with-block (e.g. the robot is constructed conditionally):
r = RemoteRobot("ROBOT_HOST_IP", 8766, owner="trainer")
r.connect()
r.acquire_lease()
try:
    ...
finally:
    r.close()   # releases the lease and closes the socket
```

The client `pip install`s with nothing but numpy and needs no ROS. See
[architecture.md](architecture.md) for the server/lease design and
[versions.md](versions.md) for the RDK-version cautions you should verify
against your install.
