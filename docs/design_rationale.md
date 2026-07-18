# Design rationale

> Community project, **NOT affiliated with Flexiv Robotics**.

This document records *why* `flexiv_control` is shaped the way it is: the field
survey it draws on, the decisions that follow, and an explicit, critical look at
an alternative "ROS-2-first" proposal.

## What the field converged on

We surveyed the controllers that the manipulation-learning and real-robot-RL
communities actually use. They disagree on details but agree on the skeleton:

- **Deoxys** (UT-Austin RPL) — a C++ real-time controller on a NUC plus a Python
  client; operational-space / impedance control; the same code path from
  `robosuite` sim to real. The de-facto standard for imitation learning on
  Franka.
- **Polymetis** (Meta FAIR) — a C++ gRPC server at 1 kHz with a Python client;
  PyBullet↔real behind config; explicitly recommends running the client on a
  **separate machine** from the real-time loop. (Effectively unmaintained now —
  borrow the architecture, not the code.)
- **frankapy / Franka-Interface** (CMU) — C++ real-time control at 1 kHz, a
  "skills" abstraction with termination handlers, shared-memory to Python.
- **SERL / HIL-SERL** (Berkeley) — a robot **server** (HTTP) plus a thin
  **Gymnasium env client**; async actor/learner; the **SpaceMouse doubles as the
  RL intervention device**. The standard for sample-efficient real-robot RL; now
  upstreamed into LeRobot.
- **LeRobot** (Hugging Face) — one `Robot` interface (`connect`,
  `get_observation`, `send_action`, `observation_features`/`action_features`).
  "Bring your own hardware" and you plug into LeRobot's data-collection,
  training, and visualization tooling; emitting a `LeRobotDataset` from the
  recorded frames is a thin caller-side step (its dataset API moves between
  releases).
- **SAIL** (arXiv 2506.11948) — a 4-level Franka hierarchy: policy trajectory → NUC
  interpolation at 100 Hz → OSC → torque at 500 Hz — a clean illustration of the
  "trajectory on top, fast servo underneath" pattern.

**The common skeleton:** a host-side real-time (or near-real-time) control loop,
a thin Python client, **one** action interface shared by every consumer, and
**sim and real behind a single backend switch**. `flexiv_control` is that
skeleton, specialized to the Rizon.

## Decisions that follow

1. **The action contract is the product.** Everything else is replaceable. We
   make `CartesianTrajectory` (+ `from_waypoint_array`), `CartesianDelta`, `JointTrajectory`,
   `GripperCommand`, and `ExecutionResult` the stable spine, so a policy, an MPC,
   an RL env, and a teleop pendant all speak one language. See
   [action_contract.md](action_contract.md).

2. **A standalone repo that is *also* pip-installable.** Mirrors Deoxys /
   Polymetis / SERL / LeRobot. `git clone` it as a peer of your project, or
   `pip install -e .` into any environment. The repo *is* the package.

3. **Server + thin client, ROS optional.** An RL or MPC researcher should be
   able to talk to the arm with a `pip install` and a socket — no ROS workspace,
   no message broker. So the core ships a dependency-free newline-JSON server and
   a `RemoteRobot` client. ROS 2 is a genuinely optional overlay for teams who
   already live there.

4. **Two RT tiers (see [architecture.md](architecture.md)).** Tier A is Python
   over the RDK's NRT modes (ships first, no root, Standard license). Tier B is
   the C++ 1 kHz `Scheduler` daemon (optional upgrade, real-time kernel, Pro
   license). Identical contract, server API, client, and Gym env across both.

5. **Safety is first-class and reported.** A shared lab can't run on "everyone
   be careful." Every setpoint passes a cheap per-tick filter; profiles are named
   YAML so a run is reproducible ("this used `tabletop_safe`"); and
   `ExecutionResult` surfaces tracking error / clipping / stop reason — exactly
   the signal a planner's "execution" failure category needs.

6. **A LeRobot adapter.** The single biggest community lever: implement
   LeRobot's `Robot` interface and Rizon users inherit LeRobot's data collection,
   training, and visualization for free.

## A critical look at the "ROS-2-first" proposal

A third-party analysis recommended a **ROS-2-first** `flexiv-control-stack`. It
got real things right and some things wrong. Treat it critically:

**Where it was right.**
- It correctly identified the current teleop stack: **MoveIt Servo** publishing
  Cartesian twists into **`flexiv_ros2` / ros2_control** on top of the RDK. That
  matches the repo. We keep a MoveIt-Servo-compatible jog input in the ROS
  overlay precisely so the existing demo keeps working.
- It was right that **safety, a single action interface, and sim/real parity**
  are the load-bearing concerns.

**Where it over-reached.**
- **"ROS 2 first" is the wrong default for the actual users.** RL and MPC
  researchers — the people this controller is for — generally do **not** want to
  stand up a ROS 2 workspace to call a policy. Making ROS the front door adds
  friction for the majority to serve a minority. We invert it: **Python core
  first, ROS optional.** Teams already in ROS lose nothing; everyone else gains a
  one-`pip`-install path.
- **"Python can't do 1 kHz, so it must be C++/ROS" is only half true.** Because
  the Rizon closes the hard real-time loop **internally**, a Python host loop at
  100–500 Hz over the NRT modes is genuinely reactive enough for chunked policy
  execution, MPC, and RL. The C++ 1 kHz path is a real *upgrade* for specific
  needs — not a prerequisite. Conflating "needs 1 kHz host control" with "needs
  ROS" is a category error: the 1 kHz loop is an RDK `Scheduler` C++ program; ROS
  is an orthogonal integration choice.
- **A heavyweight stack raises the reuse cost.** The goal is a controller a lab
  mate or a stranger can adopt in an afternoon. A ROS-2-first monolith fights
  that.

Net: we **keep** the analysis's correct emphases (safety, one interface,
sim/real parity, MoveIt-Servo compatibility) and **reject** its default
(ROS-first, C++/ROS-mandatory), replacing it with a Python-first, ROS-optional,
two-tier design.

## Non-goals (for now)

- A motion planner. Use MoveIt for planning; this layer executes and guards.
- A full dynamics simulator in `FakeBackend` — it is a kinematic stand-in for
  development and CI. Real dynamics belong in the MuJoCo backend.
- Torque-level control from Python — that is Tier-B C++ territory.
