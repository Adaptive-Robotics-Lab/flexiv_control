"""``flexiv-control`` command-line entry point.

Subcommands::

    flexiv-control serve  --backend fake --port 8766     # run the control server
    flexiv-control home   --backend fake                 # send the arm home
    flexiv-control state  --backend fake                 # print one RobotState
    flexiv-control demo                                  # offline FakeBackend demo
    flexiv-control viz    --connect <robot-pc>           # live browser mirror
                                                         # (pip install flexiv-control[viz])

Nothing here needs hardware unless you pass ``--backend flexiv_rdk``.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import __version__
from .config import RobotConfig
from .robot import Robot


def _robot(args) -> Robot:
    cfg = (
        RobotConfig.load(args.config)
        if getattr(args, "config", None)
        else RobotConfig(backend=args.backend, robot_sn=getattr(args, "robot_sn", "") or "")
    )
    if getattr(args, "backend", None):
        cfg.backend = args.backend
    return Robot(cfg)


def _cmd_serve(args) -> int:
    from .server.server import FlexivControlServer

    server = FlexivControlServer(
        robot=_robot(args), host=args.host, port=args.port, lease_ttl=args.lease_ttl
    )
    prof = server.robot.profile
    print(f"[flexiv-control] serving backend={server.robot.cfg.backend!r} "
          f"on {args.host}:{args.port} (ctrl-c to stop)")
    # Print the active envelope so the session log PROVES which profile ran --
    # which workspace box and speed cap bind is otherwise invisible out-of-band
    # state that consumers have guessed wrong.
    print(f"[flexiv-control] safety profile {prof.name!r}: "
          f"x{list(prof.ws_x)} y{list(prof.ws_y)} z{list(prof.ws_z)} "
          f"({prof.workspace_action} on violation), "
          f"caps {prof.max_linear_speed} m/s / {prof.max_angular_speed} rad/s")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[flexiv-control] shutting down")
    return 0


def _cmd_home(args) -> int:
    robot = _robot(args)
    with robot:
        robot.acquire_lease("cli")
        robot.home()
        print("[flexiv-control] homed:", np.round(robot.get_state().q, 3))
    return 0


def _cmd_zero_ft(args) -> int:
    """Zero the 6-DoF F/T sensor (fixes Flexiv event 301004 after power-on).

    A manual zero in Elements' MANUAL mode does NOT carry into remote/RDK mode;
    force control faults until ZeroFTSensor runs in the remote session. With
    --connect HOST[:PORT] this runs on a RUNNING 'flexiv-control serve' via its
    live RDK session (no restart). Otherwise it opens a direct backend (no server
    may be holding the robot). Robot must be Auto(Remote)+Enabled, arm at rest,
    NO external contact."""
    if getattr(args, "connect", None):
        host, _, port = args.connect.partition(":")
        from .client.remote_robot import RemoteRobot
        from .server import protocol as P

        robot = RemoteRobot(host, port=int(port) if port else P.DEFAULT_PORT)
        robot.connect()
        robot.acquire_lease("cli-zero-ft")
        try:
            robot.zero_ft_sensor()
            print(f"[flexiv-control] F/T sensor zeroed via {host} (running server, no restart)")
        finally:
            robot.release_lease()
            robot.disconnect()
        return 0
    robot = _robot(args)
    with robot:
        robot.acquire_lease("cli-zero-ft")
        robot.zero_ft_sensor()
        print(f"[flexiv-control] F/T sensor zeroed (backend={robot.cfg.backend!r})")
    return 0


def _cmd_state(args) -> int:
    robot = _robot(args)
    with robot:
        s = robot.get_state()
        print("mode        :", s.control_mode.value)
        print("q           :", np.round(s.q, 3))
        print("tcp_pose    :", np.round(s.tcp_pose, 4))
        print("wrench      :", np.round(s.wrench, 2))
        print("gripper     :", round(s.gripper_width, 4))
    return 0


def _cmd_demo(args) -> int:  # noqa: ARG001
    from .trajectory import CartesianTrajectory

    robot = Robot(RobotConfig(backend="fake"))
    with robot:
        robot.acquire_lease("demo")
        robot.start_cartesian_impedance()
        u = [[0.45, 0.0, 0.30, 1.0, 20],
             [0.50, 0.05, 0.25, 0.0, 20],
             [0.45, 0.0, 0.30, 1.0, 20]]
        result = robot.execute_cartesian_trajectory(CartesianTrajectory.from_waypoint_array(u))
        print("[flexiv-control] demo traj executed:")
        print("  success            :", result.success)
        print("  clipped            :", result.clipped)
        print("  path_tracking_error:", round(result.path_tracking_error, 5))
        print("  final tcp position :", np.round(result.final_state.tcp_position, 4))
    return 0


def _cmd_viz(args) -> int:
    """Live browser mirror of a running robot/server (safety + debugging).

    Connects READ-ONLY: a monitor must never own the arm, so this uses a bare
    ``RemoteRobot.connect()`` (no lease) -- ``get_state``/``get_safety_profile``
    are lease-free by design and are served from per-tick snapshots, so the
    mirror never blocks a running traj."""
    try:
        from .viz import RobotViz
        from .viz import assets as viz_assets
    except ImportError as e:
        print(f'[flexiv-control] the viz extra is not installed: {e}\n'
              '  pip install "flexiv-control[viz]"')
        return 1

    model = args.model
    if model is None and not args.no_model:
        model = viz_assets.ensure_rizon_urdf(download=args.fetch_assets)
        if model is None:
            print("[flexiv-control] no Rizon URDF found -- running in frames mode "
                  "(TCP frame + trail + planned path; fully functional).\n"
                  "  For the mesh mirror: set FLEXIV_DESCRIPTION_DIR to a checkout of\n"
                  "  github.com/flexivrobotics/flexiv_description (branch humble-v1),\n"
                  "  or re-run with --fetch-assets to download it into ~/.cache.")

    if args.connect:
        host, _, port = args.connect.partition(":")
        from .client.remote_robot import RemoteRobot
        from .server import protocol as P

        robot = RemoteRobot(host, port=int(port) if port else P.DEFAULT_PORT)
        robot.connect()  # NO lease: monitoring must not own the arm
        print(f"[flexiv-control] mirroring {host} (read-only, no lease)")
    else:
        robot = _robot(args)
        robot.connect()
        print(f"[flexiv-control] mirroring a local {robot.cfg.backend!r} backend")

    viz = RobotViz(model=model, port=args.viz_port, state_hz=args.hz)
    viz.attach(robot)
    print(f"[flexiv-control] open {viz.url} in a browser (ctrl-c to stop)")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[flexiv-control] stopping viz")
        viz.stop()
        robot.disconnect()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="flexiv-control", description=__doc__)
    p.add_argument("--version", action="version", version=f"flexiv-control {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend", default="fake",
                        help="fake | flexiv_rdk | mujoco (default: fake)")
    common.add_argument("--config", default=None, help="path or name of a robot config YAML")
    common.add_argument("--robot-sn", dest="robot_sn", default="",
                        help="Flexiv robot serial number (flexiv_rdk backend)")

    s = sub.add_parser("serve", parents=[common], help="run the control server")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8766)
    s.add_argument("--lease-ttl", type=float, default=2.0)
    s.set_defaults(func=_cmd_serve)

    h = sub.add_parser("home", parents=[common], help="send the arm to its home posture")
    h.set_defaults(func=_cmd_home)

    z = sub.add_parser("zero-ft", parents=[common],
                       help="zero the 6-DoF F/T sensor (fix event 301004 after power-on)")
    z.add_argument("--connect", default=None, metavar="HOST[:PORT]",
                   help="run on a RUNNING 'flexiv-control serve' via its live RDK session "
                        "(no restart); omit to open a direct backend (no server running)")
    z.set_defaults(func=_cmd_zero_ft)

    st = sub.add_parser("state", parents=[common], help="print one RobotState and exit")
    st.set_defaults(func=_cmd_state)

    d = sub.add_parser("demo", help="offline FakeBackend demo (no hardware)")
    d.set_defaults(func=_cmd_demo)

    v = sub.add_parser(
        "viz", parents=[common],
        help="live browser mirror + intended-motion preview (flexiv-control[viz])",
    )
    v.add_argument("--connect", default=None, metavar="HOST[:PORT]",
                   help="mirror a running 'flexiv-control serve' (read-only, no lease); "
                        "omit to mirror a local backend (e.g. --backend fake)")
    v.add_argument("--model", default=None, help="path to a Rizon URDF for the mesh mirror")
    v.add_argument("--no-model", action="store_true",
                   help="skip URDF resolution; frames mode only")
    v.add_argument("--fetch-assets", action="store_true",
                   help="allow downloading flexiv_description into ~/.cache for the mesh mirror")
    v.add_argument("--hz", type=float, default=20.0, help="state poll rate (default 20)")
    v.add_argument("--viz-port", type=int, default=8080,
                   help="port for the browser page (default 8080)")
    v.set_defaults(func=_cmd_viz)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
