"""Server-side single-writer streaming (start_servo_loop / servo_stream) and the
networked policy-server client (RemotePolicyClient) driving a receding horizon."""

from __future__ import annotations

import http.server
import json
import threading
import time

import numpy as np
import pytest

from flexiv_control import (
    CartesianTrajectory,
    RecedingHorizonRunner,
    RemotePolicyClient,
    Robot,
    RobotConfig,
    SafetyStatus,
)
from flexiv_control.client import RemoteRobot
from flexiv_control.client.remote_robot import RemoteRobotError
from flexiv_control.server import FlexivControlServer
from flexiv_control.server import protocol as P


def test_server_side_servo_loop_streams_holds_and_blocks():
    srv = FlexivControlServer(config=RobotConfig(backend="fake"), port=8811, lease_ttl=10.0)
    srv.serve_in_thread()
    time.sleep(0.3)
    try:
        rr = RemoteRobot("127.0.0.1", 8811, owner="t")
        rr.connect()
        rr.acquire_lease("t")
        rr.start_cartesian_impedance()

        rr.start_servo_loop(control_hz=200.0)
        s0 = rr.get_state()
        # Stream an UNREACHED, in-workspace target (x: 0.45 -> 0.65; box is
        # 0.25..0.75) only briefly, so the arm is still well short of it (the
        # filter caps each tick to max_linear_speed*dt) when we stop streaming.
        tgt = s0.tcp_pose.copy()
        tgt[0] += 0.20
        for _ in range(3):  # fire-and-forget; the loop is the single writer
            rr.servo_stream(tgt)
            time.sleep(0.02)
        assert rr.get_state().tcp_pose[0] - s0.tcp_pose[0] > 0.003  # the loop drove the arm

        # Stop streaming. Past command_timeout_ms (100 ms for tabletop_safe) the
        # watchdog must HOLD -- re-issue the measured pose, NOT keep tracking the
        # stale target toward `tgt`. SafetyStatus.HOLDING is set ONLY in that
        # stale branch, so this cannot pass if hold-on-stale regresses (unlike a
        # drift check against an already-reached target, which false-greens).
        time.sleep(0.4)
        p1 = rr.get_state()
        assert p1.safety_status == SafetyStatus.HOLDING
        time.sleep(0.25)
        p2 = rr.get_state()
        assert np.linalg.norm(p2.tcp_pose[:3] - p1.tcp_pose[:3]) < 0.005  # steady hold
        assert tgt[0] - p2.tcp_pose[0] > 0.02  # held well short of the unreached target

        # blocking motion RPCs are rejected by the single-writer guard specifically
        # (match the message so an unrelated failure can't keep this green).
        with pytest.raises(RemoteRobotError, match="servo loop active"):
            rr.execute_cartesian_trajectory(
                CartesianTrajectory.from_waypoint_array([[0.5, 0.0, 0.30, 1.0, 20]])
            )

        # stop the loop -> blocking motion works again
        rr.stop_servo_loop()
        p = rr.get_state().tcp_pose
        rr.execute_cartesian_trajectory(
            CartesianTrajectory.from_waypoint_array([[p[0], p[1], p[2], 1.0, 10]])
        )
        rr.disconnect()
    finally:
        srv.shutdown()


def test_servo_loop_blocks_every_backend_writer():
    """The single-writer invariant covers ALL backend writers -- not just the
    obvious motion RPCs but mode switches and gripper commands, which would
    otherwise switch the arm's mode or drive the gripper underneath the loop."""
    srv = FlexivControlServer(config=RobotConfig(backend="fake"), port=8813, lease_ttl=10.0)
    srv.serve_in_thread()
    time.sleep(0.3)
    try:
        rr = RemoteRobot("127.0.0.1", 8813, owner="t")
        rr.connect()
        rr.acquire_lease("t")
        rr.start_cartesian_impedance()
        rr.start_servo_loop(control_hz=200.0)

        # Every backend writer is refused while the loop owns the arm -- including
        # the mode switches and gripper command the first cut of _LOOP_BLOCKED missed.
        from flexiv_control import GripperCommand

        def _blocked(fn):
            with pytest.raises(RemoteRobotError, match="servo loop active"):
                fn()

        _blocked(lambda: rr.start_joint_impedance())
        _blocked(lambda: rr.start_cartesian_impedance())
        _blocked(lambda: rr.command_gripper(GripperCommand(width=0.02)))
        _blocked(lambda: rr.move_joint([0.0, -0.7, 0.0, 1.6, 0.0, 0.9, 0.0]))
        _blocked(lambda: rr.home())

        # ...but a non-writer (read-only) and the streaming verb still work.
        assert rr.get_state() is not None
        rr.servo_stream(rr.get_state().tcp_pose)

        # After stopping the loop, those same commands are accepted again.
        rr.stop_servo_loop()
        rr.command_gripper(GripperCommand(width=0.02))
        rr.start_joint_impedance()
        rr.disconnect()
    finally:
        srv.shutdown()


def test_remote_policy_client_drives_receding_horizon():
    calls = {"n": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(n))
            obs = body["observation"]
            calls["n"] += 1
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if calls["n"] > 3:  # signal "done"
                self.wfile.write(json.dumps({"traj": None}).encode())
                return
            x, y, z = obs["tcp_pose"][0], obs["tcp_pose"][1], obs["tcp_pose"][2]
            traj = CartesianTrajectory.from_pose_array(
                np.array([[x + 0.02, y, z, 1, 0, 0, 0, 1, 40]]),
            )
            self.wfile.write(json.dumps({"traj": P.trajectory_to_dict(traj)}).encode())

    httpd = http.server.HTTPServer(("127.0.0.1", 8822), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        r = Robot(RobotConfig(backend="fake"))
        r.connect()
        r.start_cartesian_impedance()
        s0 = r.get_state()
        client = RemotePolicyClient("http://127.0.0.1:8822/infer")
        n = RecedingHorizonRunner(r, client, max_steps=10).run()
        s1 = r.get_state()
        assert n == 3              # ran until the server returned traj=null
        assert calls["n"] == 4
        assert s1.tcp_pose[0] - s0.tcp_pose[0] > 0.04  # policy advanced the arm
        r.disconnect()
    finally:
        httpd.shutdown()
