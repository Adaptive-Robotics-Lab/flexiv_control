"""RemoteRobot -- the Robot API, over the network.

Same methods as :class:`flexiv_control.Robot`, but every call is an RPC to a
:class:`~flexiv_control.server.server.FlexivControlServer`. This is the seam
that lets an RL trainer or MPC optimizer run on a different machine from the one
holding the arm (Polymetis recommends exactly this), while the controller code
on both sides is identical to the in-process case::

    from flexiv_control import RemoteRobot, CartesianTrajectory

    with RemoteRobot("192.168.2.100", owner="mpc") as robot:
        robot.start_cartesian_impedance()
        while running:
            s = robot.get_state()
            robot.servo_cartesian_delta(policy(s), duration=0.05)

The client holds the lease for ``owner`` and refreshes it with a background
heartbeat, so if this process dies the arm frees itself for the next user.
Only numpy is required.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Optional

import numpy as np

from ..trajectory import CartesianTrajectory, ExecutionResult, JointTrajectory
from ..types import GripperCommand, RobotState
from ..server import protocol as P


class RemoteRobotError(RuntimeError):
    pass


class RemoteRobot:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = P.DEFAULT_PORT,
        owner: str = "remote",
        *,
        heartbeat_period: float = 0.5,
        heartbeat_max_failures: int = 3,
        timeout: float = 10.0,
        motion_timeout: float = 600.0,
    ):
        self.host = host
        self.port = port
        self.owner = owner
        self.heartbeat_period = heartbeat_period
        self.heartbeat_max_failures = heartbeat_max_failures
        self.timeout = timeout
        # Blocking motion RPCs (execute_*/move_*/home/go_home_safe and a
        # wait=True gripper command) legitimately run for many seconds -- traj
        # segments are TIME-STRETCHED to honor speed caps, and a home ritual
        # bundles several stages into one RPC. Reading those with the short
        # default timeout used to raise client-side WHILE THE ARM KEPT MOVING
        # and then desynchronize every later response on the socket.
        self.motion_timeout = motion_timeout

        self._sock: Optional[socket.socket] = None
        self._rfile = None
        self._wfile = None
        self._io_lock = threading.Lock()
        self._next_id = 0
        self._hb_thread: Optional[threading.Thread] = None
        self._hb_running = False
        self._has_lease = False

    # -- connection ----------------------------------------------------------
    def connect(self) -> "RemoteRobot":
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._rfile = self._sock.makefile("rb")
        self._wfile = self._sock.makefile("wb")
        self._call("ping")
        return self

    def close(self) -> None:
        self._stop_heartbeat()
        try:
            if self._has_lease:
                self.release_lease()
        except Exception:
            pass
        for f in (self._rfile, self._wfile):
            try:
                if f:
                    f.close()
            except Exception:
                pass
        try:
            if self._sock:
                self._sock.close()
        finally:
            self._sock = None

    def disconnect(self) -> None:
        """Alias for :meth:`close`; matches ``Robot.disconnect()`` so callers are
        API-identical whether they hold a Robot or a RemoteRobot (e.g. the Gym env
        and LeRobot adapter call ``robot.disconnect()``)."""
        self.close()

    def __enter__(self) -> "RemoteRobot":
        self.connect()
        self.acquire_lease()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- low-level RPC -------------------------------------------------------
    # Methods whose server-side handler can legitimately block for the duration
    # of a real robot motion: read them with `motion_timeout` instead of the
    # short default.
    _MOTION_METHODS = frozenset({
        "execute_cartesian_trajectory", "execute_joint_trajectory", "move_joint",
        "servo_cartesian_delta", "servo_cartesian_pose",
        "command_gripper", "home", "go_home_safe", "zero_ft_sensor",
    })

    def _call(self, method: str, **params) -> dict:
        if self._wfile is None or self._rfile is None:
            raise RemoteRobotError("not connected; call connect() first")
        with self._io_lock:
            self._next_id += 1
            rid = self._next_id
            if method in self._MOTION_METHODS and self._sock is not None:
                self._sock.settimeout(self.motion_timeout)
            try:
                self._wfile.write(P.dumps({"id": rid, "method": method, "params": params}))
                self._wfile.flush()
                line = self._rfile.readline()
            finally:
                if method in self._MOTION_METHODS and self._sock is not None:
                    self._sock.settimeout(self.timeout)
            if not line:
                raise RemoteRobotError("server closed the connection")
            resp = P.loads(line)
            # A response that does not match our request id means the stream is
            # desynchronized (e.g. an earlier read timed out and its reply is
            # still queued). Mis-attributing results on a robot-control channel
            # is far worse than failing: refuse the connection.
            if resp.get("id") != rid:
                raise RemoteRobotError(
                    f"response id {resp.get('id')!r} does not match request id {rid} "
                    f"({method}); the connection is desynchronized -- reconnect"
                )
        if not resp.get("ok", False):
            raise RemoteRobotError(resp.get("error", "unknown server error"))
        return resp.get("result", {})

    # -- lease ---------------------------------------------------------------
    def acquire_lease(self, owner: Optional[str] = None, *, force: bool = False) -> None:
        # Accept a positional owner so robot.acquire_lease("mpc") works identically
        # on Robot and RemoteRobot (no more TypeError shims at the call sites).
        if owner is not None:
            self.owner = owner
        self._call("acquire_lease", owner=self.owner, force=force)
        self._has_lease = True
        self._start_heartbeat()

    def release_lease(self) -> None:
        self._stop_heartbeat()
        self._call("release_lease", owner=self.owner)
        self._has_lease = False

    def _start_heartbeat(self) -> None:
        if self._hb_running:
            return
        self._hb_running = True
        self._hb_thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._hb_thread.start()

    def _stop_heartbeat(self) -> None:
        self._hb_running = False
        if self._hb_thread is not None:
            self._hb_thread.join(timeout=1.0)
            self._hb_thread = None

    def _heartbeat(self) -> None:
        # Tolerate transient hiccups: a single slow/dropped heartbeat must not
        # silently let the lease expire under a live client. Give up only after
        # several consecutive failures, or if the server actively rejects us
        # (lease lost) AND one re-acquire attempt fails -- a lease that merely
        # expired (e.g. heartbeats starved behind a long blocking RPC on this
        # shared socket) is recoverable, a lease taken by another owner is not.
        consecutive_failures = 0
        while self._hb_running:
            time.sleep(self.heartbeat_period)
            if not self._hb_running:
                break
            try:
                self._call("heartbeat", owner=self.owner)
                consecutive_failures = 0
            except RemoteRobotError:
                # Lease lost. If it simply expired, nobody holds it and a plain
                # re-acquire succeeds; if another owner took it, stop for good.
                try:
                    self._call("acquire_lease", owner=self.owner, force=False)
                    consecutive_failures = 0
                except Exception:
                    self._has_lease = False
                    self._hb_running = False
                    break
            except Exception:
                # Transient socket/timeout hiccup: tolerate a few, then give up.
                consecutive_failures += 1
                if consecutive_failures >= self.heartbeat_max_failures:
                    self._has_lease = False
                    self._hb_running = False
                    break

    # -- mirror of the Robot API --------------------------------------------
    def get_server_info(self) -> dict:
        """Return the remote package/protocol identity without taking a lease."""
        info = self._call("get_server_info")
        if not isinstance(info, dict):
            raise RemoteRobotError(
                "get_server_info returned a non-object response"
            )
        return info

    def set_safety_profile(self, name: str) -> None:
        self._call("set_safety_profile", owner=self.owner, name=name)

    def get_safety_profile(self):
        """Fetch the server's ACTIVE safety profile (workspace box, speed caps,
        ...). This is the single source of truth a client should prevalidate
        trajs against (``profile.validate_trajectory(traj)``) instead of keeping
        its own workspace constants that silently drift from the server's."""
        from ..safety import SafetyProfile

        return SafetyProfile.from_dict(self._call("get_safety_profile")["profile"])

    def get_state(self) -> RobotState:
        return P.state_from_dict(self._call("get_state")["state"])

    def start_cartesian_impedance(
        self,
        *,
        realtime: bool = False,
        stiffness: Optional[np.ndarray] = None,
        damping_ratio: Optional[np.ndarray] = None,
        nullspace_q: Optional[np.ndarray] = None,
    ) -> None:
        params = {"owner": self.owner, "realtime": realtime}
        if stiffness is not None:
            params["stiffness"] = np.asarray(stiffness, float).tolist()
        if damping_ratio is not None:
            params["damping_ratio"] = np.asarray(damping_ratio, float).tolist()
        if nullspace_q is not None:
            params["nullspace_q"] = np.asarray(nullspace_q, float).tolist()
        self._call("start_cartesian_impedance", **params)

    def start_joint_impedance(self, *, realtime: bool = False) -> None:
        self._call("start_joint_impedance", owner=self.owner, realtime=realtime)

    def servo_cartesian_delta(
        self,
        delta,
        *,
        duration: Optional[float] = None,
        frame: str = "base",
        gripper: Optional[GripperCommand] = None,
    ) -> ExecutionResult:
        r = self._call(
            "servo_cartesian_delta",
            owner=self.owner,
            delta=np.asarray(delta, float).tolist(),
            duration=duration,
            frame=frame,
            gripper=P.gripper_to_dict(gripper),
        )
        return P.result_from_dict(r["result"])

    def servo_cartesian_pose(
        self, pose, *, duration: float = 0.2, gripper: Optional[GripperCommand] = None
    ) -> ExecutionResult:
        r = self._call(
            "servo_cartesian_pose",
            owner=self.owner,
            pose=np.asarray(pose, float).tolist(),
            duration=duration,
            gripper=P.gripper_to_dict(gripper),
        )
        return P.result_from_dict(r["result"])

    def execute_cartesian_trajectory(
        self, traj: CartesianTrajectory, *, blocking: bool = True, raise_on_stop: bool = False
    ) -> ExecutionResult:
        # `blocking` is accepted for signature parity with Robot; the server always
        # executes synchronously, so it is a no-op on the wire. THIS connection is
        # therefore busy until the traj returns -- to abort a traj in flight,
        # call ``stop()`` from a separate client/connection (the server's stop
        # handler needs no lease and cancels the executing loop within one tick).
        r = self._call(
            "execute_cartesian_trajectory", owner=self.owner, traj=P.trajectory_to_dict(traj)
        )
        result = P.result_from_dict(r["result"])
        if raise_on_stop and not result.success:
            from ..robot import TrajectoryStoppedError

            raise TrajectoryStoppedError(result)
        return result

    def execute_joint_trajectory(
        self, traj: JointTrajectory, *, raise_on_stop: bool = False
    ) -> ExecutionResult:
        r = self._call(
            "execute_joint_trajectory", owner=self.owner, traj=P.joint_trajectory_to_dict(traj)
        )
        result = P.result_from_dict(r["result"])
        if raise_on_stop and not result.success:
            from ..robot import TrajectoryStoppedError

            raise TrajectoryStoppedError(result)
        return result

    def move_joint(
        self,
        q_target,
        *,
        duration: Optional[float] = None,
        max_joint_speed: Optional[float] = None,
        realtime: bool = False,
    ) -> ExecutionResult:
        """Interpolated joint move; give a fixed ``duration`` (s) OR a
        ``max_joint_speed`` (rad/s) cap on the largest joint excursion."""
        r = self._call(
            "move_joint",
            owner=self.owner,
            q=np.asarray(q_target, float).tolist(),
            duration=duration,
            max_joint_speed=max_joint_speed,
            realtime=realtime,
        )
        return P.result_from_dict(r["result"])

    def command_gripper(
        self, cmd: GripperCommand, *, wait: bool = False, timeout: float = 5.0
    ) -> Optional[float]:
        """Send a gripper command; ``wait=True`` blocks until the fingers settle
        and returns the final width (else ``None``)."""
        r = self._call(
            "command_gripper",
            owner=self.owner,
            gripper=P.gripper_to_dict(cmd),
            wait=wait,
            timeout=timeout,
        )
        return r.get("final_width")

    def home(
        self,
        q=None,
        *,
        max_joint_speed: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> None:
        """Move to the canonical posture: ``q`` if given, else the SERVER
        config's ``q_home`` (plus its ``gripper_home_width`` if configured)."""
        self._call(
            "home",
            owner=self.owner,
            q=None if q is None else np.asarray(q, float).tolist(),
            max_joint_speed=max_joint_speed,
            duration=duration,
        )

    def zero_ft_sensor(self) -> None:
        """Zero the robot's 6-DoF F/T sensor on the SERVER's live RDK session
        (clears event 301004 without restarting the server). Needs the lease;
        run with the arm at rest and no external contact."""
        self._call("zero_ft_sensor", owner=self.owner)

    def go_home_safe(
        self,
        *,
        q_home=None,
        lift_m: float = 0.10,
        open_gripper_width: Optional[float] = None,
        max_tcp_speed: float = 0.10,
        max_joint_speed: float = 0.3,
    ) -> ExecutionResult:
        """End-of-session exit ritual: lift straight up, open the gripper (and
        wait), then joint-move home -- each stage tolerating the previous one's
        failure. Put THIS in the ``finally`` block instead of hand-rolling it."""
        r = self._call(
            "go_home_safe",
            owner=self.owner,
            q_home=None if q_home is None else np.asarray(q_home, float).tolist(),
            lift_m=lift_m,
            open_gripper_width=open_gripper_width,
            max_tcp_speed=max_tcp_speed,
            max_joint_speed=max_joint_speed,
        )
        return P.result_from_dict(r["result"])

    def stop(self) -> None:
        self._call("stop", owner=self.owner)

    # -- single-writer streaming session (server-side hold-on-stale) --------
    def start_servo_loop(self, *, control_hz: Optional[float] = None) -> None:
        """Start a server-side single-writer ReactiveServoLoop. While it runs,
        ``servo_stream`` publishes the latest target fire-and-forget and a stall
        holds (the watchdog re-issues the measured pose); blocking motion RPCs are
        rejected until ``stop_servo_loop``."""
        params = {"owner": self.owner}
        if control_hz is not None:
            params["control_hz"] = float(control_hz)
        self._call("start_servo_loop", **params)

    def servo_stream(self, pose, gripper: Optional[GripperCommand] = None) -> None:
        self._call(
            "servo_stream", owner=self.owner,
            pose=np.asarray(pose, float).reshape(7).tolist(),
            gripper=P.gripper_to_dict(gripper),
        )

    def servo_stream_joint(self, q) -> None:
        self._call("servo_stream_joint", owner=self.owner, q=np.asarray(q, float).tolist())

    def stop_servo_loop(self) -> None:
        self._call("stop_servo_loop", owner=self.owner)
