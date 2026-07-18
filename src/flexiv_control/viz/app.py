"""RobotViz -- a browser-based live mirror + intended-motion preview.

Open ``viz.url`` in any browser on the LAN and you see, live:

* the robot (mesh mirror when a URDF is available, else a TCP frame),
* the measured TCP trail,
* the active safety profile's workspace box (amber = clip, red = reject),
* a wrench bar + status HUD (mode / stop reason / lease owner / loop health),
* and -- the headline -- the INTENDED motion of the next traj: the true
  per-tick command stream (including time-stretching), time-colored
  start->end, with waypoint knots, gripper open/close glyphs, the terminal
  pose, and an animated ghost TCP, plus a staleness banner and an optional
  Approve / Reject gate for per-traj confirmation.

Design rules (see docs/visualization.md for the full rationale):

* The TCP marker / trail / preview anchor ALWAYS come from the robot-streamed
  ``tcp_pose`` -- never local URDF FK (flexiv_rdk #82: URDF FK is cosmetic).
* The viewer polls ``get_state()`` (default 20 Hz); the server serves these
  from per-tick snapshots, so polling never blocks a running traj.
* A monitoring viewer must NOT own the arm: ``attach()`` refuses a
  lease-holding ``RemoteRobot`` unless ``allow_lease=True`` (the embedded
  planner case, where the caller legitimately holds the lease).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import viser

from ..trajectory import CartesianTrajectory, ExecutionResult
from ..safety import SafetyProfile
from ..types import RobotState
from . import assets
from .preview import (
    TrajectoryPreview,
    plan_trajectory_preview,
    pose_distance,
    time_colors,
    trail_segments,
    workspace_box_edges,
)

# Staleness thresholds for the go/no-go gate: a preview planned from a pose
# this far from the live TCP no longer starts where the robot is.
STALE_LINEAR_M = 0.005
STALE_ANGULAR_RAD = 0.035  # ~2 degrees


class RobotViz:
    """Live robot mirror + intended-motion preview, served to a browser.

    Works with anything exposing ``get_state()`` (a local
    :class:`~flexiv_control.Robot` or a connected
    :class:`~flexiv_control.RemoteRobot`).
    """

    def __init__(
        self,
        model: Optional[Union[str, Path]] = None,
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        state_hz: float = 20.0,
        trail_length: int = 512,
        control_hz: float = 100.0,
    ) -> None:
        self.state_hz = float(state_hz)
        self.control_hz = float(control_hz)
        self.server = viser.ViserServer(host=host, port=port, label="flexiv-control")
        self.server.scene.add_grid("/scene/ground", width=2.0, height=2.0)
        self.server.scene.add_frame("/scene/base", axes_length=0.09, axes_radius=0.003)

        # Land every new browser client on a useful view of the tabletop
        # workspace instead of viser's distant default.
        @self.server.on_client_connect
        def _(client) -> None:
            try:
                client.camera.position = (1.25, -1.05, 0.85)
                client.camera.look_at = (0.5, 0.0, 0.2)
            except Exception:
                pass

        # -- measured robot ---------------------------------------------------
        # The mesh mirror is driven BY NAME, never by position: the vendor URDF
        # lists the gripper's drive joint (finger_width_joint) FIRST among the
        # actuated joints, so blindly feeding state.q would twist the gripper
        # with joint1. The GN01 gripper is fully articulated through the URDF's
        # mimic chain (knuckle = 9.404*width - 0.155, the same calibration the
        # MJCF uses), driven directly by the streamed gripper width in metres.
        self._urdf_vis = None
        self._arm_joint_idx: list = []
        self._width_joint_idx: Optional[int] = None
        self._cfg_len = 0
        if model is not None:
            try:
                urdf = assets.load_urdf(Path(model))
                from viser.extras import ViserUrdf

                self._urdf_vis = ViserUrdf(
                    self.server, urdf_or_path=urdf, root_node_name="/robot/arm"
                )
                names = list(urdf.actuated_joint_names)
                self._cfg_len = len(names)
                self._arm_joint_idx = [
                    names.index(f"joint{i}") for i in range(1, 8) if f"joint{i}" in names
                ]
                if len(self._arm_joint_idx) != 7:
                    # non-vendor naming: first seven non-gripper actuated joints
                    self._arm_joint_idx = [
                        i for i, n in enumerate(names) if "finger" not in n.lower()
                    ][:7]
                if "finger_width_joint" in names:
                    self._width_joint_idx = names.index("finger_width_joint")
            except Exception as e:  # noqa: BLE001  -- assets are never load-bearing
                print(f"[RobotViz] URDF load failed ({type(e).__name__}: {e}); "
                      "running in frames mode")
                self._urdf_vis = None
        self._tcp = self.server.scene.add_frame(
            "/robot/tcp", axes_length=0.07, axes_radius=0.0035
        )
        # Parametric gripper jaws under the TCP frame, driven by the streamed
        # width -- the frames-mode stand-in. Hidden when the URDF brings the
        # real articulated GN01 mesh (its mimic chain shows the true fingers).
        jaw = dict(color=(70, 70, 80), dimensions=(0.012, 0.004, 0.05))
        self._jaw_l = self.server.scene.add_box("/robot/tcp/jaw_l", **jaw)
        self._jaw_r = self.server.scene.add_box("/robot/tcp/jaw_r", **jaw)
        if self._width_joint_idx is not None:
            self._jaw_l.visible = False
            self._jaw_r.visible = False
        # Geometry handles in viser expose transform properties only; dynamic
        # geometry (trail / workspace box) is refreshed by re-adding under the
        # SAME name, which atomically replaces the previous node.
        self._trail = None

        # -- safety-profile workspace box --------------------------------------
        self._workspace = None
        self._workspace_action = ""

        # -- GUI ----------------------------------------------------------------
        with self.server.gui.add_folder("Status"):
            self._status_md = self.server.gui.add_markdown("_waiting for state..._")
        with self.server.gui.add_folder("Contact wrench"):
            self._wrench_bar = self.server.gui.add_progress_bar(0.0)
            self._wrench_md = self.server.gui.add_markdown("")
        with self.server.gui.add_folder("Intended motion"):
            self._preview_md = self.server.gui.add_markdown("_no traj previewed_")
            # Percent-based scrub (slider bounds are fixed at creation in viser,
            # so the index mapping happens in _place_ghost).
            self._scrub = self.server.gui.add_slider(
                "preview %", min=0, max=100, step=1, initial_value=0
            )
            self._loop = self.server.gui.add_checkbox("animate ghost", True)
            self._approve_group = None  # created on demand by gate(require_click=True)

        @self._scrub.on_update
        def _(_evt) -> None:
            self._place_ghost_pct(float(self._scrub.value))

        # -- preview scene handles (replaced per preview) -----------------------
        self._plan_handles: list = []
        self._ghost = None
        self._ghost_frame = None
        self._preview: Optional[TrajectoryPreview] = None
        self._preview_lock = threading.Lock()

        # -- poller -------------------------------------------------------------
        self._robot = None
        self._profile: Optional[SafetyProfile] = None
        self._trail_buf: list = []
        self._trail_len = int(trail_length)
        self._last_state: Optional[RobotState] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._running = False
        self._gate_event = threading.Event()
        self._gate_verdict = False

    # ------------------------------------------------------------------ basics
    @property
    def url(self) -> str:
        host = self.server.get_host()
        port = self.server.get_port()
        shown = "localhost" if host in ("0.0.0.0", "::") else host
        return f"http://{shown}:{port}"

    def stop(self) -> None:
        self._running = False
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
            self._poll_thread = None
        self.server.stop()

    # ------------------------------------------------------------- state feeds
    def attach(self, robot, *, allow_lease: bool = False) -> None:
        """Mirror ``robot`` live: spawns a daemon poller calling ``get_state()``
        at ``state_hz`` and refreshing the safety profile every 2 s.

        A pure MONITOR must not own the arm: pass a ``RemoteRobot`` that was
        ``connect()``-ed WITHOUT acquiring a lease (never ``with RemoteRobot``,
        whose ``__enter__`` takes the lease). A lease-holding robot is refused
        unless ``allow_lease=True`` -- the embedded-planner case.
        """
        if getattr(robot, "_has_lease", False) and not allow_lease:
            raise ValueError(
                "this RemoteRobot HOLDS THE LEASE -- a monitoring viewer must "
                "not own the arm (use RemoteRobot(...).connect() without "
                "acquire_lease, or pass allow_lease=True if you really are "
                "the controlling process)"
            )
        self._robot = robot
        if self._poll_thread is None:
            self._running = True
            self._poll_thread = threading.Thread(
                target=self._poll_loop, daemon=True, name="flexiv-viz-poller"
            )
            self._poll_thread.start()

    def update(self, state: RobotState) -> None:
        """Manual push for callers that own their loop (no poller thread)."""
        self._apply_state(state)

    # -------------------------------------------------------------- the poller
    def _poll_loop(self) -> None:
        next_profile = 0.0
        period = 1.0 / max(self.state_hz, 1e-3)
        ghost_idx = 0
        while self._running:
            t0 = time.time()
            try:
                state = self._robot.get_state()
                self._apply_state(state)
                if t0 >= next_profile:
                    self._refresh_profile()
                    next_profile = t0 + 2.0
            except Exception as e:  # noqa: BLE001 -- a viz must never kill anything
                self._status_md.content = (
                    f"**state poll failed**: `{type(e).__name__}: {e}`"
                )
            # ghost playback (no extra thread): advance ~real time, loop.
            with self._preview_lock:
                n = 0 if self._preview is None else len(self._preview.setpoints)
            if n > 1 and self._loop.value:
                step = max(1, int(self.control_hz / max(self.state_hz, 1.0)))
                ghost_idx = (ghost_idx + step) % n
                pct = 100.0 * ghost_idx / max(n - 1, 1)
                self._scrub.value = int(pct)  # slider follows; on_update places ghost
            sleep = period - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def _refresh_profile(self) -> None:
        robot = self._robot
        prof = None
        if hasattr(robot, "get_safety_profile"):
            prof = robot.get_safety_profile()
        elif hasattr(robot, "profile"):
            prof = robot.profile
        if prof is None:
            return
        self._profile = prof
        edges = workspace_box_edges(prof)
        color = (255, 60, 60) if prof.workspace_action == "reject" else (255, 180, 40)
        # Same-name re-add atomically replaces the node (handles have no
        # geometry setters); this runs at 0.5 Hz, so the cost is negligible.
        self._workspace = self.server.scene.add_line_segments(
            "/scene/workspace", points=edges, colors=color, line_width=2.0
        )
        self._workspace_action = prof.workspace_action

    def _apply_state(self, state: RobotState) -> None:
        self._last_state = state
        pose = np.asarray(state.tcp_pose, float).reshape(7)
        self._tcp.position = pose[:3]
        self._tcp.wxyz = pose[3:7]
        half = max(float(state.gripper_width), 0.0) / 2.0 + 0.006
        self._jaw_l.position = (0.0, +half, 0.02)
        self._jaw_r.position = (0.0, -half, 0.02)
        if self._urdf_vis is not None:
            try:
                cfg = np.zeros(self._cfg_len)
                q = np.asarray(state.q, float)
                for k, idx in enumerate(self._arm_joint_idx[: len(q)]):
                    cfg[idx] = q[k]
                if self._width_joint_idx is not None:
                    cfg[self._width_joint_idx] = max(float(state.gripper_width), 0.0)
                self._urdf_vis.update_cfg(cfg)
            except Exception:  # mismatched DOF / actuated-joint count: go frames-only
                self._urdf_vis = None
        # trail (same-name re-add: geometry handles have no .points setter)
        self._trail_buf.append(pose[:3].copy())
        if len(self._trail_buf) > self._trail_len:
            self._trail_buf = self._trail_buf[-self._trail_len:]
        if len(self._trail_buf) >= 2:
            self._trail = self.server.scene.add_line_segments(
                "/robot/trail",
                points=trail_segments(np.asarray(self._trail_buf)),
                colors=(80, 160, 255),
                line_width=2.0,
            )
        # wrench HUD
        f = np.asarray(state.wrench[:3], float)
        fmag = float(np.linalg.norm(f))
        cap = 40.0
        if self._profile is not None:
            cap = float(np.min(self._profile.max_contact_wrench[:3]))
        self._wrench_bar.value = float(np.clip(100.0 * fmag / max(cap, 1e-6), 0.0, 100.0))
        self._wrench_md.content = f"|F| = **{fmag:.1f} N** / cap {cap:.0f} N"
        # status HUD
        self._status_md.content = (
            f"mode `{state.control_mode.value}` · safety `{state.safety_status.value}` · "
            f"stop `{state.stop_reason.value}`\n\n"
            f"owner `{state.active_owner or '-'}` · gripper **{state.gripper_width:.4f} m**\n\n"
            f"loop {state.loop_period_ms:.1f} ms · jitter {state.loop_jitter_us:.0f} µs · "
            f"missed {state.missed_cycles}"
        )

    # ---------------------------------------------------------- intended motion
    def preview_trajectory(
        self,
        traj: CartesianTrajectory,
        state: Optional[RobotState] = None,
        profile: Optional[SafetyProfile] = None,
        *,
        traj_id: str = "",
    ) -> TrajectoryPreview:
        """Render the traj's TRUE intended motion (the executor's own
        resolution + caps + interpolation) and return the
        :class:`~flexiv_control.viz.preview.TrajectoryPreview`."""
        if state is None:
            state = self._last_state
        if state is None and self._robot is not None:
            state = self._robot.get_state()
        if state is None:
            raise RuntimeError("preview_trajectory needs a RobotState (attach() a robot "
                               "or pass state= explicitly)")
        if profile is None:
            profile = self._profile
        pv = plan_trajectory_preview(
            traj, np.asarray(state.tcp_pose, float), profile, control_hz=self.control_hz
        )
        self._render_preview(pv, traj_id=traj_id)
        return pv

    def _render_preview(self, pv: TrajectoryPreview, *, traj_id: str = "") -> None:
        self.clear_preview()
        s = self.server.scene
        handles = []
        pts = pv.setpoints[:, :3]
        if len(pts) >= 2:
            segs = np.stack([pts[:-1], pts[1:]], axis=1)            # (N-1, 2, 3)
            cols = time_colors(len(pts))
            seg_cols = np.stack([cols[:-1], cols[1:]], axis=1)      # (N-1, 2, 3)
            handles.append(s.add_line_segments(
                "/plan/path", points=segs, colors=seg_cols, line_width=4.0))
        if len(pv.waypoints):
            handles.append(s.add_point_cloud(
                "/plan/knots", points=pv.waypoints,
                colors=(255, 255, 255), point_size=0.009, point_shape="circle"))
        for i, ev in enumerate(pv.gripper_events):
            color = (40, 220, 90) if ev.closing else (80, 150, 255)
            handles.append(s.add_icosphere(
                f"/plan/grip_{i}", radius=0.011, color=color,
                position=ev.position, opacity=0.9))
        term = pv.terminal_pose
        handles.append(s.add_frame(
            "/plan/terminal", axes_length=0.05, axes_radius=0.0025,
            position=term[:3], wxyz=term[3:7]))
        # ghost TCP (animated via the scrub slider / poller ticker)
        self._ghost = s.add_icosphere(
            "/plan/ghost", radius=0.014, color=(255, 220, 60), opacity=0.55,
            position=pts[0] if len(pts) else (0, 0, 0))
        self._ghost_frame = s.add_frame(
            "/plan/ghost_frame", axes_length=0.045, axes_radius=0.002,
            position=pv.start_pose[:3], wxyz=pv.start_pose[3:7])
        self._plan_handles = handles

        with self._preview_lock:
            self._preview = pv
        self._scrub.value = 0

        stretched = (
            f" · **time-stretched** {pv.nominal_duration_s:.1f}s → {pv.duration_s:.1f}s"
            if pv.time_stretched else ""
        )
        warn = ("\n\n⚠ " + "\n\n⚠ ".join(pv.warnings)) if pv.warnings else ""
        title = f"**traj {traj_id}**" if traj_id else "**traj**"
        self._preview_md.content = (
            f"{title}: {len(pv.waypoints)} waypoints · {pv.duration_s:.1f}s"
            f" · caps {pv.linear_speed_cap:.2f} m/s, {pv.angular_speed_cap:.2f} rad/s"
            f"{stretched}{warn}"
        )

    def _place_ghost_pct(self, pct: float) -> None:
        with self._preview_lock:
            pv = self._preview
        if pv is None or not len(pv.setpoints):
            return
        idx = int(round(np.clip(pct, 0.0, 100.0) / 100.0 * (len(pv.setpoints) - 1)))
        pose = pv.setpoints[idx]
        if self._ghost is not None:
            self._ghost.position = pose[:3]
        if self._ghost_frame is not None:
            self._ghost_frame.position = pose[:3]
            self._ghost_frame.wxyz = pose[3:7]

    def clear_preview(self) -> None:
        with self._preview_lock:
            self._preview = None
        for h in self._plan_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._plan_handles = []
        for h in (self._ghost, self._ghost_frame):
            if h is not None:
                try:
                    h.remove()
                except Exception:
                    pass
        self._ghost = None
        self._ghost_frame = None

    # ------------------------------------------------------------------- gates
    def gate(
        self, *, require_click: bool = False, timeout: Optional[float] = None
    ) -> Callable[[int, CartesianTrajectory], bool]:
        """An ``on_propose`` callable for
        :class:`~flexiv_control.RecedingHorizonRunner` (or any loop): renders
        the preview, refuses a STALE one (live TCP moved > 5 mm / 2° from the
        preview's start pose), and -- with ``require_click=True`` -- blocks
        until the operator presses Approve / Reject in the browser."""

        def _gate(step: int, traj: CartesianTrajectory) -> bool:
            state = self._robot.get_state() if self._robot is not None else self._last_state
            if state is None:
                return False
            pv = self.preview_trajectory(traj, state, traj_id=str(step))
            lin, ang = pose_distance(pv.start_pose, np.asarray(state.tcp_pose, float))
            if lin > STALE_LINEAR_M or ang > STALE_ANGULAR_RAD:
                self._preview_md.content = (
                    f"🟥 **PREVIEW STALE** (moved {1000 * lin:.1f} mm / "
                    f"{np.degrees(ang):.1f}° since planning) -- re-propose"
                )
                return False
            if not require_click:
                return True
            return self._wait_for_click(step, timeout)

        return _gate

    def _wait_for_click(self, step: int, timeout: Optional[float]) -> bool:
        if self._approve_group is None:
            self._approve_group = self.server.gui.add_button_group(
                "execute?", ("Approve", "Reject")
            )

            @self._approve_group.on_click
            def _(evt) -> None:
                self._gate_verdict = evt.target.value == "Approve"
                self._gate_event.set()

        self._gate_event.clear()
        self._preview_md.content += f"\n\n⏳ awaiting Approve/Reject for traj {step}..."
        ok = self._gate_event.wait(timeout=timeout)
        return bool(ok and self._gate_verdict)

    def on_step(self, step: int, traj: CartesianTrajectory, result: ExecutionResult) -> None:
        """Post-execution hook: clear the preview, flash the outcome, and --
        when the executor recorded a trajectory (``record=True``) -- overlay
        commanded vs measured paths for debugging."""
        self.clear_preview()
        verdict = "✅" if result.success else f"🟥 {result.stop_reason}"
        clip = " · ⚠ clipped" if result.clipped else ""
        self._preview_md.content = f"traj {step}: {verdict}{clip} · {result.summary()}"
        traj = result.log.get("trajectory") if isinstance(result.log, dict) else None
        if traj:
            rows = np.asarray(traj, float)        # [t, cmd(7), meas(7), wrench(6)]
            cmd, meas = rows[:, 1:4], rows[:, 8:11]
            if len(cmd) >= 2:
                self._plan_handles.append(self.server.scene.add_line_segments(
                    "/plan/executed_cmd",
                    points=np.stack([cmd[:-1], cmd[1:]], axis=1),
                    colors=(180, 180, 180), line_width=2.0))
                self._plan_handles.append(self.server.scene.add_line_segments(
                    "/plan/executed_meas",
                    points=np.stack([meas[:-1], meas[1:]], axis=1),
                    colors=(255, 120, 240), line_width=2.0))
