"""Safety supervisor: profiles + a per-setpoint filter.

Safety is a first-class citizen, not an afterthought, because this controller is
meant for a *shared* lab and the wider community -- "everyone be careful" does
not scale. Every Cartesian/joint setpoint passes through :class:`SafetyFilter`
before it reaches the robot. A profile is a named, version-controlled YAML file
so an experiment is reproducible: "this run used ``tabletop_safe``".

The filter is intentionally cheap (pure numpy, microseconds) so it can run
inside the 1 kHz loop. It either lightly *clips* a setpoint back into bounds or
*rejects* it (-> hold position), and always reports what it did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from . import transforms as T
from .types import RobotState, StopReason


@dataclass
class SafetyProfile:
    name: str = "tabletop_safe"

    # Axis-aligned Cartesian workspace box for the TCP, base frame (metres).
    ws_x: Tuple[float, float] = (0.25, 0.75)
    ws_y: Tuple[float, float] = (-0.45, 0.45)
    ws_z: Tuple[float, float] = (0.06, 0.70)

    # What to do with a commanded position outside the box: "clip" (default)
    # pulls it back onto the boundary and continues; "reject" protective-stops.
    # A verify-then-execute planner should prefer "reject": a silently clipped
    # waypoint means the executed path no longer matches the rollout it scored.
    workspace_action: str = "clip"

    # TCP kinematic limits.
    max_linear_speed: float = 0.25       # m/s
    max_angular_speed: float = 0.60      # rad/s
    max_pose_jump_linear: float = 0.03   # m, per control tick
    max_pose_jump_angular: float = 0.15  # rad, per control tick
    # Optional per-tick translational acceleration cap (m/s^2), applied AFTER the
    # velocity cap so the emitted command stream is C1-bounded (no velocity steps
    # into contact). Opt-in: None leaves the stream velocity-limited only.
    max_linear_accel: Optional[float] = None

    # Joint limits (margin shrinks the hard limits; speed scale caps velocity).
    joint_margin_rad: float = 0.08
    max_joint_speed_scale: float = 0.30
    # Hard joint position limits (Rizon 4/4s nominal; override per robot).
    joint_lower: np.ndarray = field(
        default_factory=lambda: np.array(
            [-2.79, -2.27, -2.97, -1.87, -2.97, -1.74, -2.97], float
        )
    )
    joint_upper: np.ndarray = field(
        default_factory=lambda: np.array(
            [2.79, 2.27, 2.97, 1.87, 2.97, 4.45, 2.97], float
        )
    )

    # Contact thresholds.
    max_contact_wrench: np.ndarray = field(
        default_factory=lambda: np.array([40, 40, 40, 5, 5, 5], float)
    )
    # Ceiling on the per-traj ``contact_wrench_allowance`` a client may
    # request on top of ``max_contact_wrench`` (held-payload transport).
    # Default ZERO: no traj can relax the contact guard unless this
    # deployment's safety YAML explicitly grants headroom (``contact:
    # max_allowance``). The server clamps requests to this, so a runaway
    # client cannot disable collision stopping.
    max_wrench_allowance: np.ndarray = field(
        default_factory=lambda: np.zeros(6, float)
    )

    # Watchdog timeouts (ms).
    command_timeout_ms: float = 100.0
    state_timeout_ms: float = 20.0
    stop_on_stale_command: bool = True
    # Opt-in state-age watchdog: if the backend ever returns a stale-stamped
    # state (e.g. an async/cached state source or a comms hiccup), protective-stop
    # instead of trusting it. Off by default since the shipped backends read fresh
    # state synchronously every tick.
    stop_on_state_stale: bool = False
    # Opt-in escalation: while a command is stale the loop HOLDS; if it stays
    # stale past this much longer timeout, escalate hold -> protective stop. None
    # holds indefinitely (the prior behaviour).
    hold_timeout_ms: Optional[float] = None

    @classmethod
    def from_dict(cls, d: dict) -> "SafetyProfile":
        p = cls(name=d.get("name", "unnamed"))
        ws = d.get("workspace", {})
        if "x" in ws:
            p.ws_x = tuple(ws["x"])
        if "y" in ws:
            p.ws_y = tuple(ws["y"])
        if "z" in ws:
            p.ws_z = tuple(ws["z"])
        if "action" in ws:
            action = str(ws["action"]).lower()
            if action not in ("clip", "reject"):
                raise ValueError(f"workspace.action must be 'clip' or 'reject', got {action!r}")
            p.workspace_action = action
        tl = d.get("tcp_limits", {})
        p.max_linear_speed = tl.get("max_linear_speed", p.max_linear_speed)
        p.max_angular_speed = tl.get("max_angular_speed", p.max_angular_speed)
        p.max_pose_jump_linear = tl.get("max_pose_jump_linear", p.max_pose_jump_linear)
        p.max_pose_jump_angular = tl.get("max_pose_jump_angular", p.max_pose_jump_angular)
        p.max_linear_accel = tl.get("max_linear_accel", p.max_linear_accel)
        jl = d.get("joint_limits", {})
        p.joint_margin_rad = jl.get("margin_rad", p.joint_margin_rad)
        p.max_joint_speed_scale = jl.get("max_joint_speed_scale", p.max_joint_speed_scale)
        if "lower" in jl:
            p.joint_lower = np.asarray(jl["lower"], float)
        if "upper" in jl:
            p.joint_upper = np.asarray(jl["upper"], float)
        ct = d.get("contact", {})
        if "max_wrench" in ct:
            p.max_contact_wrench = np.asarray(ct["max_wrench"], float)
        if "max_allowance" in ct:
            p.max_wrench_allowance = np.asarray(ct["max_allowance"], float)
        wd = d.get("watchdog", {})
        p.command_timeout_ms = wd.get("command_timeout_ms", p.command_timeout_ms)
        p.state_timeout_ms = wd.get("state_timeout_ms", p.state_timeout_ms)
        p.stop_on_stale_command = wd.get("stop_on_stale_command", p.stop_on_stale_command)
        p.stop_on_state_stale = wd.get("stop_on_state_stale", p.stop_on_state_stale)
        p.hold_timeout_ms = wd.get("hold_timeout_ms", p.hold_timeout_ms)
        return p

    def to_config_dict(self) -> dict:
        """Serialize in the same nested layout :meth:`from_dict` reads, so a
        profile can round-trip over the wire (``get_safety_profile`` RPC) and a
        client can rebuild it with ``SafetyProfile.from_dict``."""
        return {
            "name": self.name,
            "workspace": {
                "x": list(self.ws_x),
                "y": list(self.ws_y),
                "z": list(self.ws_z),
                "action": self.workspace_action,
            },
            "tcp_limits": {
                "max_linear_speed": self.max_linear_speed,
                "max_angular_speed": self.max_angular_speed,
                "max_pose_jump_linear": self.max_pose_jump_linear,
                "max_pose_jump_angular": self.max_pose_jump_angular,
                "max_linear_accel": self.max_linear_accel,
            },
            "joint_limits": {
                "margin_rad": self.joint_margin_rad,
                "max_joint_speed_scale": self.max_joint_speed_scale,
                "lower": self.joint_lower.tolist(),
                "upper": self.joint_upper.tolist(),
            },
            "contact": {
                "max_wrench": self.max_contact_wrench.tolist(),
                "max_allowance": self.max_wrench_allowance.tolist(),
            },
            "watchdog": {
                "command_timeout_ms": self.command_timeout_ms,
                "state_timeout_ms": self.state_timeout_ms,
                "stop_on_stale_command": self.stop_on_stale_command,
                "stop_on_state_stale": self.stop_on_state_stale,
                "hold_timeout_ms": self.hold_timeout_ms,
            },
        }

    def validate_trajectory(self, traj) -> list:
        """Client-side preflight: list every violation a traj would hit.

        Returns a list of human-readable strings, empty when the traj is clean.
        Workspace violations are hard errors (the filter would clip or reject
        them); speed notes flag segments the interpolator would time-stretch, so
        the caller's wall-clock estimate can account for it. This is the helper
        a planner should call BEFORE sending a traj, instead of duplicating the
        profile's workspace box in its own constants.
        """
        import math

        problems: list = []
        lo = np.array([self.ws_x[0], self.ws_y[0], self.ws_z[0]])
        hi = np.array([self.ws_x[1], self.ws_y[1], self.ws_z[1]])
        lin_cap = self.max_linear_speed
        ang_cap = self.max_angular_speed
        try:
            traj_lin = float(getattr(traj, "max_tcp_linear_speed", 0.0))
            if traj_lin > 0:
                lin_cap = min(lin_cap, traj_lin)
            traj_ang = float(getattr(traj, "max_tcp_angular_speed", 0.0))
            if traj_ang > 0:
                ang_cap = min(ang_cap, traj_ang)
        except (TypeError, ValueError):
            pass
        # The interpolator stretches on PEAK speed: its cosine blend peaks at
        # pi/2 times the segment's average speed, so the stretch threshold is
        # (pi/2)*dist/duration, not dist/duration.
        peak = math.pi / 2.0
        prev = None
        prev_quat = None
        for i, w in enumerate(traj.waypoints):
            pos = np.asarray(w.position, float).reshape(3)
            if np.any(pos < lo) or np.any(pos > hi):
                problems.append(
                    f"waypoint {i}: position {np.round(pos, 4).tolist()} outside "
                    f"workspace x{list(self.ws_x)} y{list(self.ws_y)} z{list(self.ws_z)} "
                    f"({self.workspace_action} on execution)"
                )
            if prev is not None and w.duration is not None and w.duration > 0:
                dur = float(w.duration)
                dist = float(np.linalg.norm(pos - prev))
                if peak * dist / dur > lin_cap:
                    problems.append(
                        f"waypoint {i}: peak linear speed {peak * dist / dur:.3f} m/s "
                        f"exceeds cap {lin_cap:.3f} m/s (segment will be time-stretched "
                        f"to ~{peak * dist / lin_cap:.2f}s)"
                    )
                if prev_quat is not None and w.quaternion is not None:
                    ang = float(T.quat_angle(prev_quat, w.quaternion))
                    if peak * ang / dur > ang_cap:
                        problems.append(
                            f"waypoint {i}: peak angular speed {peak * ang / dur:.3f} rad/s "
                            f"exceeds cap {ang_cap:.3f} rad/s (segment will be "
                            f"time-stretched to ~{peak * ang / ang_cap:.2f}s)"
                        )
            prev = pos
            if w.quaternion is not None:
                prev_quat = np.asarray(w.quaternion, float).reshape(4)
        return problems


@dataclass
class SafetyResult:
    ok: bool
    pose: Optional[np.ndarray] = None     # filtered TCP pose (Cartesian path)
    q: Optional[np.ndarray] = None        # filtered joint target (joint path)
    clipped: bool = False
    reason: StopReason = StopReason.NONE
    # Which limit(s) triggered a clip (ok=True). Lets the loop report WHY a
    # command was modified, not just that it was. Empty when nothing clipped.
    clip_reasons: list = field(default_factory=list)

    @property
    def primary_clip_reason(self) -> StopReason:
        return self.clip_reasons[0] if self.clip_reasons else StopReason.NONE


class SafetyFilter:
    """Per-setpoint filter. Holds the profile, dt, and the previous *commanded*
    setpoint so limits are enforced on the command stream rather than against
    live state (which would fight normal tracking lag and stall a trajectory).

    Call :meth:`reset` at the start of every motion (traj / servo session / loop
    start) so the first command is referenced to the robot's current pose. For a
    single one-shot ``filter_*`` call with no prior command, the live state is
    used as the reference, which keeps the guard meaningful in unit tests.
    """

    def __init__(self, profile: SafetyProfile, control_dt: float):
        self.p = profile
        self.dt = control_dt
        self._prev_pose: Optional[np.ndarray] = None
        self._prev_q: Optional[np.ndarray] = None
        self._prev_lin_vel: Optional[np.ndarray] = None  # for the accel cap

    def reset(
        self,
        state: Optional[RobotState] = None,
        *,
        pose: Optional[np.ndarray] = None,
        q: Optional[np.ndarray] = None,
    ) -> None:
        """Anchor the filter to a known starting setpoint.

        Pass the current :class:`RobotState` (preferred) so both the Cartesian
        and joint references are set, or pass an explicit ``pose`` / ``q``.
        """
        if state is not None:
            self._prev_pose = np.asarray(state.tcp_pose, float).reshape(7).copy()
            self._prev_q = np.asarray(state.q, float).copy()
        if pose is not None:
            self._prev_pose = np.asarray(pose, float).reshape(7).copy()
        if q is not None:
            self._prev_q = np.asarray(q, float).copy()
        # Re-anchoring zeroes the commanded velocity so the accel cap ramps from
        # rest (no spurious accel clip on the first command after a hold/reset).
        self._prev_lin_vel = np.zeros(3)

    def set_profile(self, profile: SafetyProfile) -> None:
        self.p = profile

    # -- Cartesian -----------------------------------------------------------
    def filter_cartesian(
        self,
        target_pose: np.ndarray,
        state: RobotState,
        *,
        max_contact_wrench: Optional[np.ndarray] = None,
    ) -> SafetyResult:
        p = self.p
        target_pose = np.asarray(target_pose, float).reshape(7).copy()
        cur = np.asarray(state.tcp_pose, float).reshape(7)
        clipped = False
        reasons: list = []

        # 1. Contact wrench: hard stop if exceeded (don't keep pushing). The
        #    executor passes the traj-effective cap (profile +- per-traj
        #    tightening/allowance); standalone callers get the profile cap.
        wrench_cap = p.max_contact_wrench if max_contact_wrench is None else max_contact_wrench
        if np.any(np.abs(state.wrench) > wrench_cap):
            return SafetyResult(ok=False, reason=StopReason.CONTACT_WRENCH)

        # 2. Workspace box on position: clip back to the boundary, or -- when the
        #    profile says ``workspace_action: reject`` -- protective-stop, so a
        #    verify-then-execute planner fails loudly instead of executing a
        #    silently displaced path.
        lo = np.array([p.ws_x[0], p.ws_y[0], p.ws_z[0]])
        hi = np.array([p.ws_x[1], p.ws_y[1], p.ws_z[1]])
        clipped_pos = np.clip(target_pose[:3], lo, hi)
        if not np.allclose(clipped_pos, target_pose[:3]):
            if p.workspace_action == "reject":
                return SafetyResult(ok=False, reason=StopReason.WORKSPACE)
            clipped = True
            reasons.append(StopReason.WORKSPACE)
            target_pose[:3] = clipped_pos

        # 3. Pose-jump + speed limits are enforced against the previous
        #    *commanded* setpoint, not live state: the command stream is what we
        #    police, and using live state would mistake tracking lag for a jump
        #    and progressively stall the trajectory. On the very first command
        #    (no history) fall back to the live pose so the guard still bites.
        ref = self._prev_pose if self._prev_pose is not None else cur

        lin_step = float(np.linalg.norm(target_pose[:3] - ref[:3]))
        if lin_step > p.max_pose_jump_linear:
            scale = p.max_pose_jump_linear / max(lin_step, 1e-9)
            target_pose[:3] = ref[:3] + (target_pose[:3] - ref[:3]) * scale
            clipped = True
            reasons.append(StopReason.POSE_JUMP)

        ang_step = T.quat_angle(ref[3:7], target_pose[3:7])
        if ang_step > p.max_pose_jump_angular:
            t = p.max_pose_jump_angular / max(ang_step, 1e-9)
            target_pose[3:7] = T.quat_slerp(ref[3:7], target_pose[3:7], t)
            clipped = True
            reasons.append(StopReason.POSE_JUMP)

        # 4. Speed limit (per tick) against the previous command. With the
        #    interpolator honouring the velocity cap this is a no-op for
        #    in-spec trajs; for raw servo/MPC setpoints it saturates motion
        #    toward the latest target at <= max_linear_speed.
        lin_step = float(np.linalg.norm(target_pose[:3] - ref[:3]))
        if lin_step / max(self.dt, 1e-9) > p.max_linear_speed:
            scale = (p.max_linear_speed * self.dt) / max(lin_step, 1e-9)
            target_pose[:3] = ref[:3] + (target_pose[:3] - ref[:3]) * scale
            clipped = True
            reasons.append(StopReason.TCP_SPEED)

        # 4b. Angular speed limit (per tick) against the previous command,
        #     mirroring the linear cap so max_angular_speed is actually enforced
        #     on raw servo/MPC/teleop setpoints (the server path feeds these
        #     straight in with no interpolator). Without this the only angular
        #     guard is the per-tick jump cap, which at 100 Hz permits ~15 rad/s.
        ang_step = T.quat_angle(ref[3:7], target_pose[3:7])
        if ang_step / max(self.dt, 1e-9) > p.max_angular_speed:
            t = (p.max_angular_speed * self.dt) / max(ang_step, 1e-9)
            target_pose[3:7] = T.quat_slerp(ref[3:7], target_pose[3:7], t)
            clipped = True
            reasons.append(StopReason.TCP_SPEED)

        # 4c. Acceleration cap (opt-in): bound the change in commanded
        #     translational velocity between ticks so the emitted stream is
        #     C1-continuous (no velocity step into contact). Applied last, after
        #     the velocity cap, against the previous commanded velocity.
        if p.max_linear_accel is not None:
            v_prev = self._prev_lin_vel if self._prev_lin_vel is not None else np.zeros(3)
            v_new = (target_pose[:3] - ref[:3]) / max(self.dt, 1e-9)
            dv = v_new - v_prev
            dv_norm = float(np.linalg.norm(dv))
            max_dv = p.max_linear_accel * self.dt
            if dv_norm > max_dv:
                v_new = v_prev + dv * (max_dv / max(dv_norm, 1e-9))
                target_pose[:3] = ref[:3] + v_new * self.dt
                clipped = True
                reasons.append(StopReason.TCP_SPEED)
            self._prev_lin_vel = v_new
        else:
            self._prev_lin_vel = (target_pose[:3] - ref[:3]) / max(self.dt, 1e-9)

        self._prev_pose = target_pose.copy()
        return SafetyResult(ok=True, pose=target_pose, clipped=clipped, clip_reasons=reasons)

    # -- Joint ---------------------------------------------------------------
    def filter_joint(self, target_q: np.ndarray, state: RobotState) -> SafetyResult:
        p = self.p
        target_q = np.asarray(target_q, float).copy()
        cur_q = np.asarray(state.q, float)
        clipped = False
        reasons: list = []

        lo = p.joint_lower + p.joint_margin_rad
        hi = p.joint_upper - p.joint_margin_rad
        clipped_q = np.clip(target_q, lo, hi)
        if not np.allclose(clipped_q, target_q):
            clipped = True
            reasons.append(StopReason.JOINT_LIMIT)
            target_q = clipped_q

        # Per-tick step limit from a coarse joint-speed cap (rad/s), referenced
        # to the previous joint command (see filter_cartesian for rationale).
        ref_q = self._prev_q if self._prev_q is not None else cur_q
        max_step = 2.0 * p.max_joint_speed_scale * self.dt
        step = target_q - ref_q
        big = np.abs(step) > max_step
        if np.any(big):
            step[big] = np.sign(step[big]) * max_step
            target_q = ref_q + step
            clipped = True
            reasons.append(StopReason.JOINT_LIMIT)

        self._prev_q = target_q.copy()
        return SafetyResult(ok=True, q=target_q, clipped=clipped, clip_reasons=reasons)

    # -- Watchdog ------------------------------------------------------------
    def check_command_age(self, age_ms: float) -> SafetyResult:
        if self.p.stop_on_stale_command and age_ms > self.p.command_timeout_ms:
            return SafetyResult(ok=False, reason=StopReason.STALE_COMMAND)
        return SafetyResult(ok=True)

    def check_state_age(self, age_ms: float) -> SafetyResult:
        """State-age watchdog (mirrors :meth:`check_command_age`). Opt-in via
        ``stop_on_state_stale``: a backend returning a stale-stamped state can't
        be trusted for position, so protective-stop rather than command on it."""
        if self.p.stop_on_state_stale and age_ms > self.p.state_timeout_ms:
            return SafetyResult(ok=False, reason=StopReason.STALE_STATE)
        return SafetyResult(ok=True)
