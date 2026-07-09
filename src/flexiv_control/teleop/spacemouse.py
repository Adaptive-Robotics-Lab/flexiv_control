"""SpaceMouse teleoperation -- and the same device as an RL intervention source.

Two jobs, one module, mirroring SERL/HIL-SERL: (1) classic 6-DoF teleop that
drives the unified controller, and (2) a human-intervention wrapper that lets an
operator grab control during RL rollouts (the core mechanism behind
human-in-the-loop RL). Because both go through the same Cartesian-delta action
contract, teleop demos, RL interventions, and policy actions are interchangeable.

The hardware reader (``pyspacemouse``) is optional and imported lazily; a
``ScriptedSpaceMouseSource`` lets the teleop loop, demos, and tests run with no
device attached. Install the extra with ``pip install "flexiv-control[teleop]"``.

This replaces the role of the old ``spacemouse_to_servo.py`` bridge: instead of
emitting MoveIt-Servo ``TwistStamped``, it emits the controller's frame-agnostic
Cartesian delta, so the *same* SpaceMouse can drive the conservative MoveIt path
or the fast reactive RDK path by choice of backend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..types import GripperCommand


@dataclass
class SpaceMouseState:
    """Normalised 6-DoF input in ``[-1, 1]`` plus button states."""

    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    buttons: List[int] = field(default_factory=list)

    def magnitude(self) -> float:
        return float(np.linalg.norm(np.concatenate([self.translation, self.rotation])))


class SpaceMouseSource:
    """Abstract input source."""

    def open(self) -> None: ...
    def close(self) -> None: ...
    def read(self) -> SpaceMouseState:
        raise NotImplementedError


class ScriptedSpaceMouseSource(SpaceMouseSource):
    """A deterministic source for demos/tests (no hardware).

    ``fn(t)`` returns a :class:`SpaceMouseState` for elapsed time ``t``. The
    default traces a gentle circle in the xy-plane with the deadman held.
    """

    def __init__(self, fn: Optional[Callable[[float], SpaceMouseState]] = None):
        self.fn = fn or self._default
        self._t0 = 0.0

    @staticmethod
    def _default(t: float) -> SpaceMouseState:
        return SpaceMouseState(
            translation=np.array([0.3 * np.cos(t), 0.3 * np.sin(t), 0.0]),
            rotation=np.zeros(3),
            buttons=[1, 0],  # button 0 = deadman held
        )

    def open(self) -> None:
        self._t0 = time.time()

    def read(self) -> SpaceMouseState:
        return self.fn(time.time() - self._t0)


class PySpaceMouseSource(SpaceMouseSource):
    """Real SpaceMouse via the optional ``pyspacemouse`` package."""

    def __init__(self):
        self._mouse = None

    def open(self) -> None:
        try:
            import pyspacemouse  # type: ignore
        except Exception as e:  # pragma: no cover - hardware path
            raise RuntimeError(
                "pyspacemouse not installed. `pip install \"flexiv-control[teleop]\"` "
                "(and ensure the SpaceMouse is connected / hidapi is available), or "
                "use ScriptedSpaceMouseSource for a hardware-free run."
            ) from e
        if not pyspacemouse.open():
            raise RuntimeError("failed to open SpaceMouse device")
        self._mouse = pyspacemouse

    def read(self) -> SpaceMouseState:  # pragma: no cover - hardware path
        st = self._mouse.read()
        return SpaceMouseState(
            translation=np.array([st.x, st.y, st.z]),
            rotation=np.array([st.roll, st.pitch, st.yaw]),
            buttons=list(getattr(st, "buttons", [])),
        )

    def close(self) -> None:  # pragma: no cover - hardware path
        if self._mouse is not None:
            self._mouse.close()


class SpaceMouseTeleop:
    def __init__(
        self,
        robot: object,
        source: Optional[SpaceMouseSource] = None,
        *,
        publish_hz: float = 100.0,
        pos_scale: float = 0.06,     # m/s at full deflection
        rot_scale: float = 0.5,      # rad/s at full deflection
        deadband: float = 0.05,
        deadman_button: Optional[int] = 0,
        gripper_button: int = 1,
        gripper_open_width: float = 0.09,
        gripper_close_width: float = 0.01,
        initial_open: bool = False,
        signs: Optional[List[float]] = None,
        frame: str = "base",
        owner: str = "spacemouse",
    ):
        self.robot = robot
        self.source = source or ScriptedSpaceMouseSource()
        self.publish_hz = float(publish_hz)
        self.dt = 1.0 / self.publish_hz
        self.pos_scale = float(pos_scale)
        self.rot_scale = float(rot_scale)
        self.deadband = float(deadband)
        self.deadman_button = deadman_button
        self.gripper_button = gripper_button
        self.frame = frame
        self.owner = owner
        self.gripper_open_width = float(gripper_open_width)
        self.gripper_close_width = float(gripper_close_width)
        # Matches the ROS teleop bridge convention: assume closed at start so
        # the FIRST gripper-button press opens.
        self._gripper_open = bool(initial_open)
        # Optional per-axis sign flips [x, y, z, rx, ry, rz] so the device's
        # axes can be calibrated to the robot frame (all +1 by default).
        self.signs = np.ones(6) if signs is None else np.asarray(signs, float).reshape(6)
        self._prev_gripper_button = 0

    # -- mapping -------------------------------------------------------------
    def to_delta(self, st: SpaceMouseState) -> np.ndarray:
        """SpaceMouse state -> per-tick Cartesian delta ``[dx..drz]`` (metres/rad;
        the rotation triplet is an axis-angle rotation vector, not Euler)."""
        trans = self._apply_deadband(st.translation)
        rot = self._apply_deadband(st.rotation)
        delta = np.empty(6)
        delta[:3] = trans * self.signs[:3] * self.pos_scale * self.dt
        delta[3:] = rot * self.signs[3:] * self.rot_scale * self.dt
        return delta

    def _apply_deadband(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, float)
        out = v.copy()
        out[np.abs(v) < self.deadband] = 0.0
        return out

    def _deadman_ok(self, st: SpaceMouseState) -> bool:
        if self.deadman_button is None:
            return True
        return (
            len(st.buttons) > self.deadman_button
            and bool(st.buttons[self.deadman_button])
        )

    def _gripper_from_buttons(self, st: SpaceMouseState) -> Optional[GripperCommand]:
        """Toggle on the button's RISING EDGE only.

        A held button lasts many control ticks; toggling on level would flip
        the gripper open/closed at the loop rate for as long as it is pressed.
        """
        pressed = int(
            len(st.buttons) > self.gripper_button and bool(st.buttons[self.gripper_button])
        )
        rising = pressed == 1 and self._prev_gripper_button == 0
        self._prev_gripper_button = pressed
        if not rising:
            return None
        self._gripper_open = not self._gripper_open
        width = self.gripper_open_width if self._gripper_open else self.gripper_close_width
        return GripperCommand(width=width, force=20.0, grasp=not self._gripper_open)

    # -- RL intervention (SERL / HIL-SERL pattern) ---------------------------
    def intervention(self, policy_action: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Override ``policy_action`` when the operator is actively moving the device.

        Returns ``(action, intervened)``. ``action`` is a 7-vector in ``[-1, 1]``
        (``[dx,dy,dz,drx,dry,drz,gripper]``; rotation = axis-angle rotvec) suitable for
        :class:`~flexiv_control.envs.FlexivRealEnv`; ``intervened`` flags whether
        the human took over (so the RL loop can label the transition).
        """
        st = self.source.read()
        self._gripper_from_buttons(st)  # keep the toggle state fresh
        moving = st.magnitude() > self.deadband and self._deadman_ok(st)
        if not moving:
            return np.asarray(policy_action, float).reshape(7), False
        a = np.zeros(7)
        a[:3] = np.clip(self._apply_deadband(st.translation), -1, 1)
        a[3:6] = np.clip(self._apply_deadband(st.rotation), -1, 1)
        a[6] = -1.0 if not self._gripper_open else 1.0
        return a, True

    # -- teleop loop ---------------------------------------------------------
    def run(self, duration: Optional[float] = None, *, max_ticks: Optional[int] = None) -> int:
        """Drive the robot until ``duration`` / ``max_ticks`` elapse. Returns ticks."""
        self.source.open()
        self._lease()
        self.robot.start_cartesian_impedance()
        t_start = time.time()
        next_tick = time.perf_counter()
        ticks = 0
        try:
            while True:
                if duration is not None and (time.time() - t_start) >= duration:
                    break
                if max_ticks is not None and ticks >= max_ticks:
                    break
                st = self.source.read()
                # Track button edges every tick so a press while the deadman is
                # released cannot fire a stale toggle later.
                grip = self._gripper_from_buttons(st)
                if self._deadman_ok(st):
                    delta = self.to_delta(st)
                    self.robot.servo_cartesian_delta(
                        delta, duration=self.dt, frame=self.frame, gripper=grip
                    )
                ticks += 1
                next_tick += self.dt
                sleep = next_tick - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.perf_counter()
        finally:
            self.source.close()
        return ticks

    def _lease(self) -> None:
        try:
            self.robot.acquire_lease(self.owner)
        except TypeError:
            self.robot.acquire_lease()
