"""An always-on, single-writer reactive control loop (Python tier).

This is the Python embodiment of the 1 kHz C++ daemon: one background thread is
the *only* writer to the backend. High-level code (a SpaceMouse bridge, an MPC
or RL step) just updates the latest setpoint; the loop streams it at a fixed
rate and -- crucially -- **holds position if commands go stale** (watchdog),
instead of replaying an old motion. That hold-on-silence behaviour is exactly
what you want for teleoperation and for a policy that occasionally stalls.

For the lowest latency / jitter, run the C++ daemon (``cpp/``) instead and point
a :class:`~flexiv_control.client.RemoteRobot` at it; the contract is identical.
This Python loop is the no-root, no-build tier that already covers 100-500 Hz
use on Flexiv hardware (the robot runs the hard real-time loop internally).

The loop is the single writer; everyone else mutates a small, lock-guarded
command buffer -- the realtime-buffer pattern used by ros2_control / Polymetis.
"""

from __future__ import annotations

import contextlib
import copy
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..robot import Robot
from ..types import ControlMode, GripperCommand, RobotState, SafetyStatus, StopReason


@dataclass
class LoopStats:
    period_ms: float = 0.0
    jitter_us: float = 0.0
    missed_cycles: int = 0
    command_age_ms: float = 0.0
    last_clipped: bool = False
    last_status: SafetyStatus = SafetyStatus.OK
    last_stop_reason: StopReason = StopReason.NONE


class ReactiveServoLoop:
    def __init__(
        self,
        robot: Robot,
        control_hz: Optional[float] = None,
        write_lock: Optional[threading.Lock] = None,
    ):
        self.robot = robot
        self.backend = robot.backend
        self.filter = robot.filter
        self.hz = float(control_hz or robot.control_hz)
        self.dt = 1.0 / self.hz

        # The backend-writer lock. When embedded in a server this is the SAME
        # lock its request handlers hold around backend access, so the loop's
        # per-tick writes can never overlap a handler write -- preserving the
        # single-writer-to-hardware invariant even across the two threads.
        # Standalone (no server) the loop is already the only writer, so a no-op
        # context is used.
        self._wlock = write_lock if write_lock is not None else contextlib.nullcontext()
        self._lock = threading.Lock()
        self._cartesian = True  # else joint
        self._target_pose: Optional[np.ndarray] = None
        self._target_q: Optional[np.ndarray] = None
        self._target_gripper: Optional[GripperCommand] = None
        self._cmd_stamp = 0.0
        self._traj_iter = None  # active async traj being streamed, if any

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._latest_state: Optional[RobotState] = None
        self._stats = LoopStats()

    # -- target updates (called by teleop / MPC / RL at their own rate) ------
    def set_cartesian_target(
        self, pose: np.ndarray, gripper: Optional[GripperCommand] = None
    ) -> None:
        with self._lock:
            self._cartesian = True
            self._traj_iter = None  # a manual target preempts any async traj
            self._target_pose = np.asarray(pose, float).reshape(7).copy()
            if gripper is not None:
                self._target_gripper = gripper
            self._cmd_stamp = time.time()

    def set_joint_target(self, q: np.ndarray) -> None:
        with self._lock:
            self._cartesian = False
            self._traj_iter = None
            self._target_q = np.asarray(q, float).copy()
            self._cmd_stamp = time.time()

    def enqueue_trajectory(self, traj) -> None:
        """Stage a Cartesian traj for the loop to interpolate and stream while
        you compute the next one (async infer-ahead, NOT true real-time chunking:
        traj boundaries are a hard preemptive swap, not velocity-blended).

        A newly enqueued traj PREEMPTS the current one. Relative trajs are
        resolved against the live pose and sliced to ``horizon_exec`` first; the
        loop pulls one interpolated setpoint per tick and holds the last pose when
        the traj is exhausted (the watchdog still applies if you stop enqueuing).
        """
        from ..interpolation import CartesianTrajectoryInterpolator

        s = self.backend.read_state()
        c = traj.for_execution(s.tcp_pose)
        interp = CartesianTrajectoryInterpolator(
            c, s.tcp_pose, self.hz,
            max_linear_speed=self.filter.p.max_linear_speed,
            max_angular_speed=self.filter.p.max_angular_speed,
        )
        with self._lock:
            self._cartesian = True
            self._traj_iter = iter(interp)
            self._cmd_stamp = time.time()

    # -- introspection -------------------------------------------------------
    def get_state(self) -> RobotState:
        # Snapshot the loop's latest state under the lock, then stamp diagnostics
        # onto a shallow copy so a reader never mutates the object the loop thread
        # still holds (cross-thread in-place mutation of shared state).
        with self._lock:
            base = self._latest_state
            st = self._stats
        s = self.backend.read_state() if base is None else copy.copy(base)
        s.loop_period_ms = st.period_ms
        s.loop_jitter_us = st.jitter_us
        s.missed_cycles = st.missed_cycles
        s.command_latency_ms = st.command_age_ms
        s.safety_status = st.last_status
        s.stop_reason = st.last_stop_reason
        return s

    @property
    def stats(self) -> LoopStats:
        return self._stats

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        # Seed the target with the current pose so we hold on startup.
        s = self.backend.read_state()
        # The loop's default duty is a Cartesian hold/stream; backends require a
        # Cartesian motion mode before streaming (the fake enforces this exactly
        # like the real RDK), so ensure it here rather than depending on the
        # client having remembered start_cartesian_impedance(). No _wlock here:
        # the server calls start() while already holding that (non-reentrant)
        # lock, and standalone there is no other writer yet.
        if not s.control_mode.is_cartesian:
            self.robot.start_cartesian_impedance()
            s = self.backend.read_state()
        self.filter.reset(s)
        with self._lock:
            self._target_pose = s.tcp_pose.copy()
            self._target_q = s.q.copy()
            self._cmd_stamp = time.time()
        self._running = True
        self._thread = threading.Thread(target=self._run, name="flexiv-servo-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        # The loop thread is joined (no longer a writer); take the shared writer
        # lock for the final stop so it can't race a server handler's write.
        with self._wlock:
            self.backend.stop()

    def __enter__(self) -> "ReactiveServoLoop":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _ensure_stream_mode(self, state: RobotState, cartesian: bool) -> None:
        """Backends require a matching motion mode before streaming (the fake
        enforces this like the real RDK). The loop can find itself out of mode
        after a protective stop (-> IDLE) or when the target type flips between
        Cartesian and joint, so re-ensure right before streaming. Must be called
        under ``self._wlock`` (every stream site already is)."""
        if cartesian:
            if not state.control_mode.is_cartesian:
                self.robot.start_cartesian_impedance()
        else:
            if state.control_mode.is_cartesian or state.control_mode == ControlMode.IDLE:
                self.robot.start_joint_impedance()

    @property
    def alive(self) -> bool:
        """True while the writer thread is running. A dead loop must not be
        streamed into: ``servo_stream`` keeps 'succeeding' otherwise while
        nothing reaches the robot and the hold-watchdog (which IS this loop)
        is gone."""
        return self._running and self._thread is not None and self._thread.is_alive()

    # -- the loop ------------------------------------------------------------
    def _run(self) -> None:
        try:
            self._run_inner()
        except Exception:
            # The writer thread must never die silently: stop the backend
            # best-effort, mark the fault, and de-register liveness so
            # servo_stream callers get a loud error instead of a black hole.
            try:
                self.backend.stop()
            except Exception:
                pass
            self._stats.last_status = SafetyStatus.FAULT
            self._stats.last_stop_reason = StopReason.BACKEND_FAULT
            self._running = False
            raise

    def _run_inner(self) -> None:
        next_tick = time.perf_counter()
        prev_tick = next_tick
        while self._running:
            state = self.backend.read_state()
            self._latest_state = state
            # Age of the state data itself (fresh synchronous reads are ~0 ms);
            # the opt-in state-age watchdog protective-stops on a stale backend.
            state_age_ms = (time.time() - state.stamp) * 1000.0

            # Advance an enqueued traj: pull one interpolated setpoint per tick
            # into the target, refreshing the command stamp so it doesn't go
            # stale mid-traj. When the traj is exhausted, leave the last target
            # in place (hold) until the next enqueue / manual target.
            with self._lock:
                citer = self._traj_iter
            if citer is not None:
                try:
                    pose, grip = next(citer)
                    with self._lock:
                        self._cartesian = True
                        self._target_pose = pose
                        if grip is not None:
                            self._target_gripper = grip
                        self._cmd_stamp = time.time()
                except StopIteration:
                    with self._lock:
                        self._traj_iter = None

            with self._lock:
                cartesian = self._cartesian
                target_pose = None if self._target_pose is None else self._target_pose.copy()
                target_q = None if self._target_q is None else self._target_q.copy()
                gripper = self._target_gripper
                age = time.time() - self._cmd_stamp

            self._stats.command_age_ms = age * 1000.0

            # Robot-side fault gate, checked EVERY tick. A collision reflex /
            # protective stop / E-stop on the robot is invisible to the host-side
            # safety filter (which only checks geometry against the commanded
            # setpoint), so if the backend reports a fault we must stop streaming
            # immediately rather than keep issuing setpoints into a faulted arm.
            faulted = self.backend.in_fault()

            # Read the watchdog from the LIVE profile each tick so a mid-run
            # set_safety_profile() swap applies both the timeout and the flag
            # consistently (they used to diverge: flag live, timeout cached).
            # The whole decide-and-write section runs under the shared writer
            # lock so it can never overlap a server handler's backend write (or a
            # concurrent set_safety_profile() swap of the very profile we read).
            with self._wlock:
                p = self.filter.p
                stale = age > (p.command_timeout_ms / 1000.0)
                state_stale = not self.filter.check_state_age(state_age_ms).ok
                if faulted:
                    self.backend.stop()
                    self._stats.last_status = SafetyStatus.FAULT
                    self._stats.last_stop_reason = StopReason.BACKEND_FAULT
                elif state_stale:
                    # State data too old to trust for position -> protective stop
                    # (opt-in via stop_on_state_stale).
                    self.backend.stop()
                    self._stats.last_status = SafetyStatus.STOPPED
                    self._stats.last_stop_reason = StopReason.STALE_STATE
                elif stale and p.stop_on_stale_command:
                    if p.hold_timeout_ms is not None and age > (p.hold_timeout_ms / 1000.0):
                        # Held too long -> escalate the hold to a protective stop.
                        self.backend.stop()
                        self._stats.last_status = SafetyStatus.STOPPED
                        self._stats.last_stop_reason = StopReason.STALE_COMMAND
                    else:
                        # Hold position: re-issue the *current measured* pose, don't
                        # keep tracking an old setpoint. Re-anchor the filter so the
                        # next live command is referenced to where we actually are.
                        self._stats.last_status = SafetyStatus.HOLDING
                        self.filter.reset(state)
                        self._ensure_stream_mode(state, cartesian)
                        if cartesian:
                            self.backend.stream_cartesian(state.tcp_pose)
                        else:
                            self.backend.stream_joint(state.q)
                else:
                    if cartesian and target_pose is not None:
                        sr = self.filter.filter_cartesian(target_pose, state)
                        if sr.ok:
                            self._ensure_stream_mode(state, cartesian=True)
                            self.backend.stream_cartesian(sr.pose)
                            if gripper is not None:
                                self.backend.move_gripper(gripper)
                                with self._lock:
                                    self._target_gripper = None
                            self._stats.last_clipped = sr.clipped
                            self._stats.last_status = (
                                SafetyStatus.CLIPPED if sr.clipped else SafetyStatus.OK
                            )
                            # Surface WHICH limit clipped (workspace/jump/speed),
                            # not just that something did; NONE when unclipped.
                            self._stats.last_stop_reason = sr.primary_clip_reason
                        else:
                            self.backend.stop()
                            self._stats.last_status = SafetyStatus.STOPPED
                            self._stats.last_stop_reason = sr.reason
                    elif not cartesian and target_q is not None:
                        sr = self.filter.filter_joint(target_q, state)
                        if sr.ok:
                            self._ensure_stream_mode(state, cartesian=False)
                            self.backend.stream_joint(sr.q)
                            self._stats.last_clipped = sr.clipped
                            self._stats.last_status = (
                                SafetyStatus.CLIPPED if sr.clipped else SafetyStatus.OK
                            )
                            self._stats.last_stop_reason = sr.primary_clip_reason
                        else:
                            self.backend.stop()
                            self._stats.last_status = SafetyStatus.STOPPED
                            self._stats.last_stop_reason = sr.reason

            # diagnostics
            now = time.perf_counter()
            self._stats.period_ms = (now - prev_tick) * 1000.0
            jitter = abs((now - prev_tick) - self.dt)
            self._stats.jitter_us = jitter * 1e6
            prev_tick = now

            next_tick += self.dt
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                # we overran the budget
                self._stats.missed_cycles += 1
                next_tick = time.perf_counter()
