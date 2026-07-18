"""Tests for async traj streaming (ReactiveServoLoop.enqueue_trajectory) and the
receding-horizon runner (the VLA / MPC policy-server seam)."""

from __future__ import annotations

import time

import numpy as np

from flexiv_control import (
    CartesianTrajectory,
    RecedingHorizonRunner,
    Robot,
    RobotConfig,
)
from flexiv_control.server import ReactiveServoLoop


def _robot():
    r = Robot(RobotConfig(backend="fake", control_hz=200.0))
    r.connect()
    r.start_cartesian_impedance()
    return r


def _abs_chunk(pose, n=60):
    return CartesianTrajectory.from_pose_array(
        np.concatenate([pose, [1.0, n]])[None, :]
    )


def test_enqueue_chunk_streams_and_holds():
    r = _robot()
    with ReactiveServoLoop(r) as loop:
        s0 = loop.get_state()
        tgt = s0.tcp_pose.copy()
        tgt[0] += 0.05
        loop.enqueue_trajectory(_abs_chunk(tgt, n=80))
        time.sleep(1.0)  # let the loop interpolate + stream the traj
        s1 = loop.get_state()
        assert s1.tcp_pose[0] - s0.tcp_pose[0] > 0.03  # moved toward target
        # after exhaustion the loop HOLDS the last pose (no drift)
        held = loop.get_state().tcp_pose.copy()
        time.sleep(0.3)
        assert np.linalg.norm(loop.get_state().tcp_pose[:3] - held[:3]) < 0.01


def test_enqueue_chunk_preempts():
    r = _robot()
    with ReactiveServoLoop(r) as loop:
        s0 = loop.get_state()
        far = s0.tcp_pose.copy()
        far[0] += 0.12
        loop.enqueue_trajectory(_abs_chunk(far, n=200))  # slow, long
        time.sleep(0.15)
        loop.enqueue_trajectory(_abs_chunk(s0.tcp_pose, n=80))  # preempt back toward start
        time.sleep(0.8)
        # preempted: never reached the far target
        assert loop.get_state().tcp_pose[0] < far[0] - 0.02


def test_receding_horizon_runner_blocking():
    r = _robot()
    calls = {"n": 0}

    def policy(obs):
        calls["n"] += 1
        if calls["n"] > 3:
            return None
        tgt = obs.tcp_pose.copy()
        tgt[0] += 0.02  # step +2 cm in x each cycle
        return _abs_chunk(tgt, n=40)

    s0 = r.get_state()
    runner = RecedingHorizonRunner(r, policy, max_steps=10)
    steps = runner.run()
    s1 = r.get_state()
    assert steps == 3  # policy returned None on the 4th call
    assert calls["n"] == 4
    assert s1.tcp_pose[0] - s0.tcp_pose[0] > 0.04  # progressed ~0.06


def test_receding_horizon_runner_streaming():
    r = _robot()
    calls = {"n": 0}

    def policy(obs):
        calls["n"] += 1
        tgt = obs.tcp_pose.copy()
        tgt[0] += 0.015
        return _abs_chunk(tgt, n=40)

    with ReactiveServoLoop(r) as loop:
        s0 = loop.get_state()
        runner = RecedingHorizonRunner(r, policy, max_steps=4)
        n = runner.run_streaming(loop, replan_hz=10.0)
        time.sleep(0.5)  # let the last enqueued traj play
        assert n == 4
        assert loop.get_state().tcp_pose[0] - s0.tcp_pose[0] > 0.01
