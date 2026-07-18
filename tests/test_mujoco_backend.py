"""MuJoCo backend integration tests.

Runs only when ``mujoco`` is installed AND a Rizon MJCF is available via the
``FLEXIV_RIZON_MJCF`` env var, e.g. point it at the official model::

    pip install mujoco
    git clone https://github.com/google-deepmind/mujoco_menagerie
    FLEXIV_RIZON_MJCF=mujoco_menagerie/flexiv_rizon4s/scene.xml pytest tests/test_mujoco_backend.py

Skipped in CI (where neither mujoco nor the model is present)."""

from __future__ import annotations

import os

import numpy as np
import pytest

_MJCF = os.environ.get("FLEXIV_RIZON_MJCF")
pytestmark = pytest.mark.skipif(
    not _MJCF, reason="set FLEXIV_RIZON_MJCF to a Rizon MJCF to run the mujoco tests"
)


def _robot():
    pytest.importorskip("mujoco")
    from flexiv_control import Robot, RobotConfig

    r = Robot(RobotConfig(backend="mujoco", model_path=_MJCF, control_hz=100.0))
    r.connect()
    r.home()
    return r


def test_read_state_shapes_and_unit_quat():
    r = _robot()
    s = r.get_state()
    assert s.q.shape == (7,)
    assert s.dq.shape == (7,)
    assert s.tcp_pose.shape == (7,)
    assert abs(float(np.linalg.norm(s.tcp_pose[3:7])) - 1.0) < 1e-6
    r.disconnect()


def test_gravity_compensated_hold():
    # With computed-torque gravity comp the arm holds its pose (no sag).
    r = _robot()
    r.start_cartesian_impedance()
    s0 = r.get_state()
    pose = s0.tcp_pose.copy()
    from flexiv_control import CartesianTrajectory

    # command the current pose for ~0.5 s; the TCP must not drift
    traj = CartesianTrajectory.from_pose_array(
        np.concatenate([pose, [1.0, 50]])[None, :], safety_profile="free_space_fast"
    )
    r.execute_cartesian_trajectory(traj)
    s1 = r.get_state()
    assert np.linalg.norm(s1.tcp_pose[:3] - pose[:3]) < 5e-3
    r.disconnect()


def test_cartesian_ik_tracks_target():
    from flexiv_control import CartesianTrajectory

    r = _robot()
    r.start_cartesian_impedance()
    s0 = r.get_state()
    tgt = s0.tcp_pose.copy()
    tgt[0] += 0.06
    tgt[2] -= 0.06
    traj = CartesianTrajectory.from_pose_array(
        np.concatenate([tgt, [1.0, 80]])[None, :], safety_profile="free_space_fast"
    )
    res = r.execute_cartesian_trajectory(traj)
    s1 = r.get_state()
    assert res.success
    assert np.linalg.norm(s1.tcp_pose[:3] - tgt[:3]) < 0.015  # IK converged < 1.5 cm
    r.disconnect()


def test_joint_move_tracks():
    r = _robot()
    q_tgt = np.array([0.1, -0.6, 0.0, 1.5, 0.0, 0.9, 0.0])
    r.move_joint(q_tgt, duration=1.0)
    s = r.get_state()
    assert np.max(np.abs(s.q - q_tgt)) < 0.02
    r.disconnect()
