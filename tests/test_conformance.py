"""Conformance regression tests: gym registration, HIL-SERL intervention,
the backend contract, and the torque gate added in the standard-gaps pass."""

from __future__ import annotations

import numpy as np
import pytest

from flexiv_control.backends import get_backend
from flexiv_control.teleop.spacemouse import (
    ScriptedSpaceMouseSource,
    SpaceMouseState,
    SpaceMouseTeleop,
)
from flexiv_control.types import ControlMode, RobotState


def _teleop(source_fn=None):
    return SpaceMouseTeleop(robot=object(), source=ScriptedSpaceMouseSource(source_fn))


# -- backend contract --------------------------------------------------------
def test_backend_contract_fake():
    b = get_backend("fake")
    b.connect()
    s = b.read_state()
    assert isinstance(s, RobotState)
    assert s.q.shape == (7,)
    assert s.tcp_pose.shape == (7,)
    assert s.wrench.shape == (6,)
    # stream_* accept correctly-shaped setpoints once a matching motion mode is
    # active (the fake now enforces the real backend's mode precondition).
    b.set_mode(ControlMode.NRT_CARTESIAN_MOTION_FORCE)
    b.stream_cartesian(np.array([0.45, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0]))
    b.set_mode(ControlMode.NRT_JOINT_IMPEDANCE)
    b.stream_joint(np.zeros(7))


def test_mujoco_backend_fails_loudly_without_model():
    # Constructible without a model, but connect() fails LOUDLY (no fake success):
    # ValueError if mujoco is installed but no model_path was given; ImportError
    # if mujoco itself is not installed.
    mj = get_backend("mujoco")  # constructible (model_path defaults to None)
    with pytest.raises((ValueError, ImportError, NotImplementedError)):
        mj.connect()


# -- torque gate -------------------------------------------------------------
def test_rt_joint_torque_is_gated_off():
    flexivrdk = pytest.importorskip("flexivrdk")
    from flexiv_control.backends.flexiv_rdk import FlexivRdkBackend

    b = FlexivRdkBackend("X")  # allow_torque defaults to False
    b._robot = type("R", (), {"SwitchMode": staticmethod(lambda *_a, **_k: None)})()
    with pytest.raises(RuntimeError):
        b.set_mode(ControlMode.RT_JOINT_TORQUE)
    # Opt-in lets it through only when the installed RDK exposes that mode.
    # Older supported RDK builds intentionally fail loudly instead of
    # pretending that an unavailable real-time mode was selected.
    b2 = FlexivRdkBackend("X", allow_torque=True)
    b2._robot = type("R", (), {"SwitchMode": staticmethod(lambda *_a, **_k: None)})()
    if hasattr(flexivrdk.Mode, "RT_JOINT_TORQUE"):
        b2.set_mode(ControlMode.RT_JOINT_TORQUE)
    else:
        with pytest.raises(RuntimeError, match="does not expose"):
            b2.set_mode(ControlMode.RT_JOINT_TORQUE)


# -- SpaceMouse intervention (the previously-untested HIL-SERL path) ----------
def test_intervention_overrides_when_moving():
    policy = np.zeros(7, np.float32)
    a, intervened = _teleop().intervention(policy)  # default source: moving + deadman held
    assert intervened is True
    assert a.shape == (7,)
    assert np.all(a >= -1.0) and np.all(a <= 1.0)
    assert not np.allclose(a[:3], 0.0)  # human translation present


def test_intervention_passes_through_when_idle():
    policy = np.arange(7, dtype=np.float32) / 7.0
    idle = _teleop(lambda t: SpaceMouseState(buttons=[1, 0]))  # deadman held, no motion
    a, intervened = idle.intervention(policy)
    assert intervened is False
    assert np.allclose(a, policy)


def test_intervention_passes_through_when_deadman_released():
    policy = np.ones(7, np.float32)
    released = _teleop(
        lambda t: SpaceMouseState(translation=np.ones(3), buttons=[0, 0])  # moving but no deadman
    )
    a, intervened = released.intervention(policy)
    assert intervened is False
    assert np.allclose(a, policy)


# -- gym registration + intervention wrapper ---------------------------------
def test_gym_make_and_intervention_wrapper():
    gym = pytest.importorskip("gymnasium")
    from flexiv_control import RobotConfig
    from flexiv_control.envs import SpacemouseIntervention

    env = gym.make("FlexivReal-v0", config=RobotConfig(backend="fake"))
    env.reset(seed=0)
    teleop = _teleop()  # moving -> will override
    wrapped = SpacemouseIntervention(env, teleop)
    _obs, _r, _term, _trunc, info = wrapped.step(np.zeros(7, np.float32))
    assert info.get("intervened") is True
    assert "intervene_action" in info
    assert np.asarray(info["intervene_action"]).shape == (7,)
    env.close()
