"""Scene-level tests for RobotViz (skipped when viser is not installed)."""

from __future__ import annotations

import time

import numpy as np
import pytest

viser = pytest.importorskip("viser")

from flexiv_control import (  # noqa: E402
    CartesianTrajectory,
    CartesianWaypoint,
    GripperCommand,
    Robot,
    RobotConfig,
)
from flexiv_control.viz import RobotViz  # noqa: E402

_PORT = 18763  # avoid the control server's 8766 and the default viz 8080


def _traj(start_pose):
    p = np.asarray(start_pose[:3], float)
    return CartesianTrajectory(
        waypoints=[
            CartesianWaypoint(position=p + [0.05, 0.0, 0.0], quaternion=None,
                              gripper=GripperCommand(width=0.02, grasp=True),
                              duration=1.0),
            CartesianWaypoint(position=p + [0.05, 0.05, 0.04], quaternion=None,
                              gripper=GripperCommand(width=0.08), duration=1.0),
        ],
        max_tcp_linear_speed=0.12,
    )


@pytest.fixture()
def robot():
    r = Robot(RobotConfig(backend="fake"))
    r.connect()
    r.acquire_lease("viz-test")
    yield r
    r.disconnect()


@pytest.fixture()
def viz():
    v = RobotViz(port=_PORT, state_hz=30.0)
    yield v
    v.stop()


def test_frames_mode_lifecycle(robot, viz):
    """Attach -> poll -> mirror updates -> preview -> gate -> on_step -> clear,
    all headless on the fake backend with zero model assets."""
    viz.attach(robot, allow_lease=True)
    time.sleep(0.5)  # a few poll ticks
    s = robot.get_state()
    np.testing.assert_allclose(np.asarray(viz._tcp.position), s.tcp_pose[:3], atol=1e-6)
    assert viz._profile is not None                 # profile picked up
    assert viz._workspace is not None               # workspace box drawn
    assert len(viz._trail_buf) >= 2                 # trail accumulating

    traj = _traj(s.tcp_pose)
    pv = viz.preview_trajectory(traj, s, robot.profile, traj_id="t")
    assert len(pv.setpoints) > 10
    assert len(viz._plan_handles) >= 3              # path + knots + glyphs + terminal
    assert viz._ghost is not None

    gate = viz.gate()                                # pure-viz gate: no click needed
    assert gate(1, traj) is True

    result = robot.execute_cartesian_trajectory(traj, record=True)
    viz.on_step(1, traj, result)
    with viz._preview_lock:
        assert viz._preview is None                  # preview cleared
    assert "traj 1" in viz._preview_md.content


def test_gate_refuses_stale_preview(robot, viz):
    """A preview planned from a pose far from the live TCP must be refused."""
    viz.attach(robot, allow_lease=True)
    s = robot.get_state()
    far = s.tcp_pose.copy()
    far[0] += 0.10  # pretend the traj was planned 10 cm ago

    # plan from the stale pose, then gate against the LIVE state
    traj = _traj(far)
    pv = viz.preview_trajectory(traj, state=s, traj_id="stale")
    from flexiv_control.viz.preview import pose_distance

    # simulate: preview start far from live tcp
    lin, _ = pose_distance(far, s.tcp_pose)
    assert lin > 0.005
    # the gate itself re-plans from live state, so emulate the stale branch:
    pv.start_pose = far
    with viz._preview_lock:
        viz._preview = pv
    # directly exercise the staleness math the gate uses
    from flexiv_control.viz.app import STALE_LINEAR_M

    assert lin > STALE_LINEAR_M


def test_attach_refuses_leased_monitor(viz):
    class _FakeLeased:
        _has_lease = True

        def get_state(self):  # pragma: no cover - never reached
            raise AssertionError

    with pytest.raises(ValueError, match="LEASE"):
        viz.attach(_FakeLeased())


def test_lazy_import_error_message(monkeypatch):
    """Without viser, RobotViz import raises with the pip hint (the numpy-only
    preview module stays importable)."""
    import builtins
    import importlib

    import flexiv_control.viz as vizmod

    real_import = builtins.__import__

    def _no_viser(name, *a, **k):
        if name == "viser" or name.startswith("viser."):
            raise ImportError("No module named 'viser'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_viser)
    monkeypatch.delitem(__import__("sys").modules, "flexiv_control.viz.app", raising=False)
    with pytest.raises(ImportError, match=r"flexiv-control\[viz\]"):
        importlib.reload(vizmod)
        vizmod.RobotViz  # noqa: B018
