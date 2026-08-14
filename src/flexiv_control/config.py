"""Configuration loading.

Configs are plain YAML so they can be version-controlled and diffed. We ship a
set of sane defaults under ``configs/`` at the repo root; users override by
passing their own path. Loading is forgiving: PyYAML is optional and we fall
back to a tiny built-in parser for the simple files we ship, so the core has
zero hard third-party deps beyond numpy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .safety import SafetyProfile


def _config_search_dirs() -> list:
    """Where to look for shipped configs, in priority order.

    Works for an editable/source checkout (the documented install path) and for
    a packaged copy, and honours the ``FLEXIV_CONTROL_CONFIGS`` env override so a
    lab can point at its own config repo without touching code.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    dirs = []
    env = os.environ.get("FLEXIV_CONTROL_CONFIGS")
    if env:
        dirs.append(os.path.abspath(env))
    # repo-root configs/ (src/flexiv_control/config.py -> repo root is two dirs up of src)
    dirs.append(os.path.abspath(os.path.join(here, "..", "..", "configs")))
    # configs/ packaged next to the source (wheel installs)
    dirs.append(os.path.join(here, "configs"))
    return dirs


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        text = f.read()
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except Exception:
        # Minimal fallback so the shipped configs load without PyYAML.
        return _mini_yaml(text)


def _mini_yaml(text: str) -> dict:
    """Very small YAML subset parser for the flat configs we ship.

    Supports nested maps via indentation, scalars, and inline ``[a, b, c]``
    lists. Not general; just enough for our config files. Install PyYAML for
    anything richer.
    """
    root: dict = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        key, _, val = raw.strip().partition(":")
        key = key.strip()
        val = val.split("#", 1)[0].strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if val == "":
            node: dict = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = _parse_scalar(val)
    return root


def _parse_scalar(v: str):
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(x.strip()) for x in inner.split(",")]
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v.strip('"').strip("'")


@dataclass
class RobotConfig:
    robot_id: str = "rizon4s_lab"
    backend: str = "fake"
    robot_sn: str = ""
    gripper_name: Optional[str] = None
    n_joints: int = 7
    control_hz: float = 100.0
    # Direct joint torque is an expert/research interface.  It is disabled by
    # default and must be enabled by the robot deployment config before the
    # facade can enter RT_JOINT_TORQUE or emit one torque command.
    allow_joint_torque: bool = False
    default_safety_profile: str = "tabletop_safe"
    q_home: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -0.7, 0.0, 1.6, 0.0, 0.9, 0.0], float)
    )
    # Gripper opening (metres) that belongs to the canonical home posture, so
    # ``Robot.home()``/``go_home_safe()`` can restore the FULL recorded posture
    # (joints AND gripper). None = leave the gripper alone.
    gripper_home_width: Optional[float] = None
    # MuJoCo backend (sim / real2sim2real): path to a Rizon MJCF, the control
    # substep duration, and an optional TCP site name.
    model_path: Optional[str] = None
    # MuJoCo control substep duration. None = match the control rate (1/control_hz)
    # so a traj plays back at the SAME speed in sim as the loop streams it; the
    # old fixed 0.02 s decoupled sim time from control_hz (trajs ran too fast/slow).
    control_dt: Optional[float] = None
    mujoco_tcp_site: Optional[str] = None
    mujoco_gripper_actuator: Optional[str] = None
    mujoco_gripper_width_scale: float = 1.0
    mujoco_gripper_width_offset: float = 0.0

    @classmethod
    def load(cls, path_or_name: str) -> "RobotConfig":
        path = _resolve(path_or_name, "robots")
        d = _load_yaml(path)
        c = cls()
        c.robot_id = d.get("robot_id", c.robot_id)
        c.backend = d.get("backend", c.backend)
        c.robot_sn = str(d.get("robot_sn", c.robot_sn))
        c.gripper_name = d.get("gripper_name", c.gripper_name)
        c.n_joints = int(d.get("n_joints", c.n_joints))
        c.control_hz = float(d.get("control_hz", c.control_hz))
        c.allow_joint_torque = bool(
            d.get("allow_joint_torque", c.allow_joint_torque)
        )
        c.default_safety_profile = d.get("default_safety_profile", c.default_safety_profile)
        if "q_home" in d:
            c.q_home = np.asarray(d["q_home"], float)
        if "gripper_home_width" in d and d["gripper_home_width"] is not None:
            c.gripper_home_width = float(d["gripper_home_width"])
        c.model_path = d.get("model_path", c.model_path)
        if "control_dt" in d:
            c.control_dt = float(d["control_dt"])
        c.mujoco_tcp_site = d.get("mujoco_tcp_site", c.mujoco_tcp_site)
        c.mujoco_gripper_actuator = d.get("mujoco_gripper_actuator", c.mujoco_gripper_actuator)
        c.mujoco_gripper_width_scale = float(
            d.get("mujoco_gripper_width_scale", c.mujoco_gripper_width_scale)
        )
        c.mujoco_gripper_width_offset = float(
            d.get("mujoco_gripper_width_offset", c.mujoco_gripper_width_offset)
        )
        return c


def load_safety_profile(path_or_name: str) -> SafetyProfile:
    path = _resolve(path_or_name, "safety")
    return SafetyProfile.from_dict(_load_yaml(path))


def _resolve(path_or_name: str, subdir: str) -> str:
    """Accept either a filesystem path or a bare profile name."""
    if os.path.isfile(path_or_name):
        return path_or_name
    tried = []
    for base in _config_search_dirs():
        candidate = os.path.join(base, subdir, path_or_name)
        for p in (candidate, candidate + ".yaml", candidate + ".yml"):
            if os.path.isfile(p):
                return p
        tried.append(candidate)
    raise FileNotFoundError(
        f"config not found: {path_or_name!r} (looked in: {', '.join(tried)})"
    )
