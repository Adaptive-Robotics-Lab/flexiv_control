"""MuJoCo backend for sim and real-to-sim-to-real.

The *same* ``CartesianTrajectory`` / ``CartesianDelta`` action and the *same*
``RobotState`` observation drive sim and the real Rizon behind one backend
switch -- the Polymetis (pybullet->real) / Deoxys (robosuite->real) sim=real
property, now realized for Flexiv. Cartesian setpoints are tracked with a
damped-least-squares Jacobian IK step per control tick; joint setpoints drive
the model's position actuators.

Model
-----
Pass a Rizon MJCF via ``model_path``. The official DeepMind
``mujoco_menagerie/flexiv_rizon4s/scene.xml`` works out of the box (7 hinge
joints + 7 position actuators); point ``model_path`` at your lab's calibrated
MJCF (with gripper / tool / table) for real2sim2real. The TCP frame defaults to a
site named ``tcp``/``flange``/``attachment_site`` if present, else the last site
in the model; override with ``tcp_site=``.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .. import transforms as T
from ..types import ControlMode, GripperCommand, RobotState
from .base import RobotBackend


class MujocoBackend(RobotBackend):
    def __init__(
        self,
        model_path: Optional[str] = None,
        n_joints: int = 7,
        control_dt: float = 0.02,
        *,
        tcp_site: Optional[str] = None,
        gripper_actuator: Optional[str] = None,
        gripper_width_scale: float = 1.0,
        gripper_width_offset: float = 0.0,
        ik_gain: float = 1.0,
        ik_damping: float = 1e-3,
        gravity_comp: bool = True,
    ):
        self.model_path = model_path
        self.n_joints = int(n_joints)
        self.dt = float(control_dt)
        self._tcp_site_name = tcp_site
        self._gripper_actuator_name = gripper_actuator
        # Gripper actuator ctrl = scale * width_m + offset. Identity by default;
        # for the GN01 4-bar the actuated knuckle is 9.404 * width - 0.155.
        self._grip_scale = float(gripper_width_scale)
        self._grip_offset = float(gripper_width_offset)
        self.ik_gain = float(ik_gain)
        self.ik_damping = float(ik_damping)
        # Cancel gravity + Coriolis each substep (computed-torque feedforward) so
        # the position actuators hold/track cleanly -- the real Rizon is likewise
        # gravity-compensated by its internal impedance loop. Without this the
        # P-type model actuators sag several cm under gravity.
        self.gravity_comp = bool(gravity_comp)

        self._connected = False
        self._mode = ControlMode.IDLE
        self._mj = None
        self._m = None
        self._d = None
        self._qadr: Optional[np.ndarray] = None   # qpos addresses of the arm joints
        self._vadr: Optional[np.ndarray] = None   # dof addresses of the arm joints
        self._act: List[int] = []                 # arm actuator ids
        self._site = -1                           # TCP site id
        self._grip_act = -1                       # optional gripper actuator id

    # -- lifecycle ----------------------------------------------------------
    def connect(self) -> None:
        try:
            import mujoco
        except Exception as exc:  # pragma: no cover - depends on env
            raise ImportError("MujocoBackend needs `pip install mujoco`") from exc
        if not self.model_path:
            raise ValueError(
                "MujocoBackend requires `model_path` (a Rizon MJCF). Use the "
                "official model from github.com/google-deepmind/mujoco_menagerie "
                "(flexiv_rizon4s/scene.xml), or your lab's calibrated MJCF."
            )
        self._mj = mujoco
        self._m = mujoco.MjModel.from_xml_path(self.model_path)
        self._d = mujoco.MjData(self._m)
        self._resolve_ids()
        mujoco.mj_forward(self._m, self._d)
        self._connected = True

    def _resolve_ids(self) -> None:
        mj, m = self._mj, self._m
        n = min(self.n_joints, m.nu)
        self._act = list(range(n))
        qadr, vadr = [], []
        for a in self._act:
            jid = int(m.actuator_trnid[a, 0])   # actuator -> target joint
            qadr.append(int(m.jnt_qposadr[jid]))
            vadr.append(int(m.jnt_dofadr[jid]))
        self._qadr = np.asarray(qadr, int)
        self._vadr = np.asarray(vadr, int)

        # TCP site: explicit name -> common names -> last site in the model.
        self._site = -1
        if self._tcp_site_name:
            self._site = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, self._tcp_site_name)
            if self._site < 0:
                raise ValueError(f"tcp_site {self._tcp_site_name!r} not found in model")
        if self._site < 0:
            for nm in ("tcp", "flange", "attachment_site", "ee", "gripper", "tool0"):
                sid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, nm)
                if sid >= 0:
                    self._site = sid
                    break
        if self._site < 0 and m.nsite > 0:
            self._site = m.nsite - 1
        if self._site < 0:
            raise ValueError("model has no site usable as TCP; pass tcp_site=")

        self._grip_act = -1
        if self._gripper_actuator_name:
            self._grip_act = mj.mj_name2id(
                m, mj.mjtObj.mjOBJ_ACTUATOR, self._gripper_actuator_name
            )

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # -- state --------------------------------------------------------------
    def _site_pose(self):
        mj, d = self._mj, self._d
        pos = np.array(d.site_xpos[self._site], float)
        quat = np.zeros(4)
        mj.mju_mat2Quat(quat, d.site_xmat[self._site])  # (w, x, y, z)
        return pos, quat

    def read_state(self) -> RobotState:
        mj, m, d = self._mj, self._m, self._d
        mj.mj_forward(m, d)  # refresh site_xpos/jacobians after the last integration
        q = np.array(d.qpos[self._qadr], float)
        dq = np.array(d.qvel[self._vadr], float)
        tau = np.array(d.actuator_force[self._act], float) if m.nu else np.zeros(self.n_joints)
        pos, quat = self._site_pose()
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mj.mj_jacSite(m, d, jacp, jacr, self._site)
        v = jacp[:, self._vadr] @ dq
        w = jacr[:, self._vadr] @ dq
        # invert the actuator mapping back to a width in metres
        gw = (
            (float(d.ctrl[self._grip_act]) - self._grip_offset) / self._grip_scale
            if self._grip_act >= 0
            else 0.0
        )
        return RobotState(
            stamp=float(d.time),
            q=q, dq=dq, tau=tau,
            tcp_pose=np.concatenate([pos, quat]),
            tcp_vel=np.concatenate([v, w]),
            wrench=np.zeros(6),
            gripper_width=gw,
            control_mode=self._mode,
        )

    def set_mode(self, mode: ControlMode, **kwargs) -> None:  # noqa: D401
        # The sim tracks position/Cartesian setpoints; impedance/force params are
        # accepted (for contract parity) but not modelled here.
        self._mode = mode

    # -- streaming ----------------------------------------------------------
    def _substeps(self) -> int:
        return max(1, int(round(self.dt / float(self._m.opt.timestep))))

    def _step_n(self, n: int) -> None:
        mj, m, d = self._mj, self._m, self._d
        if self.gravity_comp:
            for _ in range(n):
                mj.mj_step1(m, d)                  # position/velocity quantities -> qfrc_bias
                d.qfrc_applied[:] = d.qfrc_bias    # cancel gravity + Coriolis
                mj.mj_step2(m, d)                  # integrate
        else:
            for _ in range(n):
                mj.mj_step(m, d)

    def stream_joint(self, q: np.ndarray) -> None:
        m, d = self._m, self._d
        q = np.asarray(q, float).reshape(-1)[: self.n_joints]
        lo = m.actuator_ctrlrange[self._act, 0]
        hi = m.actuator_ctrlrange[self._act, 1]
        q = np.where(hi > lo, np.clip(q, lo, hi), q)  # clip only where ctrl is limited
        d.ctrl[self._act] = q
        self._step_n(self._substeps())

    def stream_joint_torque(self, tau: np.ndarray) -> None:
        if self._mode != ControlMode.RT_JOINT_TORQUE:
            raise RuntimeError(
                "stream_joint_torque requires RT_JOINT_TORQUE mode"
            )
        m, d = self._m, self._d
        command = np.asarray(tau, dtype=float).reshape(self.n_joints)
        lo = m.actuator_ctrlrange[self._act, 0]
        hi = m.actuator_ctrlrange[self._act, 1]
        command = np.where(hi > lo, np.clip(command, lo, hi), command)
        d.ctrl[self._act] = command
        self._step_n(self._substeps())

    def stream_cartesian(self, pose: np.ndarray, wrench: Optional[np.ndarray] = None) -> None:
        mj, m, d = self._mj, self._m, self._d
        pose = np.asarray(pose, float).reshape(7)
        cur_pos, cur_quat = self._site_pose()
        err = np.empty(6)
        err[:3] = pose[:3] - cur_pos
        # world-frame orientation error as a rotation vector
        err[3:] = T.quat_to_rotvec(T.quat_mul(pose[3:7], T.quat_conj(cur_quat)))

        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mj.mj_jacSite(m, d, jacp, jacr, self._site)
        J = np.vstack([jacp[:, self._vadr], jacr[:, self._vadr]])  # 6 x n
        lam2 = self.ik_damping ** 2
        dq = self.ik_gain * (J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), err))
        q = np.array(d.qpos[self._qadr], float) + dq
        self.stream_joint(q)

    def move_gripper(self, cmd: GripperCommand) -> None:
        if self._grip_act < 0:
            return
        m, d = self._m, self._d
        # grasp=True mirrors the RDK semantics (Gripper.Grasp ignores width and
        # closes until contact): command full close and let contact physics /
        # the actuator force limit stop the fingers on the object. A width move
        # tracks the commanded width as before.
        width_cmd = 0.0 if cmd.grasp else float(cmd.width)
        ctrl = self._grip_scale * width_cmd + self._grip_offset
        lo, hi = m.actuator_ctrlrange[self._grip_act]
        d.ctrl[self._grip_act] = min(max(ctrl, lo), hi) if hi > lo else ctrl
        # Advance physics so a standalone command_gripper() visibly actuates the
        # fingers, matching the real backend's immediate Gripper.Move semantics
        # (otherwise the fingers only move on the next stream_* tick).
        self._step_n(self._substeps())

    # -- safety / motion ----------------------------------------------------
    def stop(self) -> None:
        self._mode = ControlMode.IDLE

    def home(self, q_home: Optional[np.ndarray] = None) -> None:
        if q_home is not None and self._d is not None:
            self._d.qpos[self._qadr] = np.asarray(q_home, float).reshape(-1)[: self.n_joints]
            self._d.qvel[self._vadr] = 0.0
            self._mj.mj_forward(self._m, self._d)
