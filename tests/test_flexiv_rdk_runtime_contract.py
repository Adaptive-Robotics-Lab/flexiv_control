from types import SimpleNamespace

import numpy as np
import pytest

from flexiv_control.backends.flexiv_rdk import FlexivRdkBackend
from flexiv_control.types import ControlMode, GripperCommand


class _Robot:
    def __init__(self, states):
        self._states = states

    def states(self):
        return self._states

    def fault(self):
        return False


class _Gripper:
    def __init__(self, states=None):
        self._states = states
        self.calls = []

    def states(self):
        return self._states

    def Move(self, width, velocity, force):
        self.calls.append(("move", width, velocity, force))

    def Grasp(self, force):
        self.calls.append(("grasp", force))


def _backend(*, robot_states=None, gripper_states=None):
    backend = object.__new__(FlexivRdkBackend)
    backend.robot_sn = "Rizon4s-test"
    backend.n_joints = 7
    backend._gripper_name = "GN01"
    backend._allow_torque = False
    if robot_states is None:
        robot_states = SimpleNamespace(
            q=np.zeros(7),
            dq=np.zeros(7),
            tau=np.zeros(7),
            tcp_pose=np.array([0, 0, 0, 1, 0, 0, 0], float),
            tcp_vel=np.zeros(6),
            ext_wrench_in_tcp=np.zeros(6),
        )
    backend._robot = _Robot(robot_states)
    backend._gripper = _Gripper(
        gripper_states
        if gripper_states is not None
        else SimpleNamespace(width=0.08, force=0.0, is_moving=False)
    )
    backend._mode = ControlMode.IDLE
    backend._connected = True
    backend._runtime_info = {}
    backend._gripper_limits = {
        "source": "flexivrdk.Gripper.params",
        "device_name": "GN01",
        "min_width_m": 0.0,
        "max_width_m": 0.085,
        "min_velocity_m_s": 0.005,
        "max_velocity_m_s": 0.2,
        "min_force_n": 1.0,
        "max_force_n": 80.0,
    }
    return backend


def test_read_state_rejects_missing_robot_field():
    states = SimpleNamespace(
        dq=np.zeros(7),
        tau=np.zeros(7),
        tcp_pose=np.array([0, 0, 0, 1, 0, 0, 0], float),
        tcp_vel=np.zeros(6),
        ext_wrench_in_tcp=np.zeros(6),
    )
    with pytest.raises(RuntimeError, match="RobotStates.*q"):
        _backend(robot_states=states).read_state()


def test_read_state_rejects_bad_shape_and_nonfinite_values():
    backend = _backend()
    backend._robot._states.dq = np.zeros(6)
    with pytest.raises(RuntimeError, match="shape"):
        backend.read_state()
    backend = _backend()
    backend._robot._states.tcp_pose[0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        backend.read_state()


def test_read_state_rejects_missing_gripper_field():
    backend = _backend(
        gripper_states=SimpleNamespace(width=0.08, force=0.0)
    )
    with pytest.raises(RuntimeError, match="GripperStates.*is_moving"):
        backend.read_state()


def test_read_state_rejects_configured_but_unavailable_gripper():
    backend = _backend()
    backend._gripper = None
    with pytest.raises(RuntimeError, match="refusing to fabricate"):
        backend.read_state()


def test_fault_query_is_fail_closed_when_missing_or_raising():
    backend = _backend()
    backend._robot.fault = None
    assert backend.in_fault() is True

    def failed_fault_read():
        raise RuntimeError("transport failed")

    backend._robot.fault = failed_fault_read
    assert backend.in_fault() is True


def test_gripper_move_is_checked_against_runtime_params():
    backend = _backend()
    backend.move_gripper(
        GripperCommand(
            width=0.08,
            velocity=0.1,
            force=40.0,
            grasp=False,
        )
    )
    assert backend._gripper.calls == [("move", 0.08, 0.1, 40.0)]

    for command, field in (
        (
            GripperCommand(
                width=0.09,
                velocity=0.1,
                force=40.0,
                grasp=False,
            ),
            "width",
        ),
        (
            GripperCommand(
                width=0.08,
                velocity=0.001,
                force=40.0,
                grasp=False,
            ),
            "velocity",
        ),
        (
            GripperCommand(
                width=0.08,
                velocity=0.1,
                force=100.0,
                grasp=False,
            ),
            "force",
        ),
    ):
        with pytest.raises(ValueError, match=field):
            backend.move_gripper(command)


def test_grasp_checks_only_the_parameter_rdk_consumes():
    backend = _backend()
    backend.move_gripper(
        GripperCommand(
            width=999.0,
            velocity=999.0,
            force=20.0,
            grasp=True,
        )
    )
    assert backend._gripper.calls == [("grasp", 20.0)]


def test_gripper_command_fails_closed_without_runtime_limits():
    backend = _backend()
    backend._gripper_limits = None
    with pytest.raises(RuntimeError, match="limits are unavailable"):
        backend.move_gripper(GripperCommand())
