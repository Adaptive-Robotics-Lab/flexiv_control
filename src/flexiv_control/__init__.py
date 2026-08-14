"""flexiv_control -- a unified, real-time execution & safety layer for Flexiv Rizon arms.

One action contract for teleoperation, MPC, reinforcement learning, and
real-to-sim-to-real manipulation. Backend-agnostic (real Rizon via Flexiv RDK,
a dependency-free FakeBackend, or MuJoCo), with an optional cross-process server,
an optional C++ 1 kHz daemon, an optional ROS 2 overlay, a Gymnasium env, and a
LeRobot adapter.

This is a community project and is **not affiliated with Flexiv Robotics**.

Quick start (no hardware needed)::

    from flexiv_control import Robot, RobotConfig, CartesianTrajectory

    robot = Robot(RobotConfig(backend="fake"))
    robot.connect()
    robot.start_cartesian_impedance()
    traj = CartesianTrajectory.from_waypoint_array([[0.45, 0.0, 0.30, 1.0, 20],
                                          [0.50, 0.0, 0.25, 0.0, 20]])
    result = robot.execute_cartesian_trajectory(traj)
    print(result.success, result.path_tracking_error)
    robot.disconnect()
"""

from __future__ import annotations

__version__ = "0.4.0.dev0"

# --- core data types
from .types import (  # noqa: F401
    ControlMode,
    ForceControlParams,
    GripperCommand,
    ImpedanceParams,
    JointImpedanceParams,
    RobotState,
    SafetyStatus,
    StopReason,
)

# --- the action contract
from .trajectory import (  # noqa: F401
    CartesianTrajectory,
    CartesianDelta,
    CartesianWaypoint,
    TrajectoryRepresentation,
    ExecutionResult,
    JointGripperForceTarget,
    JointGripperTarget,
    JointTrajectory,
    JointWaypoint,
    JointTorqueTrajectory,
    JointTorqueWaypoint,
)

# --- safety + config
from .safety import SafetyFilter, SafetyProfile  # noqa: F401
from .config import RobotConfig, load_safety_profile  # noqa: F401

# --- backends (light ones only; heavy/optional ones load lazily by name)
from .backends import FakeBackend, RobotBackend, get_backend  # noqa: F401

# --- the one facade
from .robot import TrajectoryStoppedError, LeaseError, Robot  # noqa: F401

# --- receding-horizon / VLA policy-server seam
from .recede import RecedingHorizonRunner, console_confirm  # noqa: F401
from .policy_client import RemotePolicyClient  # noqa: F401

__all__ = [
    "__version__",
    # types
    "ControlMode",
    "ForceControlParams",
    "GripperCommand",
    "ImpedanceParams",
    "JointImpedanceParams",
    "RobotState",
    "SafetyStatus",
    "StopReason",
    # action contract
    "CartesianTrajectory",
    "CartesianDelta",
    "CartesianWaypoint",
    "TrajectoryRepresentation",
    "ExecutionResult",
    "JointGripperForceTarget",
    "JointGripperTarget",
    "JointTrajectory",
    "JointWaypoint",
    "JointTorqueTrajectory",
    "JointTorqueWaypoint",
    # safety + config
    "SafetyFilter",
    "SafetyProfile",
    "RobotConfig",
    "load_safety_profile",
    # backends
    "FakeBackend",
    "RobotBackend",
    "get_backend",
    # facade
    "Robot",
    "LeaseError",
    "TrajectoryStoppedError",
    # receding horizon
    "RecedingHorizonRunner",
    "RemotePolicyClient",
    "console_confirm",
]


def __getattr__(name: str):
    """Lazily expose optional, heavier components without importing their deps eagerly."""
    if name == "RemoteRobot":
        from .client.remote_robot import RemoteRobot

        return RemoteRobot
    if name == "FlexivControlServer":
        from .server.server import FlexivControlServer

        return FlexivControlServer
    if name == "FlexivRealEnv":
        from .envs.gym_env import FlexivRealEnv

        return FlexivRealEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
