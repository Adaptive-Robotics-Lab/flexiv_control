import re

import numpy as np

from flexiv_control import Robot, RobotConfig
from flexiv_control.backends.fake import FakeBackend


class _RuntimeLimitsBackend(FakeBackend):
    def runtime_info(self) -> dict:
        return {
            "joint_limits": {
                "source": "test.RobotInfo",
                "position_min_rad": [-1.0] * 7,
                "position_max_rad": [1.0] * 7,
                "velocity_max_rad_s": [
                    1.0,
                    1.1,
                    1.2,
                    1.3,
                    1.4,
                    1.5,
                    1.6,
                ],
            },
            "current_safety_limits": {
                "source": "test.Safety.current_limits",
                "position_min_rad": [-0.9] * 7,
                "position_max_rad": [0.8] * 7,
                "velocity_max_normal_rad_s": [0.8] * 7,
                "velocity_max_reduced_rad_s": [0.4] * 7,
            },
        }


def test_runtime_joint_contract_takes_intersection_and_has_digest():
    robot = Robot(
        config=RobotConfig(backend="fake"),
        backend=_RuntimeLimitsBackend(),
    )
    contract = robot.server_runtime_info()["effective_joint_limits"]

    assert np.allclose(robot.profile.joint_lower, -0.9)
    assert np.allclose(robot.profile.joint_upper, 0.8)
    assert np.allclose(robot._joint_velocity_limits, 0.4)
    assert np.allclose(contract["hard_position_min_rad"], -0.9)
    assert np.allclose(contract["hard_position_max_rad"], 0.8)
    assert np.allclose(
        contract["enforced_position_min_rad"],
        -0.9 + robot.profile.joint_margin_rad,
    )
    assert np.allclose(
        contract["enforced_position_max_rad"],
        0.8 - robot.profile.joint_margin_rad,
    )
    assert re.fullmatch(r"[0-9a-f]{64}", contract["sha256"])
    assert contract["sources"] == [
        "configured_safety_profile",
        "test.RobotInfo",
        "test.Safety.current_limits",
    ]


def test_profile_change_cannot_expand_beyond_runtime_limits():
    robot = Robot(
        config=RobotConfig(backend="fake"),
        backend=_RuntimeLimitsBackend(),
    )
    robot.set_safety_profile("free_space_fast")

    assert np.all(robot.profile.joint_lower >= -0.9)
    assert np.all(robot.profile.joint_upper <= 0.8)
    assert np.allclose(robot._joint_velocity_limits, 0.4)
