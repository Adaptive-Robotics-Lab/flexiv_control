from importlib.metadata import version as distribution_version
import re
import time

import numpy as np
import pytest

from flexiv_control import (
    CartesianTrajectory,
    JointTrajectory,
    JointWaypoint,
    RobotConfig,
    __version__,
)
from flexiv_control.client import RemoteRobot, RemoteRobotError
from flexiv_control.server import FlexivControlServer
from flexiv_control.server import protocol as P


@pytest.fixture()
def server():
    # port 0 lets the OS pick a free port
    srv = FlexivControlServer(
        config=RobotConfig(backend="fake", control_hz=200.0),
        host="127.0.0.1",
        port=0,
        lease_ttl=1.0,
    )
    srv.start()
    port = srv._tcp.server_address[1]
    srv.serve_in_thread()
    yield srv, port
    srv.shutdown()


def test_remote_state_and_chunk(server):
    srv, port = server
    with RemoteRobot("127.0.0.1", port, owner="tester") as robot:
        s = robot.get_state()
        assert s.q.shape == (7,)
        robot.start_cartesian_impedance()
        res = robot.execute_cartesian_trajectory(
            CartesianTrajectory.from_waypoint_array(
                [[0.45, 0.0, 0.30, 1.0, 20], [0.48, 0.0, 0.28, 0.0, 20]]
            )
        )
        assert res.success
        assert np.allclose(res.final_state.tcp_position, [0.48, 0.0, 0.28], atol=1e-3)


def test_server_info_is_read_only_and_pins_trajectory_protocol(server, monkeypatch):
    srv, port = server

    def backend_access_forbidden(*args, **kwargs):
        raise AssertionError("get_server_info must not access the robot")

    monkeypatch.setattr(srv.robot, "get_state", backend_access_forbidden)
    robot = RemoteRobot("127.0.0.1", port, owner="identity-probe").connect()
    try:
        assert srv.lease.owner == ""
        assert robot._has_lease is False
        first = robot.get_server_info()
        second = robot.get_server_info()
        assert first == second == P.server_info(**srv.robot.server_runtime_info())
        assert srv.lease.owner == ""
        assert robot._has_lease is False
    finally:
        robot.close()

    assert first["schema"] == "flexiv-control.server-info.v3"
    assert first["package"] == "flexiv-control"
    assert first["package_version"] == __version__
    assert first["protocol_id"] == "flexiv-control.trajectory-rpc.v3"
    assert first["protocol_fingerprint_sha256"] == P.PROTOCOL_FINGERPRINT_SHA256
    assert first["source_fingerprint_sha256"] == P.SOURCE_FINGERPRINT_SHA256
    assert first["control_hz"] == pytest.approx(200.0)
    assert first["active_safety_profile"] == srv.robot.profile.name
    assert re.fullmatch(r"[0-9a-f]{64}", first["protocol_fingerprint_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", first["source_fingerprint_sha256"])
    assert P.PROTOCOL_CONTRACT["trajectory_rpcs"] == {
        "execute_cartesian_trajectory": "traj",
        "execute_joint_trajectory": "traj",
    }
    assert P.PROTOCOL_CONTRACT["identity_rpc"]["runtime_fields"] == {
        "required": ["control_hz", "active_safety_profile"],
        "hardware_when_available": [
            "runtime_hardware_identity",
            "gripper_limits",
            "joint_limits",
            "current_safety_limits",
            "effective_joint_limits",
        ],
    }
    assert P.PROTOCOL_CONTRACT["joint_trajectory_contract"] == {
        "schema": P.JOINT_TRAJECTORY_SCHEMA,
        "trajectory_fields": [
            "schema",
            "waypoints",
            "initial_positions",
            "initial_gripper_width",
            "max_joint_speed_scale",
            "interpolation",
            "strict_timing",
            "safety_profile",
        ],
        "waypoint_fields": ["positions", "n_frames", "duration", "gripper"],
        "gripper_target_fields": ["width", "force", "velocity"],
        "rpc_identity_fields": ["protocol_id", "protocol_fingerprint_sha256"],
        "explicit_initial_target": ["initial_positions", "initial_gripper_width"],
        "strict_timing": "authoritative-n_frames-reject-no-clip-or-time-stretch",
        "continuity": "measured-rebase-every-rpc-prior-ack-provenance-only",
        "first_emitted_joint_bound": "effective-runtime-rate-times-control-period",
        "gripper_execution_anchor": "current-measured-width-every-rpc",
        "predispatch_revalidation": [
            "first-joint-setpoint-after-mode-transition",
            "first-gripper-event-from-current-measured-width",
        ],
        "numeric_json_types": "numbers-and-arrays-only-no-strings-or-booleans",
        "gripper": "exact-Move-target-concurrent-at-segment-boundary",
        "interpolation": ["cosine", "linear"],
        "max_joint_speed_scale": "finite-(0,1]-active-profile-ceiling",
    }


def test_distribution_metadata_matches_runtime_version():
    assert distribution_version("flexiv-control") == __version__


def test_server_info_rejects_missing_or_invalid_runtime_fields():
    with pytest.raises(ValueError, match="missing required runtime"):
        P.server_info()
    with pytest.raises(ValueError, match="control_hz"):
        P.server_info(
            control_hz=0.0,
            active_safety_profile="tabletop_safe",
        )
    with pytest.raises(ValueError, match="active_safety_profile"):
        P.server_info(control_hz=100.0, active_safety_profile="")


def test_remote_servo_delta(server):
    srv, port = server
    with RemoteRobot("127.0.0.1", port, owner="tester") as robot:
        robot.start_cartesian_impedance()
        before = robot.get_state().tcp_position.copy()
        robot.servo_cartesian_delta([0.0, 0.01, 0.0, 0, 0, 0], duration=0.05)
        after = robot.get_state().tcp_position
        assert after[1] > before[1]


def test_joint_trajectory_contract_is_enforced_before_streaming(server):
    srv, port = server
    with RemoteRobot("127.0.0.1", port, owner="joint-contract") as robot:
        q0 = robot.get_state().q.copy()
        with pytest.raises(RemoteRobotError, match="exceeds active"):
            robot.execute_joint_trajectory(
                JointTrajectory(
                    waypoints=[JointWaypoint(positions=q0, duration=0.02)],
                    max_joint_speed_scale=0.31,
                    interpolation="linear",
                )
            )
        assert np.array_equal(robot.get_state().q, q0)

        result = robot.execute_joint_trajectory(
            JointTrajectory(
                waypoints=[
                    JointWaypoint(positions=q0, duration=0.016),
                    JointWaypoint(positions=q0, duration=0.016),
                ],
                max_joint_speed_scale=0.29,
                interpolation="linear",
            )
        )
        assert result.success
        assert result.log["joint_interpolation"] == "linear"
        assert result.log["effective_max_joint_speed_scale"] == pytest.approx(0.29)
        # 2 x 3.2 requested ticks at the 200 Hz fixture -> 3 + 3 = 6.
        assert result.log["scheduled_segment_ticks"] == [3, 3]
        assert result.log["scheduled_total_ticks"] == 6


def test_joint_interpolation_round_trips_and_old_payload_is_refused():
    traj = JointTrajectory(
        waypoints=[JointWaypoint(np.zeros(7), duration=0.1)],
        interpolation="linear",
    )
    payload = P.joint_trajectory_to_dict(traj)
    assert payload["schema"] == P.JOINT_TRAJECTORY_SCHEMA
    assert payload["interpolation"] == "linear"
    assert P.joint_trajectory_from_dict(payload).interpolation == "linear"

    del payload["interpolation"]
    with pytest.raises(ValueError, match="keys do not match"):
        P.joint_trajectory_from_dict(payload)


def test_lease_blocks_second_client(server):
    srv, port = server
    a = RemoteRobot("127.0.0.1", port, owner="alice").connect()
    a.acquire_lease()
    b = RemoteRobot("127.0.0.1", port, owner="bob").connect()
    with pytest.raises(RemoteRobotError):
        b.acquire_lease()  # alice holds it
    # but commands without the lease are rejected too
    with pytest.raises(RemoteRobotError):
        b.start_cartesian_impedance()
    a.close()
    # after alice releases, bob can take it
    b.acquire_lease()
    b.close()


def test_lease_expires_after_ttl(server):
    srv, port = server
    a = RemoteRobot("127.0.0.1", port, owner="alice").connect()
    # acquire but then kill heartbeat to simulate a dead client
    a.acquire_lease()
    a._stop_heartbeat()
    time.sleep(1.2)  # > lease_ttl
    b = RemoteRobot("127.0.0.1", port, owner="bob").connect()
    b.acquire_lease()  # should succeed: alice's lease expired
    b.close()
    a.close()


def test_fresh_lease_owner_does_not_inherit_stale_stop(server):
    """A dying session's safety stop must not leak into the next session.

    alice latches a stop (client stop / disconnect handler) with no motion in
    flight, so nothing consumes the cooperative-cancel flag. Pre-fix, bob's
    FIRST traj then instant-aborted with ``stop=user dur=0.00`` (observed
    live on hardware); acquiring the lease as a FRESH owner now clears the
    stale flag."""
    srv, port = server
    a = RemoteRobot("127.0.0.1", port, owner="alice").connect()
    a.acquire_lease()
    a.stop()  # latched; no motion in flight consumes it
    a.release_lease()
    a.close()

    b = RemoteRobot("127.0.0.1", port, owner="bob").connect()
    b.acquire_lease()
    b.start_cartesian_impedance()
    res = b.execute_cartesian_trajectory(
        CartesianTrajectory.from_waypoint_array([[0.45, 0.0, 0.30, 1.0, 20]])
    )
    assert res.success, f"first traj of a fresh session aborted: {res.stop_reason}"
    assert str(res.stop_reason) != "user"
    b.close()
