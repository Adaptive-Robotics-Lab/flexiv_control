import time

import numpy as np
import pytest

from flexiv_control import CartesianTrajectory, RobotConfig
from flexiv_control.client import RemoteRobot, RemoteRobotError
from flexiv_control.server import FlexivControlServer


@pytest.fixture()
def server():
    # port 0 lets the OS pick a free port
    srv = FlexivControlServer(config=RobotConfig(backend="fake", control_hz=200.0),
                              host="127.0.0.1", port=0, lease_ttl=1.0)
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
            CartesianTrajectory.from_waypoint_array([[0.45, 0.0, 0.30, 1.0, 20],
                                          [0.48, 0.0, 0.28, 0.0, 20]])
        )
        assert res.success
        assert np.allclose(res.final_state.tcp_position, [0.48, 0.0, 0.28], atol=1e-3)


def test_remote_servo_delta(server):
    srv, port = server
    with RemoteRobot("127.0.0.1", port, owner="tester") as robot:
        robot.start_cartesian_impedance()
        before = robot.get_state().tcp_position.copy()
        robot.servo_cartesian_delta([0.0, 0.01, 0.0, 0, 0, 0], duration=0.05)
        after = robot.get_state().tcp_position
        assert after[1] > before[1]


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
    a.stop()                      # latched; no motion in flight consumes it
    a.release_lease()
    a.close()

    b = RemoteRobot("127.0.0.1", port, owner="bob").connect()
    b.acquire_lease()
    b.start_cartesian_impedance()
    res = b.execute_cartesian_trajectory(
        CartesianTrajectory.from_waypoint_array([[0.45, 0.0, 0.30, 1.0, 20]]))
    assert res.success, f"first traj of a fresh session aborted: {res.stop_reason}"
    assert str(res.stop_reason) != "user"
    b.close()
