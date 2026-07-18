import numpy as np

from flexiv_control import CartesianTrajectory, CartesianWaypoint
from flexiv_control import transforms as T
from flexiv_control.interpolation import CartesianTrajectoryInterpolator


def test_quat_mul_identity():
    q = T.quat_normalize([0.3, 0.1, -0.2, 0.5])
    ident = np.array([1.0, 0, 0, 0])
    assert np.allclose(T.quat_mul(ident, q), q)


def test_quat_conj_inverse():
    q = T.quat_normalize([0.3, 0.1, -0.2, 0.5])
    prod = T.quat_mul(q, T.quat_conj(q))
    assert np.allclose(prod, [1, 0, 0, 0], atol=1e-9)


def test_slerp_endpoints():
    a = np.array([1.0, 0, 0, 0])
    b = T.rotvec_to_quat([0, 0, np.pi / 2])
    assert np.allclose(T.quat_slerp(a, b, 0.0), a)
    assert np.allclose(T.quat_slerp(a, b, 1.0), T.quat_normalize(b))


def test_rotvec_roundtrip():
    rv = np.array([0.1, -0.2, 0.3])
    q = T.rotvec_to_quat(rv)
    assert np.allclose(T.quat_to_rotvec(q), rv, atol=1e-9)


def test_integrate_pose_translation():
    pose = np.array([0.4, 0.0, 0.3, 1, 0, 0, 0], float)
    out = T.integrate_pose(pose, [0.05, -0.02, 0.01, 0, 0, 0])
    assert np.allclose(out[:3], [0.45, -0.02, 0.31])
    assert np.allclose(out[3:7], [1, 0, 0, 0])


def test_interpolator_tick_count_and_endpoint():
    start = np.array([0.4, 0.0, 0.3, 1, 0, 0, 0], float)
    wp = CartesianWaypoint(position=[0.5, 0.0, 0.3], n_frames=10)
    traj = CartesianTrajectory(waypoints=[wp])
    interp = CartesianTrajectoryInterpolator(traj, start, control_hz=100.0)
    setpoints = interp.setpoints()
    assert len(setpoints) == 10  # n_frames at the matching rate
    last_pose, _ = setpoints[-1]
    assert np.allclose(last_pose[:3], [0.5, 0.0, 0.3], atol=1e-6)


def test_interpolator_holds_orientation():
    start = np.array([0.4, 0.0, 0.3, 1, 0, 0, 0], float)
    wp = CartesianWaypoint(position=[0.5, 0.0, 0.3], duration=0.05)  # quaternion None -> hold
    interp = CartesianTrajectoryInterpolator(CartesianTrajectory(waypoints=[wp]), start, 100.0)
    for pose, _ in interp:
        assert np.allclose(pose[3:7], [1, 0, 0, 0], atol=1e-9)
