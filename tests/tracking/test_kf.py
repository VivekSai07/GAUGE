import numpy as np
import pytest

from tracking.kf import ConstantVelocityKF


def test_predict_advances_position_by_velocity():
    kf = ConstantVelocityKF(
        dt=0.1,
        process_var=1e-4,
        meas_var=1e-3,
        init_state=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        init_cov=np.eye(6) * 0.01,
    )
    kf.predict()
    np.testing.assert_allclose(kf.x[:3], [0.1, 0.0, 0.0], atol=1e-9)


def test_converges_to_true_constant_velocity():
    rng = np.random.default_rng(42)
    dt = 0.05
    true_vel = np.array([0.2, -0.1, 0.0])
    true_pos = np.array([0.0, 0.0, 0.5])

    kf = ConstantVelocityKF(
        dt=dt,
        process_var=1e-5,
        meas_var=1e-4,
        init_state=np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
        init_cov=np.eye(6) * 0.1,
    )

    for _ in range(200):
        true_pos = true_pos + true_vel * dt
        z = true_pos + rng.normal(scale=np.sqrt(1e-4), size=3)
        kf.predict()
        kf.update(z)

    np.testing.assert_allclose(kf.x[3:], true_vel, atol=0.05)
    np.testing.assert_allclose(kf.x[:3], true_pos, atol=0.05)


def test_mahalanobis_distance_zero_for_perfect_measurement():
    kf = ConstantVelocityKF(
        dt=0.1,
        process_var=1e-4,
        meas_var=1e-3,
        init_state=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        init_cov=np.eye(6) * 0.01,
    )
    kf.predict()
    d = kf.update(kf.x[:3].copy())
    assert d < 1e-6


def test_innovation_distance_does_not_mutate_state():
    kf = ConstantVelocityKF(
        dt=0.1,
        process_var=1e-4,
        meas_var=1e-3,
        init_state=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        init_cov=np.eye(6) * 0.01,
    )
    kf.predict()
    x_before = kf.x.copy()
    P_before = kf.P.copy()

    kf.innovation_distance(np.array([5.0, 5.0, 5.0]))

    np.testing.assert_array_equal(kf.x, x_before)
    np.testing.assert_array_equal(kf.P, P_before)


def test_innovation_distance_matches_update_return_value():
    """innovation_distance() must report exactly what update() used to report
    for the same measurement, so existing gate-threshold-dependent behavior
    is unchanged for accepted measurements."""
    z = np.array([0.07, -0.03, 0.11])

    kf_a = ConstantVelocityKF(
        dt=0.1,
        process_var=1e-4,
        meas_var=1e-3,
        init_state=np.array([0.0, 0.0, 0.0, 1.0, 0.2, -0.1]),
        init_cov=np.eye(6) * 0.05,
    )
    kf_b = ConstantVelocityKF(
        dt=0.1,
        process_var=1e-4,
        meas_var=1e-3,
        init_state=np.array([0.0, 0.0, 0.0, 1.0, 0.2, -0.1]),
        init_cov=np.eye(6) * 0.05,
    )
    kf_a.predict()
    kf_b.predict()

    distance_non_mutating = kf_a.innovation_distance(z)
    distance_from_update = kf_b.update(z)

    assert distance_non_mutating == pytest.approx(distance_from_update)
