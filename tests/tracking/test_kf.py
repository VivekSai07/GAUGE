import numpy as np
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
