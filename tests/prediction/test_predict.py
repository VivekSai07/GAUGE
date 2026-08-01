from itertools import pairwise

import numpy as np

from prediction.predict import propagate


def test_propagate_constant_velocity_matches_hand_computation():
    dt = 0.1
    F = np.eye(6)
    F[0:3, 3:6] = np.eye(3) * dt
    Q = np.eye(6) * 1e-4

    x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    P0 = np.eye(6) * 0.01

    result = propagate(x0, P0, F, Q, steps=3)

    assert len(result) == 3
    # Independently hand-compute the expected trajectory.
    expected_positions = [
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.3, 0.0, 0.0],
    ]
    for (x_k, P_k), expected_pos in zip(result, expected_positions):
        np.testing.assert_allclose(x_k[:3], expected_pos, atol=1e-9)
        assert P_k.shape == (6, 6)


def test_covariance_grows_monotonically():
    dt = 0.1
    F = np.eye(6)
    F[0:3, 3:6] = np.eye(3) * dt
    Q = np.eye(6) * 1e-3

    x0 = np.zeros(6)
    P0 = np.eye(6) * 0.01

    result = propagate(x0, P0, F, Q, steps=5)
    traces = [np.trace(P_k) for _, P_k in result]
    assert all(t2 > t1 for t1, t2 in pairwise(traces))
