import numpy as np
from control.panda_kinematics import panda_fk_symbolic, panda_fk_numpy


def test_casadi_fk_matches_independent_numpy_fk_at_zero_config():
    fk = panda_fk_symbolic()
    q = np.zeros(7)
    casadi_result = np.array(fk(q)).flatten()
    numpy_result = panda_fk_numpy(q)
    np.testing.assert_allclose(casadi_result, numpy_result, atol=1e-9)


def test_casadi_fk_matches_independent_numpy_fk_at_random_config():
    rng = np.random.default_rng(7)
    fk = panda_fk_symbolic()
    for _ in range(5):
        q = rng.uniform(-1.5, 1.5, size=7)
        casadi_result = np.array(fk(q)).flatten()
        numpy_result = panda_fk_numpy(q)
        np.testing.assert_allclose(casadi_result, numpy_result, atol=1e-6)


def test_fk_output_is_finite_and_reasonable_reach():
    fk = panda_fk_symbolic()
    q = np.zeros(7)
    pos = np.array(fk(q)).flatten()
    assert np.all(np.isfinite(pos))
    # Panda's reach is roughly within 1m of the base.
    assert np.linalg.norm(pos) < 1.5
