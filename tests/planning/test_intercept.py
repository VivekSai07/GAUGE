import numpy as np

from planning.intercept import solve_intercept


def test_stationary_object_intercept_time_equals_distance_over_speed():
    obj_pos0 = np.array([1.0, 0.0, 0.0])
    obj_vel = np.array([0.0, 0.0, 0.0])
    ee_pos = np.array([0.0, 0.0, 0.0])
    ee_max_speed = 1.0

    result = solve_intercept(obj_pos0, obj_vel, ee_pos, ee_max_speed)

    assert result is not None
    point, t = result
    np.testing.assert_allclose(t, 1.0, atol=1e-6)
    np.testing.assert_allclose(point, [1.0, 0.0, 0.0], atol=1e-6)


def test_moving_object_matches_hand_solved_quadratic():
    obj_pos0 = np.array([2.0, 0.0, 0.0])
    obj_vel = np.array([-1.0, 0.0, 0.0])  # moving toward the EE
    ee_pos = np.array([0.0, 0.0, 0.0])
    ee_max_speed = 2.0

    result = solve_intercept(obj_pos0, obj_vel, ee_pos, ee_max_speed)
    assert result is not None
    point, t = result

    # Independently solve a*t^2 + b*t + c = 0 in the test.
    rel_p = obj_pos0 - ee_pos
    a = obj_vel @ obj_vel - ee_max_speed**2
    b = 2 * rel_p @ obj_vel
    c = rel_p @ rel_p
    disc = b**2 - 4 * a * c
    roots = [(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)]
    expected_t = min(r for r in roots if r > 0)

    np.testing.assert_allclose(t, expected_t, atol=1e-6)
    expected_point = obj_pos0 + obj_vel * expected_t
    np.testing.assert_allclose(point, expected_point, atol=1e-6)


def test_unreachable_object_returns_none():
    obj_pos0 = np.array([100.0, 0.0, 0.0])
    obj_vel = np.array([50.0, 0.0, 0.0])  # fleeing faster than EE can chase
    ee_pos = np.array([0.0, 0.0, 0.0])
    ee_max_speed = 1.0

    result = solve_intercept(obj_pos0, obj_vel, ee_pos, ee_max_speed)
    assert result is None
