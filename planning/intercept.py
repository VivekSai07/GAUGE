"""Closed-form time-to-intercept for a constant-velocity target."""
import numpy as np


def solve_intercept(
    obj_pos0: np.ndarray,
    obj_vel: np.ndarray,
    ee_pos: np.ndarray,
    ee_max_speed: float,
) -> tuple[np.ndarray, float] | None:
    rel_p = obj_pos0 - ee_pos
    a = float(obj_vel @ obj_vel - ee_max_speed**2)
    b = float(2 * rel_p @ obj_vel)
    c = float(rel_p @ rel_p)

    if abs(a) < 1e-12:
        # Degenerate to linear equation b*t + c = 0.
        if abs(b) < 1e-12:
            return None
        t = -c / b
        candidates = [t] if t > 0 else []
    else:
        disc = b**2 - 4 * a * c
        if disc < 0:
            return None
        sqrt_disc = np.sqrt(disc)
        roots = [(-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)]
        candidates = [r for r in roots if r > 0]

    if not candidates:
        return None

    t_star = min(candidates)
    point = obj_pos0 + obj_vel * t_star
    return point, t_star
