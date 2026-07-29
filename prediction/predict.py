"""Forward propagation of a linear-Gaussian state estimate."""
import numpy as np


def propagate(
    x: np.ndarray, P: np.ndarray, F: np.ndarray, Q: np.ndarray, steps: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    results = []
    x_k, P_k = x.copy(), P.copy()
    for _ in range(steps):
        x_k = F @ x_k
        P_k = F @ P_k @ F.T + Q
        results.append((x_k.copy(), P_k.copy()))
    return results
