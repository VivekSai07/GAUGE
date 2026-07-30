"""Constant-velocity Kalman filter for 3D position tracking."""
import numpy as np


class ConstantVelocityKF:
    def __init__(
        self,
        dt: float,
        process_var: float,
        meas_var: float,
        init_state: np.ndarray,
        init_cov: np.ndarray,
    ):
        self.dt = dt
        self.x = np.asarray(init_state, dtype=np.float64).copy()
        self.P = np.asarray(init_cov, dtype=np.float64).copy()

        self.F = np.eye(6)
        self.F[0:3, 3:6] = np.eye(3) * dt

        self.H = np.zeros((3, 6))
        self.H[0:3, 0:3] = np.eye(3)

        # Discretized white-noise-acceleration process noise, block per axis.
        q = process_var
        Q_block = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * q
        self.Q = np.zeros((6, 6))
        for axis in range(3):
            idx = [axis, axis + 3]
            self.Q[np.ix_(idx, idx)] = Q_block

        self.R = np.eye(3) * meas_var

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def innovation_distance(self, z: np.ndarray) -> float:
        """Mahalanobis distance of measurement `z` against the current
        predicted state, with NO mutation of `self.x`/`self.P`.

        Callers that need to gate a measurement before deciding whether to
        incorporate it (see `tracking.track.Track.step`) should call this
        first, and only call `update()` once the measurement has passed the
        gate -- `update()` always applies the correction unconditionally.
        """
        z = np.asarray(z, dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return float(np.sqrt(y.T @ np.linalg.inv(S) @ y))

    def update(self, z: np.ndarray) -> float:
        z = np.asarray(z, dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        mahalanobis = float(np.sqrt(y.T @ np.linalg.inv(S) @ y))

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

        return mahalanobis
