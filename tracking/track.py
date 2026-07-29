"""Track state machine: gating + m/n confirmation logic."""
from collections import deque
from enum import Enum

import numpy as np

from tracking.kf import ConstantVelocityKF


class TrackStatus(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


class Track:
    def __init__(
        self,
        kf: ConstantVelocityKF,
        gate_threshold: float,
        m: int,
        n: int,
        max_consecutive_misses: int,
    ):
        self.kf = kf
        self.gate_threshold = gate_threshold
        self.m = m
        self.n = n
        self.max_consecutive_misses = max_consecutive_misses
        self.hit_history: deque[bool] = deque(maxlen=n)
        self.consecutive_misses = 0
        self.status = TrackStatus.TENTATIVE

    def step(self, measurement: np.ndarray | None) -> TrackStatus:
        self.kf.predict()

        hit = False
        if measurement is not None:
            distance = self.kf.update(measurement)
            if distance <= self.gate_threshold:
                hit = True

        self.hit_history.append(hit)
        self.consecutive_misses = 0 if hit else self.consecutive_misses + 1

        if self.consecutive_misses >= self.max_consecutive_misses:
            self.status = TrackStatus.LOST
        elif sum(self.hit_history) >= self.m:
            self.status = TrackStatus.CONFIRMED
        elif self.status != TrackStatus.CONFIRMED:
            self.status = TrackStatus.TENTATIVE

        return self.status
