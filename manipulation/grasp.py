"""Grasp-commit decision: gated on distance tolerance and track confidence.

Optionally also gated on estimate covariance (`cov_threshold`) -- the
design spec's "covariance-gated commit" novelty axis (Section 2/3.4): commit
only once the track is m/n-confirmed AND the estimate's covariance has
shrunk below a threshold. This is OFF by default (`cov_threshold=None`) so
every existing caller/test/config is completely unaffected unless a
threshold is explicitly supplied.
"""
import numpy as np

from tracking.track import TrackStatus


class GraspExecutor:
    def __init__(
        self,
        position_tolerance: float,
        confidence_required: bool = True,
        cov_threshold: float | None = None,
    ):
        self.position_tolerance = position_tolerance
        self.confidence_required = confidence_required
        self.cov_threshold = cov_threshold

    def should_close(
        self,
        ee_pos: np.ndarray,
        target_pos: np.ndarray,
        track_status: TrackStatus,
        covariance: np.ndarray | None = None,
    ) -> bool:
        distance = float(np.linalg.norm(ee_pos - target_pos))
        within_tolerance = distance <= self.position_tolerance

        if self.confidence_required and track_status != TrackStatus.CONFIRMED:
            return False

        if self.cov_threshold is not None:
            if covariance is None:
                raise ValueError(
                    "cov_threshold is set but no `covariance` was passed to "
                    "should_close()."
                )
            position_cov_trace = float(np.trace(np.asarray(covariance)[:3, :3]))
            if position_cov_trace > self.cov_threshold:
                return False

        return within_tolerance
