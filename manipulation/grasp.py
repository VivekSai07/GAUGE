"""Grasp-commit decision: gated on distance tolerance and track confidence."""
import numpy as np

from tracking.track import TrackStatus


class GraspExecutor:
    def __init__(self, position_tolerance: float, confidence_required: bool = True):
        self.position_tolerance = position_tolerance
        self.confidence_required = confidence_required

    def should_close(
        self, ee_pos: np.ndarray, target_pos: np.ndarray, track_status: TrackStatus
    ) -> bool:
        distance = float(np.linalg.norm(ee_pos - target_pos))
        within_tolerance = distance <= self.position_tolerance
        if self.confidence_required:
            return within_tolerance and track_status == TrackStatus.CONFIRMED
        return within_tolerance
