import numpy as np
from tracking.track import TrackStatus
from manipulation.grasp import GraspExecutor


def test_closes_when_within_tolerance_and_confirmed():
    grasp = GraspExecutor(position_tolerance=0.02)
    ee_pos = np.array([0.5, 0.0, 0.1])
    target = np.array([0.51, 0.0, 0.1])
    assert grasp.should_close(ee_pos, target, TrackStatus.CONFIRMED) is True


def test_does_not_close_when_too_far():
    grasp = GraspExecutor(position_tolerance=0.02)
    ee_pos = np.array([0.5, 0.0, 0.1])
    target = np.array([0.6, 0.0, 0.1])
    assert grasp.should_close(ee_pos, target, TrackStatus.CONFIRMED) is False


def test_does_not_close_when_track_unconfirmed():
    grasp = GraspExecutor(position_tolerance=0.02)
    ee_pos = np.array([0.5, 0.0, 0.1])
    target = np.array([0.505, 0.0, 0.1])
    assert grasp.should_close(ee_pos, target, TrackStatus.TENTATIVE) is False


def test_confidence_not_required_when_disabled():
    grasp = GraspExecutor(position_tolerance=0.02, confidence_required=False)
    ee_pos = np.array([0.5, 0.0, 0.1])
    target = np.array([0.505, 0.0, 0.1])
    assert grasp.should_close(ee_pos, target, TrackStatus.TENTATIVE) is True
