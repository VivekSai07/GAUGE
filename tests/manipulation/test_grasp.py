import numpy as np
import pytest

from manipulation.grasp import GraspExecutor
from tracking.track import TrackStatus


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


def test_cov_gate_disabled_by_default_ignores_covariance():
    """cov_threshold defaults to None -- covariance-gating is off, and a
    huge covariance must not block a grasp that would otherwise close."""
    grasp = GraspExecutor(position_tolerance=0.02)
    ee_pos = np.array([0.5, 0.0, 0.1])
    target = np.array([0.51, 0.0, 0.1])
    huge_cov = np.eye(6) * 1000.0
    assert (
        grasp.should_close(ee_pos, target, TrackStatus.CONFIRMED, covariance=huge_cov)
        is True
    )


def test_cov_gate_blocks_when_covariance_too_high():
    grasp = GraspExecutor(position_tolerance=0.02, cov_threshold=0.01)
    ee_pos = np.array([0.5, 0.0, 0.1])
    target = np.array([0.51, 0.0, 0.1])
    # trace(P[:3,:3]) = 3.0, well above the 0.01 threshold.
    high_cov = np.eye(6) * 1.0
    assert (
        grasp.should_close(ee_pos, target, TrackStatus.CONFIRMED, covariance=high_cov)
        is False
    )


def test_cov_gate_passes_when_covariance_low_enough():
    grasp = GraspExecutor(position_tolerance=0.02, cov_threshold=0.01)
    ee_pos = np.array([0.5, 0.0, 0.1])
    target = np.array([0.51, 0.0, 0.1])
    # trace(P[:3,:3]) = 0.003, below the 0.01 threshold.
    low_cov = np.eye(6) * 0.001
    assert (
        grasp.should_close(ee_pos, target, TrackStatus.CONFIRMED, covariance=low_cov)
        is True
    )


def test_cov_gate_requires_covariance_argument_when_enabled():
    grasp = GraspExecutor(position_tolerance=0.02, cov_threshold=0.01)
    ee_pos = np.array([0.5, 0.0, 0.1])
    target = np.array([0.51, 0.0, 0.1])
    with pytest.raises(ValueError):
        grasp.should_close(ee_pos, target, TrackStatus.CONFIRMED)
