import numpy as np

from tracking.kf import ConstantVelocityKF
from tracking.track import Track, TrackStatus


def make_track(**overrides):
    kf = ConstantVelocityKF(
        dt=0.1,
        process_var=1e-4,
        meas_var=1e-3,
        init_state=np.zeros(6),
        init_cov=np.eye(6) * 0.01,
    )
    defaults = {"gate_threshold": 9.0, "m": 3, "n": 5, "max_consecutive_misses": 3}
    defaults.update(overrides)
    return Track(kf=kf, **defaults)


def test_starts_tentative():
    track = make_track()
    assert track.status == TrackStatus.TENTATIVE


def test_confirms_after_m_of_n_hits():
    track = make_track()
    status = TrackStatus.TENTATIVE
    for _ in range(3):
        status = track.step(measurement=np.zeros(3))
    assert status == TrackStatus.CONFIRMED


def test_stays_tentative_with_too_few_hits():
    track = make_track()
    status = track.step(measurement=np.zeros(3))
    status = track.step(measurement=None)
    assert status == TrackStatus.TENTATIVE


def test_goes_lost_after_max_consecutive_misses():
    track = make_track()
    status = TrackStatus.TENTATIVE
    for _ in range(4):
        status = track.step(measurement=None)
    assert status == TrackStatus.LOST


def test_gate_rejects_far_measurement_as_miss():
    track = make_track(gate_threshold=1.0)
    # First hit near zero to seed the estimate.
    track.step(measurement=np.zeros(3))
    # A wildly distant measurement should be gated out (treated as a miss).
    status = track.step(measurement=np.array([100.0, 100.0, 100.0]))
    assert status == TrackStatus.TENTATIVE


def test_gated_out_measurement_leaves_filter_state_unchanged():
    """A rejected (gated-out) measurement must have ZERO effect on the
    Kalman filter's state -- not just on the reported hit/miss bookkeeping.
    Regression test for a bug where kf.update() mutated self.x/self.P before
    the gate was checked, so an outlier corrupted the estimate even though it
    was correctly counted as a miss."""
    track = make_track(gate_threshold=1.0)
    # First hit near zero to seed the estimate.
    track.step(measurement=np.zeros(3))

    kf = track.kf
    x_before, P_before, F, Q = kf.x.copy(), kf.P.copy(), kf.F, kf.Q
    # What predict() alone (no update) should produce -- the only thing
    # that's allowed to happen to kf.x/kf.P when the next measurement is
    # gated out.
    expected_x = F @ x_before
    expected_P = F @ P_before @ F.T + Q

    # A wildly distant measurement should be gated out (treated as a miss)
    # and must not move kf.x/kf.P away from those post-predict() values.
    status = track.step(measurement=np.array([100.0, 100.0, 100.0]))

    assert status == TrackStatus.TENTATIVE
    np.testing.assert_allclose(kf.x, expected_x)
    np.testing.assert_allclose(kf.P, expected_P)
