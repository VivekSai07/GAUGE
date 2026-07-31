# GAUGE — Gated Uncertainty-Aware Grasping Engine

[![CI](https://github.com/VivekSai07/GAUGE/actions/workflows/ci.yml/badge.svg)](https://github.com/VivekSai07/GAUGE/actions/workflows/ci.yml)

A MuJoCo-simulated Franka Panda arm with an eye-in-hand RGB-D camera that
detects, tracks (Kalman filter with Mahalanobis gating and m/n track
confirmation), predicts, computes a closed-form interception point for, and
grasps an object moving at constant velocity on a conveyor belt — a personal
project applying object-tracking coursework (KF/EKF/UKF, gating, track
initiation, m/n logic) to robotic manipulation.

Pure Python, no ROS2/C++/Pinocchio/acados: perception, tracking, prediction,
interception planning, and control (a kinematic MPC via CasADi + IPOPT) are
plain Python modules wired together in `run_conveyor_demo.py`.

## Demonstrated result

The Franka genuinely picks up the cube: `run_one_episode()` returns
`contact_verified: True` — both fingers simultaneously in physical contact
with the object (MuJoCo's own contact array, not inferred from distance),
sustained for a real hold, not an instant. Fingertip-to-object accuracy at
the commit instant is **~3.9cm**, close to the originally-targeted 3cm.
Verified deterministic and reproducible by
`tests/test_integration_conveyor.py`.

Getting here took three rounds of real fixes, not tuning: the conveyor
object had to become a genuinely physically-simulated body (not a scripted
ghost immune to contact forces), the MPC needed actual orientation control
(position-only tracking left the object centered in aggregate distance but
off the gripper's closing axis), and grip friction needed raising so the
object didn't slip out under gravity. See
[the design spec's Section 12](docs/superpowers/specs/2026-07-29-dynamic-object-tracking-manipulation-design.md#12-demonstrated-accuracy--known-limitation)
("Round 3" in particular) for the full root-cause story — including why an
earlier, honestly-measured ~4.4cm accuracy number still described a system
that had never once picked anything up.

## Running it

```bash
uv sync
uv run pytest -v                       # full test suite
uv run python run_conveyor_demo.py     # one closed-loop episode, headless
uv run python run_conveyor_demo.py --render   # same, with a live viewer window
```

## Project structure

```
perception/     RGB-D -> 3D centroid (classical color/depth segmentation)
tracking/       Constant-velocity Kalman filter, gating, m/n confirmation
prediction/     Forward propagation of state + covariance
planning/       Closed-form interception solver
control/        Panda forward kinematics (CasADi) + kinematic MPC
manipulation/   Grasp-commit decision
sim/            MuJoCo conveyor scene (Panda + eye-in-hand camera)
configs/        Run parameters
```

## Documentation

- [Design spec](docs/superpowers/specs/2026-07-29-dynamic-object-tracking-manipulation-design.md) — architecture, novelty positioning, and the demonstrated-accuracy writeup
- [Implementation plan](docs/superpowers/plans/2026-07-29-conveyor-mvp.md) — the task-by-task build plan
