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

The Franka reaches, targets, and closes on the cube with fingertip-to-object
accuracy at the commit instant of **~3.9cm**, and mechanically registers
contact (`grasped: True`). But `contact_verified` — checked *after* a real
~10cm lift, not at the instant the gripper closes — currently reads
**`False`**: the grasp does not yet survive being lifted. This is an honest,
verified, currently-open limitation, not an oversight — see below.

Getting here took four rounds of real fixes, not tuning. Rounds 1-3: the
conveyor object had to become a genuinely physically-simulated body (not a
scripted ghost immune to contact forces), the MPC needed actual orientation
control (position-only tracking left the object centered in aggregate
distance but off the gripper's closing axis), and grip friction needed
raising so the object didn't slip out under gravity — these got a momentary
contact check to read `True`. Round 4 moved that same check to *after* a
real lift (the momentary check turned out to be too weak a bar, informed by
a second working reference implementation's explicit "Verify Lift" step)
and root-caused, with `systematic-debugging`, exactly why the grasp doesn't
survive one: a sharp, mechanically-grounded tolerance cliff (~3cm along the
gripper's closing axis) that this project's current 64×64 RGB-D
color-segmentation accuracy sits right on top of. See
[the design spec's Section 12](docs/superpowers/specs/2026-07-29-dynamic-object-tracking-manipulation-design.md#12-demonstrated-accuracy--known-limitation)
("Round 4" in particular) for the full root-cause story — eight-plus
hypotheses tested against real runs, an isolation experiment that pins the
exact failure threshold, and why an earlier, honestly-measured
`contact_verified: True` still described a system that would drop the
object the moment it was picked up.

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
