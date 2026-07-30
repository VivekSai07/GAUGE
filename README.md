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

At `configs/conveyor.yaml`'s shipped `grasp.position_tolerance: 0.035`, the
system grasps at a true (ground-truth) fingertip-to-object error of
**~4.4cm**, verified deterministically by `tests/test_integration_conveyor.py`
— close to the originally-targeted 3cm. The conveyor object is a real,
physically-simulated body (not a scripted ghost), so the gripper can
actually hold it. See
[the design spec's Section 12](docs/superpowers/specs/2026-07-29-dynamic-object-tracking-manipulation-design.md#12-demonstrated-accuracy--known-limitation)
for the full history, including an earlier ~7cm result and the three
specific fixes (real object physics, a smoothed approach target, and
fingertip-consistent targeting) that closed most of the gap.

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
