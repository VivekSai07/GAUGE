# GAUGE — Gated Uncertainty-Aware Grasping Engine

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

At `configs/conveyor.yaml`'s shipped `grasp.position_tolerance: 0.075`, the
system grasps at a true (ground-truth) end-effector-to-object error of
**~7.1cm**, verified deterministically by `tests/test_integration_conveyor.py`.
This is an honest, documented outcome, not the originally-targeted 3cm — see
[the design spec's Section 12](docs/superpowers/specs/2026-07-29-dynamic-object-tracking-manipulation-design.md#12-demonstrated-accuracy--known-limitation)
for the full root-cause analysis (kinematic-MPC vs. real actuator dynamics,
perception noise, and the flange-vs-fingertip frame gap) and what a genuine
accuracy improvement would require.

## Running it

```bash
uv sync
uv run pytest -v                     # full test suite
uv run python run_conveyor_demo.py   # one closed-loop episode
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
