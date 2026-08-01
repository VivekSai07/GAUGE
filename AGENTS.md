# AGENTS.md

GAUGE: a MuJoCo-simulated Franka Panda that detects, tracks (Kalman filter),
predicts, and grasps a moving conveyor object via a closed-form interception
solver and a kinematic MPC. Pure Python, no ROS2/Pinocchio/acados. See
[README.md](README.md) for the project pitch and demonstrated result.

## Architecture

Pipeline stages, each a standalone package with **no cross-package imports**
(all `__init__.py` files are empty, no exports): `perception/` → `tracking/`
→ `prediction/` → `planning/` → `control/` → `manipulation/`, with
`sim/conveyor_scene.py` providing the MuJoCo env. Everything is wired
together only in [run_conveyor_demo.py](run_conveyor_demo.py) — if you add a
new stage or swap an implementation, wire it there, not via inter-package
imports. Full architecture/novelty rationale:
[design spec §3](docs/superpowers/specs/2026-07-29-dynamic-object-tracking-manipulation-design.md#3-system-architecture).

## Build / test / run

```bash
uv sync                              # install deps (uv-managed venv already present)
uv run pytest -v                     # full test suite
uv run python run_conveyor_demo.py   # one closed-loop episode
uv run python run_conveyor_demo.py --render  # same, with MuJoCo viewer window
uv run ruff check .                  # lint
uv run ruff format .                 # format
```

`ruff` lint/format is enforced in CI (`extend-exclude`s the vendored
menagerie submodule and `docs/`, which has historical planning/spec docs
with embedded code snippets that aren't live code). Run both before
committing.

## Code conventions

- Type-hinted throughout, modern syntax (`np.ndarray | None`, not `Optional`).
- Stateful components are plain classes with explicit `__init__` (e.g.
  `KinematicMPC` in [control/mpc.py](control/mpc.py), `ConstantVelocityKF` in
  [tracking/kf.py](tracking/kf.py)); no dataclasses anywhere in the codebase.
- Pure computations are module-level functions, not classes (e.g.
  `panda_fk_numpy` in [control/panda_kinematics.py](control/panda_kinematics.py),
  `segment_object_centroid` in [perception/segment.py](perception/segment.py),
  `solve_intercept` in [planning/intercept.py](planning/intercept.py)).
- Numerics: numpy for arrays/linear algebra, CasADi (`ca.SX`, `ca.Opti`) for
  the symbolic FK and MPC formulation.

## The docstring convention (read before changing behavior)

Module and class docstrings here document **numbered deviations from the
original spec/task briefs with the rationale for each**, because this is a
personal upskilling project built task-by-task. These aren't boilerplate —
they explain non-obvious tuning that will look like a bug otherwise. Read
them before touching the file. Examples:
[control/mpc.py](control/mpc.py) (warm-start oscillation fix, posture
nullspace, terminal cost), [run_conveyor_demo.py](run_conveyor_demo.py) (6
deviations: camera transform, joint limits, staging, close-range switch,
Z-clearance), [sim/conveyor_scene.py](sim/conveyor_scene.py) (8 deviations:
MuJoCo XML includes, actuator types, gripper remap), and
[control/panda_kinematics.py](control/panda_kinematics.py) (a DH-parameter
bug that was caught via ground-truth cross-check against MuJoCo).

When you make a behavior-changing fix, document it the same way (a numbered
entry + short rationale in the docstring), don't just change the code
silently.

## Known pitfalls

- **Depth is systematically biased**, not just noisy: segmented centroid
  lands on the visible top face, `~0.02m` above true center — corrected via
  `depth_bias` in [perception/segment.py](perception/segment.py).
- **Real per-joint limits are asymmetric** (e.g. joint4 is entirely
  negative, joint6 entirely positive) — always read from
  `env.model.jnt_range`, never assume `±q_max`.
- **Arm actuators are position-servos**, not velocity-controlled — `step()`
  integrates an internal `q_target` and drives to it.
- **Gripper ctrlrange is `0–255`** (0=closed, 255=open), a remap of the
  physical 0–0.04m range.
- Close-range interception (`solve_intercept`) ill-conditions as remaining
  distance shrinks; below `_CLOSE_RANGE_M` the demo switches to tracking raw
  KF position instead of a lookahead intercept.
- The full accuracy ceiling (~7.1cm grasp error, why it isn't 3cm, and what
  a genuine improvement would need) is root-caused in
  [design spec §12](docs/superpowers/specs/2026-07-29-dynamic-object-tracking-manipulation-design.md#12-demonstrated-accuracy--known-limitation) —
  don't re-tune `grasp.position_tolerance` without reading it first.

## Config

[configs/conveyor.yaml](configs/conveyor.yaml) is nested YAML loaded once via
`yaml.safe_load()` in `run_conveyor_demo.py`, then passed around as plain
dicts (each stage does `config["kf"]`, `config.get("posture_weight", 0.0)`,
etc.) — no schema/validation layer. Inline comments in the YAML explain why
each value was tuned; preserve them when editing.

## Testing

Tests mirror the source layout 1:1 (`tracking/kf.py` →
[tests/tracking/test_kf.py](tests/tracking/test_kf.py)). Use
`np.testing.assert_allclose(..., atol=...)` or `pytest.approx()` for numeric
assertions, not `==`. [tests/test_integration_conveyor.py](tests/test_integration_conveyor.py)
is the end-to-end regression check — it runs a full closed-loop episode and
asserts `grasped` plus `grasp_error_m <= position_tolerance + 0.01`; treat
it as the source of truth for "did my change break grasping."

## Docs

- [Design spec](docs/superpowers/specs/2026-07-29-dynamic-object-tracking-manipulation-design.md) — architecture, novelty positioning, §12 accuracy writeup.
- [Implementation plan](docs/superpowers/plans/2026-07-29-conveyor-mvp.md) — task-by-task build history.
