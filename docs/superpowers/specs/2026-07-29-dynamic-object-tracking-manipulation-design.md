# GAUGE: Gated Uncertainty-Aware Grasping Engine — Design Spec

*Working name: **GAUGE**. Easily renamed later — alternatives considered in Naming section below.*

## 1. Purpose & Scope

Personal upskilling / portfolio project (not a thesis). Goal: build a complete
perception-to-action pipeline on a simulated Franka Emika Panda that tracks a
moving object with an eye-in-hand RGB-D camera, estimates its hidden state,
predicts its trajectory, plans an interception point, and executes a dynamic
grasp — synchronizing with the object rather than requiring it to stop.

This extends an Advanced Mobile Robotics course covering hidden state
estimation, motion models, KF/EKF/UKF/PF, gating, track initiation/termination,
m/n logic, and data association, into the manipulation domain.

**Constraints established during brainstorming:**
- Simulation-only (no real robot access, ever, for this project)
- Personal project, not a thesis — bar is a well-engineered, demo-able,
  portfolio-quality system, not a novel research contribution
- Timeline: 1-2 month focused sprint
- Background: Python-first, little-to-no C++, some prior ROS2 exposure,
  comfortable with control theory / optimization (MPC, CasADi-style tools)
- Hardware: NVIDIA RTX PRO 2000 (Blackwell), 8GB VRAM, Windows 11 Enterprise,
  CUDA 13.2
- Architecture decision: **pure Python pipeline, no ROS2** in the core system
  (explicitly chosen over ROS2-as-middleware to avoid WSL2/Linux setup
  overhead eating into the sprint budget, and to match the Python-first /
  low-C++ background)

## 2. Novelty Positioning

Closest existing work:
- Menon, Bekris et al., "Dynamic Grasping with Reachability and Motion
  Awareness" (Rutgers, ICRA 2022) — Panda-class arm intercepting a moving
  conveyor object; the closest direct analog to the conveyor MVP here.
- Bäuml, Wimböck, Hirzinger (DLR) — kinematically optimal ball catching.
- Kim & Perez (MIT) — probabilistic dynamic-grasp planning under motion
  uncertainty.
- Table-tennis/badminton robot literature — prediction under uncertainty in a
  fast interception setting.
- AnyGrasp / GraspNet / Contact-GraspNet — static grasp-pose generators for
  *unknown* objects at a *frozen* snapshot. Relevant only as a contrast: they
  don't solve the actual problem here (grasp pose of a moving object), which
  is itself worth noting explicitly in any write-up.

**Chosen differentiator: uncertainty-aware interception commit.** The
manipulator does not chase a raw point estimate. It only commits to an
interception trajectory once the track is m/n-confirmed *and* the estimate's
covariance has shrunk below a threshold; the MPC cost is covariance-weighted.
This is the natural bridge from the tracking course (gating, m/n logic, track
confirmation) into manipulation, is cheap to build given the estimation
machinery already learned, and produces a concrete ablation: naive
point-tracking commits early and mis-grasps near high-uncertainty regions
(e.g. a pendulum's turning points), while gated commit does not.

**Explicitly deferred to "future work"** (not in MVP scope): active-perception
camera-path optimization (trading interception speed for keeping the object
in frame), multi-object tracking/grasping, unknown/arbitrary trajectories.

**MVP target sequencing:**
1. **Conveyor belt** (constant-velocity, linear KF) first — gets the full
   pipeline working end-to-end quickly (an early working demo matters for
   momentum and portfolio value).
2. **Pendulum** (nonlinear, EKF/UKF) second, within the same sprint if time
   allows. Built-in ablation: a physically-parameterized pendulum-state UKF
   (state `[θ, θ̇]`, nonlinear measurement model through the known
   pivot/length) should visibly outperform a naive constant-velocity Cartesian
   KF at the swing's turning points — a well-known effect, but demonstrating
   it hands-on is the right scope and makes a strong write-up figure.

## 3. System Architecture

Modular pipeline; each stage is a standalone Python module with a narrow
interface, independently unit-testable before the full loop is wired together.

```
Perception → Tracking/Estimation → Prediction → Interception Planner → MPC Controller → Grasp Executor
                                        ↑                                      |
                                        └──────────── replan loop ─────────────┘
```

| Stage | Input | Output | Depends on |
|---|---|---|---|
| Perception | RGB-D frame (eye-in-hand) | 3D object centroid (camera frame) | MuJoCo renderer |
| Tracking | Raw 3D measurement + track state | Filtered state estimate + covariance, track status (tentative/confirmed/lost) | Perception |
| Prediction | Confirmed track state + covariance | Predicted trajectory + covariance over MPC horizon | Tracking |
| Interception planner | Predicted trajectory, EE reachability | Target intercept point + time | Prediction |
| MPC controller | Intercept target, current robot state | Joint velocity trajectory | Interception planner |
| Grasp executor | EE pose vs. intercept pose, track confidence | Gripper open/close command | MPC controller, Tracking |

### 3.1 Perception
- **MVP:** classical color+depth segmentation on the synthetic RGB-D frame →
  connected components → centroid → back-projected to 3D via known camera
  intrinsics/extrinsics.
- **Stretch:** swap in a YOLOv8-nano detector trained on auto-labeled sim
  frames (MuJoCo provides ground-truth masks essentially for free, so labeling
  cost is near zero).

### 3.2 Tracking
- Linear KF (constant-velocity, 6-state) for the conveyor.
- UKF with physically-parameterized pendulum state `[θ, θ̇]` and nonlinear
  measurement model for the pendulum, benchmarked against a naive Cartesian
  CV-KF baseline.
- Mahalanobis-distance gating on innovations; m/n logic (e.g. 3-of-5) for
  track confirmation — reused directly from the course.

### 3.3 Prediction
Propagate mean + covariance forward through the motion model over the MPC
horizon (closed-form for CV; sigma-point propagation for the pendulum UKF).

### 3.4 Interception planning
- CV case: closed-form time-to-intercept (quadratic in `t`, respecting EE
  reachability/velocity limits).
- Pendulum case: numeric line search over candidate intercept times within
  the predicted horizon.
- The "commit" step is gated on track-confirmed + covariance-below-threshold
  (the core novelty axis).

### 3.5 Control
- **Kinematic MPC** via CasADi + IPOPT (pure Python, no C++ build step).
  Controls joint velocities (`q̇ = u`); tracks the intercept pose via forward
  kinematics written directly as CasADi symbols (Panda's 7-DOF FK — no
  external dependency, no Windows install risk).
- MuJoCo still simulates full dynamics/contacts/rendering as the "real
  world" — only the MPC's internal predictive model is simplified to
  kinematics. Standard practice for reaching-type tasks; keeps the CasADi/
  IPOPT problem small enough to solve in real time on the available hardware.
- Cost: track predicted intercept pose + effort/jerk penalty + joint-limit/
  workspace constraints. Receding horizon, replanned on each track update.
- *(A full torque-level dynamic MPC via Pinocchio was considered and rejected
  — real build-toolchain risk on Windows with limited C++ background, for
  benefit not needed at this project's scope.)*

### 3.6 Grasp execution
Fixed relative grasp pose computed from the object's predicted orientation
(object geometry — cylinder/box — is known in sim, so AnyGrasp/GraspNet-style
unknown-object grasp generation is unneeded complexity). Gripper closes when
EE is within tolerance of the intercept pose.

## 4. Biggest Technical Challenges

1. **Closed-loop latency budget** — detection→estimation→prediction→
   planning→control each add delay; total loop delay vs. object speed
   determines whether interception is geometrically feasible at all. Should
   be instrumented/logged from day one.
2. **MPC real-time solve** — CasADi/IPOPT solved every replan step is the
   most likely bottleneck. Mitigate with short horizons + warm-starting from
   the previous solve.
3. **Eye-in-hand coupling** — moving toward interception changes what the
   camera sees (motion blur, object leaving frame, degraded viewing angle) —
   a genuine perception/control coupling, not just an engineering nuisance.
4. **Grasp timing** — deciding the close-gripper trigger from a moving,
   uncertain estimate rather than a static known pose.

## 5. Simulator Choice

| Simulator | Assessment |
|---|---|
| **MuJoCo (chosen)** | Native Windows Python bindings, CPU-bound physics (8GB VRAM irrelevant), built-in offscreen RGB-D rendering, Panda+gripper MJCF model available from MuJoCo Menagerie, large Python ecosystem, fastest iteration, no Linux/WSL2 needed. |
| Isaac Sim | Needs Linux for full feature set; 8GB VRAM is tight for its RTX renderer; steep install/learning curve. Right tool for RL-at-scale or photorealism later — overkill/risky for a 1-2 month sprint. |
| Gazebo Harmonic | Built around ROS2 integration — ROS2 was explicitly ruled out, so its main advantage doesn't apply. Weaker RGB-D realism, Linux-native. |
| Drake | Strong optimization/MPC pedigree but steeper learning curve, smaller community/resume recognition than MuJoCo. |
| Newton Physics | Too new/immature as of 2026 to bet a short sprint on. |

**Decision: MuJoCo**, via the `mujoco` Python package + MuJoCo Menagerie's
Panda model, using its built-in offscreen renderer for synthetic RGB-D.

## 6. Python Environment

**uv**, chosen over conda/venv/poetry. Conda's main historical advantage in
robotics (RoboStack's prebuilt ROS2 conda-forge packages) doesn't apply since
ROS2 was ruled out. uv is fast, has a proper lockfile for reproducibility,
handles PyTorch's CUDA wheel index cleanly, and works natively on Windows.

## 7. Technology Stack

| Included | Why |
|---|---|
| `mujoco` | Sim + rendering |
| `numpy`, `scipy` | Filtering math (KF/UKF) |
| `casadi` | NMPC (ships prebuilt IPOPT — no C++ build step) |
| `opencv-python` | Classical perception (segmentation, camera math) |
| `torch` + `ultralytics` | *Stretch only* — YOLOv8-nano on auto-labeled sim frames |
| `matplotlib` / `pyqtgraph` | Real-time plots: ground truth vs. estimate vs. prediction, covariance ellipses |
| `wandb` | Experiment tracking across filter/target ablations (free tier) |
| `omegaconf` + `hydra-core` | Config + multirun sweeps (used for the CV-KF vs. EKF vs. UKF ablation) |
| `pytest` | Per-module unit tests |
| `uv` | Env/dependency management |

**Explicitly excluded, and why:**
- ROS2 — decided against; Windows/WSL2 setup overhead against a tight
  timeline, no core-architecture need once pure-Python was chosen.
- MoveIt2 / OMPL — no sampling-based motion planning needed for a smooth MPC
  reaching task.
- Pinocchio / acados / Eigen / C++ toolchain — avoided via the kinematic-MPC
  simplification (Section 3.5); real Windows build risk for benefit not
  needed at this scope.
- Isaac Sim / Gazebo — see Section 5.
- AnyGrasp / GraspNet / SAM2 — solve a different problem (static
  unknown-object grasping); object geometry is known in sim here.

## 8. Project Structure

```
dynamic-object-tracking/
├── perception/        # RGB-D → 3D centroid (classical CV; optional YOLO swap)
├── tracking/           # KF / EKF / UKF, gating, m/n confirmation
├── prediction/         # forward propagation of state + covariance
├── planning/            # interception point/time solver
├── control/             # CasADi kinematic MPC
├── manipulation/     # grasp pose logic, gripper trigger
├── sim/                    # MuJoCo scenes (conveyor, pendulum), Panda assets
├── configs/            # Hydra/OmegaConf YAML (target type, filter type, MPC params)
├── experiments/     # ablation run scripts (CV-KF vs EKF vs UKF, gated vs naive commit)
├── logs/                  # wandb + local run artifacts
└── tests/                 # pytest, one module at a time
```

## 9. Milestones (6-8 Week Sprint)

1. **Wk1** — MuJoCo scene: Panda + gripper + conveyor + eye-in-hand camera;
   sanity-check kinematics/rendering.
2. **Wk2** — Perception module + logging/plotting harness.
3. **Wk3** — Conveyor CV-KF tracking with gating + m/n confirmation,
   validated against ground truth.
4. **Wk4** — Interception planner + kinematic MPC → **first end-to-end
   conveyor grasp (MVP milestone)**.
5. **Wk5** — Pendulum scene + naive Cartesian KF baseline + physically-
   parameterized UKF; run the ablation.
6. **Wk6** — Wire pendulum into full pipeline → **end-to-end pendulum
   grasp**.
7. **Wk7-8 (stretch)** — gated-vs-naive-commit ablation write-up, optional
   YOLO perception swap, polish + demo video + README.

## 10. Naming

Working name: **GAUGE** — *Gated Uncertainty-Aware Grasping Engine*. Directly
names the actual novelty axis (gated, uncertainty-aware commit); "gauge" also
reads as "to judge/measure," fitting an estimation-heavy project.

Other strong candidates considered:
- **CONFIRM** — plays on "track confirmation" (the literal m/n-logic term
  from the course) while reading as an ordinary word implying trust-before-
  acting.
- **KESTREL** — no forced acronym (AnyGrasp-style). Kestrels hover in place
  watching prey before diving — a strong metaphor for an eye-in-hand system
  that tracks-then-commits.
- **GateGrasp** — AnyGrasp-style compound name, immediately legible as a repo
  name.
- **RAPTOR** — *Reactive Adaptive Pursuit and Tracking for Object Retrieval*
  — punchy, bird-of-prey metaphor for precision interception.

Longer list considered: TRACE, DIME (Dynamic Interception via Motion
Estimation), INCEPTOR, GAMBIT (Gated Adaptive Model-predictive Bayesian
Interception & Tracking), STRIKE, RAVEN, PursuitGrasp, InterceptNet,
DynaGrasp, ChaseGrasp, SwiftCatch, OrbitGrasp, FlowCatch, MovingTargetNet,
PREY (Predictive Robotic intErception sYstem), FALCON, GRIP-T, VelocityGrasp,
ReflexGrasp, ClosedLoopCatch.

## 11. Explicitly Out of Scope (Future Work)

- Active-perception camera-path optimization (trading interception speed for
  keeping the object in frame)
- Multi-object tracking and grasping
- Unknown/arbitrary target trajectories
- Real robot deployment / sim-to-real transfer
- ROS2 wrapper layer around the finished Python modules
- Full torque-level dynamic MPC (Pinocchio-based)
