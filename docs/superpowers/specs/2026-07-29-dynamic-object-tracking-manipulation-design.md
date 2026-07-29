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
- Operational-space/impedance control with a real dynamic model (i.e.
  controlling to the actual position-servo/contact dynamics rather than a
  kinematic-only integrator model) — see Section 12 for why the MVP's
  kinematic MPC cannot close its residual accuracy gap without this
- Active-vision re-acquisition during final approach (re-confirming the
  object's position from a fresh, close-range measurement partway through
  the descent, rather than committing to a single hold-still track taken
  from the reset pose) — see Section 12
- Defining the target/grasp-commit gate/reported accuracy metric
  consistently at the tool-center-point (fingertip) instead of the Panda's
  flange (`hand`) frame — the pipeline currently commands and measures the
  flange throughout, ~0.10m from the actual fingertip contact point; see
  Section 12 for why this is not a simple offset fix

## 12. Demonstrated Accuracy & Known Limitation

**Achieved real grasp accuracy: ~10.8cm** true (ground-truth) end-effector-
to-object error **at the instant `GraspExecutor` actually commits the grasp**
— this is the number the shipped system delivers, measured directly against
`env.get_object_ground_truth()` (not the perception/tracking estimate), and
it is what `tests/test_integration_conveyor.py` verifies passes with margin
at the shipped `grasp.position_tolerance: 0.11`. This, not the closest-
approach figure below, is the MVP's headline, honest, demonstrated result —
not the originally-specified `grasp.position_tolerance: 0.03` (3cm) target.

A separate, smaller number appears throughout Tasks 12-13's reports and is
easy to mistake for the achieved result: **~6-9cm** is the closest true
distance the end-effector *ever reaches* to the object over a full episode
(0.061m at Task 12's `home`-keyframe reset pose, 0.064m at Task 13's
shrunk-excursion `_RESET_QPOS`) — but at the original 0.03 tolerance that
this figure was measured under, `GraspExecutor.should_close` **never actually
fires** within the step budget (`grasped: False` on every run). No grasp is
committed at that 6-9cm figure; it is the best-case approach distance the
system reaches but declines to act on, not a delivered result. The system
only ever actually grasps at the looser, shipped 0.11 tolerance, and only at
the larger ~10.8cm error described above.

**Task 14's key empirical finding, and why 10.8cm ≠ 6-9cm:**
`GraspExecutor.should_close` gates the grasp-commit instant on distance to
the live Kalman-filter *estimate*, not ground truth, so `grasp.
position_tolerance` controls not only whether a grasp is ever attempted but
*when* — a looser gate commits earlier, against a less-converged MPC
solution, which increases the true error at the commit instant relative to
the closest approach the same trajectory would eventually reach. A
fine-grained sweep (0.03–0.15m, all fully deterministic — this simulation has
no randomness anywhere) found a stable plateau from ~0.098m to ~0.114m where
the gate reliably commits at true error 0.108m (step 1675, byte-
reproducible); `grasp.position_tolerance` is set to **0.11m** as the point in
that plateau with safety margin on both sides. See `task-14-report.md` for
the full sweep.

**Why the gap exists:**
1. **Kinematic-only MPC vs. real position-servo dynamics.** `control/mpc.py`'s
   `KinematicMPC` plans as if `q̇ = u` were directly realizable (an idealized
   velocity-controlled integrator). The real Panda actuators in this MuJoCo
   model are position-servos with their own PD dynamics (Section 3.5 already
   flags this as a deliberate simplification for tractability) — the plant the
   MPC actually drives does not match the model it plans against, producing a
   genuine, unmodeled steady-state tracking gap, worse than pure numerical
   convergence error.
2. **A systematic, correctable perception bias — not an irreducible noise
   floor (now corrected in code).** The camera-frame-to-world-frame centroid
   measurement, checked directly against ground truth, has a measured
   residual of ~2-2.5cm — but this is *not* unstructured noise that
   "averages out" or is irreducible by tuning. It was measured as a stable,
   systematic **+0.020m offset concentrated in the Z axis alone**, consistent
   across different arm poses, and it is exactly the conveyor object's box
   half-height (`OBJECT_HALF_HEIGHT_M = 0.02` in `sim/conveyor_scene.py`):
   the segmentation centroid lands on the visible *top face* of the box, not
   its volumetric center, because the wrist camera looks down at the object
   from above. Since the object's geometry is known by design (see Section
   7's "object geometry is known" constraint, which this project already
   relies on elsewhere, e.g. Section 8/manipulation grasp-pose logic), this
   was a one-line correctable offset, not an inherent sensing limitation —
   and has been fixed: `perception/segment.py`'s `segment_object_centroid`
   now takes an optional, additive `depth_bias` parameter (default 0.0,
   backward-compatible with existing callers/tests), applied to the raw
   camera-frame depth *before* deprojection (so it also proportionally
   corrects the deprojected x/y, not just z). `run_conveyor_demo.py` passes
   `sim.conveyor_scene.OBJECT_HALF_HEIGHT_M` for this parameter. Measured
   directly against ground truth over the same static-hold window used to
   discover the bias, this cut the mean per-measurement residual from
   ~0.0205m (with a +0.0196m Z-axis mean, i.e. almost entirely the top-face
   bias) to ~0.0095m (Z-axis mean ~-0.0004m, i.e. the systematic component is
   gone; what remains is genuine, unbiased, sub-centimeter x/y noise). This
   perception improvement does not, on its own, move the headline ~10.8cm
   grasp-commit figure above by a meaningful amount — the flange/TCP offset
   (item 3 below) and the kinematic-MPC/real-actuator-dynamics mismatch
   (item 1 above) dominate the remaining gap — but it is a genuine, verified
   fix to a previously-mischaracterized error source, and it should not be
   re-introduced or re-described as "noise" in future work on this pipeline.
3. **The flange (`hand`) frame, not the fingertip/TCP, is what is commanded,
   gated on, and reported throughout this pipeline — undisclosed until now.**
   `panda_fk_numpy`/`panda_fk_symbolic` (and therefore the MPC's target, the
   grasp-distance gate, and every accuracy number in this section) all use
   the Panda's `"hand"` (flange) body frame, not the actual fingertip contact
   point. Measured directly against the open gripper's fingertip pad geoms at
   the shipped reset pose, this offset is **~0.10m** (0.109m; distinct from,
   and larger than, the ~6cm hand-to-finger-*body*-origin figure noted in
   `task-12-report.md`'s finding #8, which measured to the finger body's
   joint origin rather than the pad tip). Every accuracy number in this
   section — the ~10.8cm headline, the 6-9cm closest-approach figure, and the
   tolerance sweep — is stated in terms of this flange frame, not the TCP
   where contact actually happens. This is a specific, nameable contributor
   to the reported gap, distinct from (2) above, and it is **not** a cheap
   fix: the reviewer-tested attempt of simply raising `_Z_CLEARANCE_M` (the
   Z-clearance offset in `run_conveyor_demo.py`) to compensate made results
   *worse*, not better, because `GraspExecutor.should_close`'s gate is
   computed against the same flange frame as the target, so shifting one
   without consistently redefining the other just moves the gate rather than
   closing the flange-to-TCP gap.
4. **Required arm excursion from reset pose to conveyor height.** The arm
   must travel a large distance (~0.57m at the `home` keyframe, ~0.24m at
   Task 13's shrunk-excursion reset pose) from its resting configuration down
   to the conveyor's operating height, which destabilizes the eye-in-hand
   wrist camera's orientation en route (the reconfiguration needed to reach
   downward inherently reorients the whole downstream kinematic chain,
   including the camera) and interacts with real, unmodeled floor/gripper
   contact forces near the bottom of the descent.

**What was tried to close the gap** (see `task-12-report.md` and
`task-13-report.md` for full detail): Task 12 ran an exhaustive tuning sweep
across effort weight, terminal weight, posture weight, control frequency,
MPC horizon, intercept lead time, Z-clearance offset, and track-confirmation
thresholds — no combination reliably produced true error under ~0.06m. Task
13 tested the specific hypothesis that shrinking the required excursion (by
changing the arm's reset joint configuration) would close the gap; a
systematic sweep of reset-pose heights from ~0.11m to ~0.55m found accuracy
is *non-monotonic* in excursion size (the largest excursion, `home` itself,
produced the best accuracy in the whole sweep) — because the eye-in-hand
camera's ground-footprint field of view shrinks along with the excursion,
cutting how long the object stays visible before the tracker's m/n
confirmation gate can even fire. Both investigations independently converged
on the same conclusion: this is a structural accuracy ceiling of the
kinematic-MPC + position-servo-actuator + classical-perception architecture,
not a remaining tuning knob.

**What a genuine fix would require:**
- An operational-space/impedance controller built against the arm's real
  dynamic model (rather than a kinematic-only integrator assumption).
- Active-vision re-acquisition during the final approach (re-confirming the
  object's position from a fresh, close-range measurement rather than
  relying on a single hold-still track established from the reset pose).
- **Defining the target, the grasp-commit gate, and the reported accuracy
  metric consistently at the tool-center-point (fingertip) instead of the
  flange** — a specific, nameable candidate item alongside the two above,
  motivated directly by finding (3) above. As noted there, this is not a
  drop-in offset fix (raising the Z-clearance alone was tested and made
  things worse, since the gate itself is flange-based); it would require
  redefining the FK target frame, the MPC's Cartesian cost, and
  `GraspExecutor`'s distance check all together, at the TCP.

All three items above are recorded in Section 11 as future work, explicitly
out of this MVP's scope.
