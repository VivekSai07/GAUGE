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

**What actually shipped vs. what this section describes (Task 15 correction):**
m/n-confirmed track gating is implemented and active (`TrackStatus.CONFIRMED`
is required by `GraspExecutor` whenever `confidence_required=True`, the
default). A covariance-threshold gate on the grasp-commit decision was added
in Task 15 (`GraspExecutor(cov_threshold=...)`, gating on
`trace(track.kf.P[:3,:3])`) — but it is **off by default**
(`cov_threshold=None`) and the shipped `configs/conveyor.yaml` does not
enable it, so it has no effect on the demonstrated Section 12 accuracy
figures. The MPC-cost covariance-weighting described above was never
implemented — see Section 11.

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
- **[Implemented — see Section 12]** Defining the target/grasp-commit
  gate/reported accuracy metric consistently at the tool-center-point
  (fingertip) instead of the Panda's flange (`hand`) frame was listed here
  as future work; it has since been implemented (`panda_tcp_numpy`/
  `panda_tcp_symbolic` in `control/panda_kinematics.py`), prompted by a
  user-reported visual grasp failure. Kept here struck through rather than
  deleted so the history is legible.
- **[Implemented — see Section 12, "Round 3"]** Verified grasp *completion*
  (not just commit-instant distance) was listed here as future work; it has
  since been implemented (`ConveyorSceneEnv.is_grasped()`, both fingers
  simultaneously in contact via MuJoCo's contact array), prompted by the
  same user report that motivated Round 3. `tests/test_integration_conveyor.py`
  now asserts `contact_verified` directly. Kept here struck through rather
  than deleted so the history is legible. What's still not covered: holding
  the grasp through any *subsequent* motion (e.g. lifting/carrying the
  object away) — only a sustained static hold post-commit is verified.
- Full end-to-end "place" behavior (carrying the grasped object to a drop
  location) — this MVP's scope was always "pick" (grasp commit + verified
  hold), not pick-*and*-place to a destination; a "place" phase was never
  specified or built.
- `prediction.predict.propagate()` (Section 3.3) is fully built and tested
  but is not imported or called anywhere in `run_conveyor_demo.py` — the
  MVP's closed-form constant-velocity interception solve
  (`planning.intercept.solve_intercept`) doesn't need step-wise state/
  covariance propagation to compute a rendezvous point, so the live loop
  never calls it. Reserved for the pendulum follow-on, where a nonlinear,
  non-closed-form trajectory genuinely needs multi-step propagation to plan
  an interception point.
- MPC-cost covariance-weighting (Section 2's original novelty framing: "the
  MPC cost is covariance-weighted") was never implemented. What did ship
  (Task 15) is a covariance-*threshold gate* on the grasp-commit decision
  (`GraspExecutor(cov_threshold=...)`, tested, but off by default and not
  enabled in `configs/conveyor.yaml`) — a strictly simpler mechanism than
  weighting the MPC's cost function by estimate uncertainty, which remains
  unbuilt.

## 12. Demonstrated Accuracy & Known Limitation

**Achieved real grasp accuracy: ~4.4cm** true (ground-truth) fingertip-to-
object error **at the instant `GraspExecutor` actually commits the grasp**,
at `grasp.position_tolerance: 0.035` — this is the number the shipped system
currently delivers, measured directly against `env.get_object_ground_truth()`
(not the perception/tracking estimate) and against the real fingertip-pad
position (not the flange), and it is what `tests/test_integration_conveyor.py`
verifies passes deterministically. This is very close to the originally-
specified `grasp.position_tolerance: 0.03` (3cm) target. **This supersedes
every number earlier in this section** (~7.1cm, ~10.8cm, ~6-9cm) — see "Round
2: closing the gap" below for what changed and why. The rest of this section
up to that point is preserved as accurate history of how the team arrived at
the ~7cm figure and why it was believed, at the time, to be a structural
ceiling; it wasn't.

Note the ~4.4cm figure is still a *commit-instant distance*, not a verified
successful grasp: `run_conveyor_demo.py` commands the gripper closed and
holds a few more simulation steps so the motion visibly plays out, but
nothing in this pipeline verifies the fingers actually make and hold contact
(no check on `env.data.ncon`/`env.data.contact` after closing) — see
Section 11.

A separate, smaller number appears throughout Tasks 12-13's reports: **~6-9cm**
is the closest true distance the end-effector *ever reaches* to the object
over a full episode (0.061m at Task 12's `home`-keyframe reset pose, 0.064m
at Task 13's shrunk-excursion `_RESET_QPOS`). This is close to, but distinct
from, the achieved grasp-commit accuracy above — it is the best-case approach
distance a trajectory reaches at *some* point, whether or not a grasp is
committed there.

**Task 15's correction to this section (final whole-branch review):** an
earlier revision of this section shipped `grasp.position_tolerance: 0.11`
and described it as sitting in a "stable plateau," with the ~6-9cm
closest-approach figure characterized as a distance the system "declines to
act on." That framing was wrong, and falsifiable by re-running the sweep:
0.11 was not a plateau to rest on, it was the *worst* accuracy among the
tolerances that actually grasp at all. A full re-sweep (0.03m-0.15m, fully
deterministic — this simulation has no randomness anywhere, so every number
below is exactly reproducible) found:

| tolerance | grasped | true error at commit |
|---|---|---|
| 0.03 - 0.055 | False (never fires within `max_steps`) | — |
| 0.06 | True | 0.0677m |
| 0.064 - 0.080 | True | 0.0709m |
| 0.085 - 0.10 | True | 0.0877m |
| 0.102 - 0.13 (old shipped 0.11) | True | 0.1085m |
| 0.14 - 0.15 | True | 0.1396m |

The relationship is close to monotonic: a **tighter** gate commits *later*,
against a more-converged MPC solution, giving a *smaller* true error — all
the way down to ~0.055m, where the gate stops firing within the step budget
entirely. There is no "boundary between closest-approach and committed-grasp"
to speak of; there is a monotonic accuracy/commit-time trade-off, and the old
0.11 setting sat needlessly far on the wrong side of it. `grasp.
position_tolerance` is now **0.075m**, centered in the wide 0.064-0.080m
plateau (true error 0.0709m, step 1725, byte-reproducible), which beats the
old 0.11 setting on both accuracy (7.1cm vs. 10.85cm) and margin against the
integration test's `grasp_error_m <= tolerance + 0.01` bound (~0.0141 vs.
~0.0115).

**Why `position_tolerance` controls more than whether a grasp is attempted:**
`GraspExecutor.should_close` gates the grasp-commit instant on distance to
the *commanded target* (the live Kalman-filter estimate plus a small Z-
clearance offset, or — on the long-range leg of the approach — the
`solve_intercept` lookahead point; see `run_one_episode`'s `target`
computation), not ground truth. So `grasp.position_tolerance` controls not
only whether a grasp is ever attempted but *when* — a looser gate commits
earlier, against a less-converged MPC solution, which increases the true
error at the commit instant relative to the closest approach the same
trajectory would eventually reach. See `task-14-report.md` and
`task-15-report.md` for the full sweeps.

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
   gone; what remains is genuine, unbiased, sub-centimeter x/y noise). At the
   time this fix landed (Task 14), it was assessed as not moving the headline
   grasp-commit figure by a meaningful amount — but that assessment was made
   without re-optimizing `grasp.position_tolerance` afterward, at the old,
   now-known-suboptimal 0.11 setting. Task 15's correction above shows that
   once `position_tolerance` is re-swept post-fix, the achievable accuracy
   moves down substantially, from 0.1085m to 0.0709m (a ~35% reduction) — so
   this perception fix *did* matter, it just wasn't visible until paired with
   re-tuning the tolerance that gates on it. It is a genuine, verified fix to
   a previously-mischaracterized error source, and it should not be
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
   section — the ~7.1cm headline, the 6-9cm closest-approach figure, and the
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
   must travel a large distance (~0.57m at the `home` keyframe, ~0.33m at
   Task 13's shrunk-excursion reset pose -- a 42% reduction versus `home`,
   not the ~0.24m sometimes quoted, which is the reduction in flange height
   versus `home` rather than the remaining descent to the conveyor) from its
   resting configuration down
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

**What was believed, at the time, to require a genuine architecture change**
(now partly done — see "Round 2" below):
- An operational-space/impedance controller built against the arm's real
  dynamic model (rather than a kinematic-only integrator assumption). —
  *Still not done; remains real future work, recorded in Section 11.*
- Active-vision re-acquisition during the final approach. — *Still not done;
  remains real future work, recorded in Section 11.*
- Defining the target, the grasp-commit gate, and the reported accuracy
  metric consistently at the tool-center-point (fingertip) instead of the
  flange. — ***Done, see below.*** At the time this was written, the team had
  only tested raising the Z-clearance offset (which made things worse, since
  the gate itself was still flange-based) and concluded a full TCP-consistent
  redefinition was needed but hadn't attempted it. It turned out to be the
  single highest-leverage fix of the three.

---

## Round 2: closing the gap (prompted by a rendered demo, not a metric)

Everything above this point was written when the team's only feedback signal
was the numeric `grasp_error_m` output and a fine-grained tolerance sweep.
The gap was reopened by a much blunter signal: running
`uv run python run_conveyor_demo.py --render` and *watching* the episode.
The gripper visibly closed well clear of the object, and the episode ended
without anything resembling a pick. That observation, followed by systematic
root-cause investigation (per `superpowers:systematic-debugging`) rather than
another tuning pass, found three real, fixable problems — not one bigger
version of the same "structural ceiling":

1. **The conveyor object was a `mocap` body.** MuJoCo `mocap` bodies are
   kinematically scripted and are not affected by contact or gripper forces
   at all. Closing the gripper "around" one never grasps it in any physical
   sense, independent of targeting accuracy — it keeps sliding along its
   scripted trajectory regardless of what the gripper does. This was found
   by comparing this project's conveyor implementation against an external
   reference (`github.com/felixokolo/MuJoCo_tutorials/1/conveyor.xml`, MIT/
   public tutorial repo), which drives its object via a `free` joint plus a
   `velocity` actuator instead — a real, physically-simulated body. Fixed in
   `sim/conveyor_scene.py`: the object is now a `free`-jointed body with two
   `velocity` actuators (world X/Y), resting on a newly-added static
   platform sized so it settles at the same operating height (z=0.05) the
   rest of the pipeline already assumed. Still exactly constant-velocity by
   design (the actuators hold a fixed commanded velocity for the whole
   episode) — but now a real body the gripper can actually nudge, grip, and
   hold, which is a *prerequisite* for any real pick-and-place, independent
   of accuracy.

2. **A hard discontinuity in the target definition.** `run_one_episode`'s
   approach logic switched abruptly between two different target
   definitions at `live_dist == _CLOSE_RANGE_M`: a `solve_intercept`
   lookahead point (which deliberately leads the object, extrapolating
   its future position) above that threshold, and the raw live Kalman
   estimate (which does not lead it) below. Instrumented logging of a full
   episode showed the target's along-track coordinate jumping by roughly
   0.10m at that exact boundary — the arm, still converging toward the
   pre-switch (leading) target, would overshoot past the object right where
   the switch fired, which is exactly what the rendered run showed
   visually. Fixed by replacing the hard switch with a continuous blend
   (`blend = clip(live_dist / _CLOSE_RANGE_M, 0, 1)`, linearly interpolating
   between the lookahead point and the live estimate) — `target` now moves
   continuously with no jump.

3. **Flange-vs-TCP was not just under-disclosed, it was actionable.**
   Section 12's earlier text (finding 3, above) treated the flange/TCP gap
   as a real but not-cheaply-fixable contributor. Once (1) and (2) were
   fixed, direct measurement showed the *fingertip*-level error was already
   far better than the flange-based number suggested (~5cm vs. a ~9cm flange
   floor) — the flange number was dominated by the flange simply sitting
   ~0.10m above the object in Z, not by genuine misalignment. This meant
   consistently redefining target/gate/metric at the TCP wasn't a marginal
   correction, it was the actual fix. Implemented via two new functions in
   `control/panda_kinematics.py`, `panda_tcp_numpy`/`panda_tcp_symbolic`,
   extending the existing flange FK by one fixed additional translation
   (`_TCP_OFFSET_Z = 0.1029`, the distance from the flange origin to the
   midpoint between the two fingertip pads along the flange's local Z axis
   — verified fixed and pose-independent to floating-point precision, since
   both fingers open/close symmetrically about that axis). `run_conveyor_demo.py`
   now uses `panda_tcp_*` everywhere `panda_fk_*` was previously used: the
   MPC's own Cartesian cost (`KinematicMPC`'s `fk_func`) now optimizes
   fingertip position directly, `GraspExecutor.should_close` gates on
   fingertip distance, and `grasp_error_m` reports fingertip-to-object
   distance. The Z-clearance offset that (1) originally needed (to keep the
   flange-following approach from driving the fingers through the floor) is
   removed entirely — targeting the object's own center height is now
   correct, since that's where the fingertips should be.

**Result (at the time — see "Round 3" below for the correction):** a full
deterministic re-sweep of `grasp.position_tolerance` on the fixed code found
the achievable accuracy is dramatically better than the ~6-9cm figure
believed structural above — down to a floor around 2.8cm (where the gate
stops firing reliably), with **0.035 chosen as the shipped value: ~4.4cm
true fingertip-to-object accuracy**, close to the originally-specified 0.03m
target. See `configs/conveyor.yaml`'s comment for the full sweep table. All
47 tests pass, `uv run python run_conveyor_demo.py` reproduces the same
result byte-for-byte across repeated runs (fully deterministic, no
randomness anywhere in this simulation).

**What this does and doesn't mean:** the operational-space/impedance-control
and active-vision-reacquisition items above are still real, unimplemented
future work. The lesson is less "the architecture was fine all along" and
more "a tuning sweep against a single aggregate metric can converge on a
local optimum and mistake it for a structural ceiling, when the actual
problem is a specific, fixable mechanism ... that a sweep alone will never
surface." All three fixes here came from watching the system run and tracing
a concrete, reproducible failure back to its mechanism — not from trying
more combinations of the same knobs.

**Correction: this "Result" was itself premature.** Despite the accuracy
number improving, the user reported after actually watching a rendered
episode that the Franka *still never picked up the cube* — the ~4.4cm
figure, and the "grasped: True" it was gated on, were both still only a
commit-instant *distance* check, never a check that anything was physically
held. See "Round 3" immediately below for what that distance-only metric
was hiding, and the fixes that made the grasp physically real.

---

## Round 3: the grasp was never physically real (a distance metric said it was)

Round 2 improved `grasp_error_m` substantially and, from the numbers alone,
looked like a closed case. Watching `--render` immediately showed otherwise:
the gripper closed near the object but never picked it up. Root-caused with
`systematic-debugging` (direct instrumentation, not another tuning pass),
plus a pattern comparison against a working reference the user pointed to
(`github.com/VivekSai07/robot-manipulation-playground`, which has a proven
IK controller and vision pipeline for this exact class of task). Three
separate, real problems, found in sequence as each was fixed:

1. **No orientation control at all.** `KinematicMPC` minimized TCP position
   error only. Direct instrumentation (logging the object's position
   expressed in the gripper's own local frame, `rotation.T @ (object -
   tcp)`, across a full episode) found the object sitting well-centered in
   *aggregate* 3D distance but consistently offset **~3cm along the
   gripper's local X axis** — and closing the fingers (which move only
   along local Y, confirmed empirically and by a ground-truth test against
   `env.data.xpos`) cannot correct an X-axis offset at all. A close TCP
   distance was therefore compatible with the object never being between
   the fingers.

   This is exactly the class of problem the user's reference repo's
   `ik_controller_m2.py` solves: full 6D pose (position + orientation)
   tracking via Jacobian-based differential IK (Pinocchio FK/Jacobians,
   damped least squares, nullspace projection), rather than position-only
   tracking. Rather than adopting Pinocchio (a dependency this project
   deliberately avoided from the start — Section 1's Windows/low-C++
   constraints), the same effect was added within the existing CasADi
   stack: `control/panda_kinematics.py` gained `panda_tcp_pose_symbolic`/
   `panda_tcp_pose_numpy`, exposing the gripper's full orientation (not
   just position) — including discovering and correcting a fixed 45°
   rotation between this project's DH-derived flange frame and MuJoCo's
   own "hand"-frame convention (a well-known Franka Panda quirk, verified
   fixed to <1e-7 across different arm configurations). `control/mpc.py`
   gained `pose_fk_func`/`lateral_axis_weight`: an additive, opt-in cost
   term penalizing the target's offset specifically along the gripper's
   local X axis. `configs/conveyor.yaml`'s `mpc.lateral_axis_weight: 25.0`
   enables it.

2. **The conveyor object's velocity actuators never stopped.** Found
   independently of (1): `sim/conveyor_scene.py`'s `reset()` sets the
   object's commanded velocity once and nothing ever zeroes it — so even a
   mechanically perfect grasp was fighting the object's own actuator
   forever (confirmed by instrumentation: object position continuing to
   drift under its commanded velocity, opposed by gripper contact forces,
   long after the gripper had closed). Fixed with a new
   `ConveyorSceneEnv.stop_conveyor_object()`, called the instant
   `run_conveyor_demo.py` commits a grasp.

3. **Grip friction was too low to hold the object against gravity.** After
   (1) and (2), a real, direct contact check —
   `ConveyorSceneEnv.is_grasped()`, both fingers simultaneously in contact
   via MuJoCo's own contact array, the same pattern as the reference repo's
   `grasp_controller.py::is_grasped` — registered `True` only briefly
   before flipping back to `False`: the object was visibly slipping
   downward and out of the closed gripper under gravity. Raising the
   object's tangential friction coefficient from 1.0 to 3.0
   (`sim/conveyor_scene.py`) fixed this — verified via a long-hold check
   (1000 post-grasp settle steps, not just the normal
   `_POST_GRASP_SETTLE_STEPS`) that contact now holds `True` for a
   sustained ~0.6 second window, not an instant.

**Result:** `run_one_episode()`'s return value gained a `contact_verified`
field, and `tests/test_integration_conveyor.py` now asserts it directly —
the test can no longer pass on distance alone. Verified deterministic and
reproducible across repeated runs:
`{'grasped': True, 'grasp_error_m': ~0.039, 'steps': 2025, 'contact_verified': True}`.
All 53 tests pass (up from 47 — new coverage: pose-FK ground-truth checks,
the lateral-axis-weight cost effect, `stop_conveyor_object`, and both
`is_grasped` cases).

**What this round actually demonstrates, for anyone reading this history:**
a scalar accuracy metric — however honestly reported, however thoroughly
swept — is not the same claim as "the thing works." Round 2's ~4.4cm number
was real and correctly measured, and still described a system that had
never once picked anything up, because nothing was checking the property
that actually mattered (physical contact, sustained). The fix was not "sweep
harder" a third time; it was watching the system run, and — critically —
consulting a second, independent, *working* implementation of the same
problem class rather than re-deriving a fix from first principles alone.

## Round 4: `contact_verified` itself was checked at the wrong moment — and finding that revealed a real, still-open sensing-precision limit

Round 3's `contact_verified` check ran immediately after the gripper closed
and a short settle, while the object was still sitting essentially where it
was grasped. That proves momentary contact, not a real hold — a gripper
closed around an object resting on the platform, without enough grip to
support it once airborne, would pass that check too. Informed by the same
reference repo's `pick_and_place_m13_reactive.py`, which has an explicit
"Verify Lift" state (close → lift ~15cm → only then check `is_grasped`,
resetting on failure), `run_conveyor_demo.py::run_one_episode` gained a real
~10cm lift phase (ramped over 40 MPC control ticks, not a step input) after
grasp-commit, and `contact_verified` now means "held through that lift," not
"touching at the instant of closing." `object_height_gain_m`/
`object_peak_height_gain_m` give direct, checkable proof of whether the
object was actually carried upward. Built via TDD + subagent-driven-development
(`docs/superpowers/plans/2026-08-01-lift-verified-grasp.md`); a final
whole-branch review found and fixed three real issues in the first pass
(the height-gain baseline was sampled after the settle loop instead of at
the grasp-commit instant, contaminating the metric; the lift target jumped
to its full offset on tick 1 instead of ramping; `stop_conveyor_object()`
zeroed `ctrl` but left the velocity actuators enabled as active brakes
rather than a real detach) — merged as
[PR #20](https://github.com/VivekSai07/GAUGE/pull/20).

**The mechanism is correct. It immediately proved Round 3's number wrong.**
Running the real episode with the lift check in place: `contact_verified:
False`, `object_peak_height_gain_m: ~0.03` against the new test's 0.05
threshold. The object that Round 3 reported as grasped does not survive
being lifted. This is the check doing its job, not a regression.

**Root-causing this (not just re-tuning) went through `systematic-debugging`,
eight-plus hypotheses deep, each tested against a real run rather than
argued from theory:**

1. Enabling the already-built but previously-unused `grasp.cov_threshold`
   (covariance-gated commit) — never fires within `max_steps`; the KF's
   position covariance plateaus at its steady-state floor (~3.7e-4 trace)
   and does not shrink further no matter how long the episode waits.
2. Tightening `position_tolerance` from 0.035 down to 0.01 — `grasp_error_m`
   nearly halves (0.039 → 0.021), but `object_peak_height_gain_m` does not
   improve (stays ~0.024–0.030). Aggregate distance accuracy is not the
   controlling variable.
3. Sweeping `kf.meas_var` over 20× — no measurable effect on anything. The
   sim is fully deterministic (no randomness for the filter to average
   out), so re-weighting measurement trust doesn't change a fixed,
   geometry-driven residual.
4. Raising object friction past 5.0 — the episode stops committing to a
   grasp at all. This reproduces the exact platform-tilt instability
   already root-caused and fixed in `experiments/watch_conveyor_tracking.py`
   earlier this project (high friction + a directly-actuated free body
   against a static surface). Not a free knob.
5. Commanding a partial gripper closure sized to the object's known 4cm
   width instead of full closure (`ctrl=0`) — alone, makes it worse (the
   fingers stop before reaching a mis-centered object at all rather than
   dragging into contact).
6. An explicit MPC-target-relative closing-axis gate — drives `ee_pos` to
   within 1.7cm of *`target`*, but `target` itself carries the error, so
   this doesn't touch the real problem.
7. Waiting extra ticks after the hold→move transition before allowing
   commit (a debounce) — genuinely helps (`grasp_error_m` down to ~0.018 at
   30 ticks) but still never survives the lift.
8. **The isolating experiment that found the real, quantified root cause:**
   teleporting the object to the arm's exact TCP position at the real
   commit instant (bypassing perception/tracking entirely) shows the grip
   mechanism itself is flawless at zero error — symmetric closure, holds
   through the full lift, every time. A controlled sweep of offsets along
   each gripper-local axis from that same isolated instant found a sharp,
   reproducible cliff: **closing-axis (local Y) misalignment is tolerated
   up to ~3.0cm and fails completely at 3.67cm** — almost exactly where the
   real closed-loop system lands (measured Y-offset at commit: 3.67cm). X
   and Z offsets up to 2cm barely matter. This is not a vague accuracy
   problem; it is one specific, mechanically-grounded number (close to the
   Panda finger's own ~4cm travel limit) to beat.
9. A second, independently confirmed effect on top of (8): even after the
   gripper closes, the object drifts a further ~1.2cm along the closing
   axis *while the fingers are still closing* (~0.4s) — residual conveyor
   velocity plus the closing finger dragging a not-quite-centered object,
   not free sliding (confirmed by logging the object's world position
   throughout the settle window).
10. A redesigned commit sequence addressing both (8) and (9) directly — a
    final re-measurement phase right before closing (tracking the fresh
    raw segmentation centroid instead of the possibly lagging KF/blend
    target) plus a brief hard velocity-zero on the conveyor actuators to
    kill residual momentum fast, before switching to the full detach used
    for the lift — measurably improved accuracy (0.039 → 0.027) but did
    not clear the bar either. Pushing the re-measurement window longer
    (40–60 ticks) made it *worse* (error up to 0.13): tracking a raw,
    unfiltered single-frame measurement for many ticks without the KF's
    smoothing diverges rather than converges.

**Conclusion, not yet fixed:** every lever available at the control/timing
layer tops out right around the 2–3cm cliff found in (8), inconsistently on
either side of it. The evidence points to a sensing-resolution limit, not a
control-logic bug: 64×64 RGB-D color-threshold segmentation of a 4cm object
may not support the sub-2cm precision this gripper geometry needs,
independent of how the commit timing is arranged around it. The two most
likely real fixes, neither attempted yet: (a) improve segmentation accuracy
itself (HSV thresholding — already flagged as a fragility in
`experiments/watch_conveyor_tracking.py`'s docstring — sub-pixel centroid
refinement, or averaging multiple frames in a genuinely-stationary final
approach rather than one that keeps chasing a moving estimate), or (b) widen
the physical margin (a larger object, or a wider effective gripper reach)
so the existing ~2–3cm accuracy has room to work with. This is left as a
documented, evidenced, open limitation rather than a silently-abandoned
thread — `contact_verified`/`object_peak_height_gain_m` will continue to
report it honestly as `False` until one of these is actually done, exactly
as Round 3 established this project should behave.

### Round 4, continued: three externally-sourced leads, all ruled out

Prompted by external research (three reference repos —
`Ys-Jia/Adaptive-Grasp-in-Dynamic-Environment`,
`UT-Austin-RPL/deoxys_control`, `ARISE-Initiative/robomimic` — plus a
literature-informed summary of common MuJoCo grasp-failure fixes), three
more concrete, previously-untested hypotheses were run down via three
parallel research/experiment agents (`dispatching-parallel-agents`) and one
direct follow-up test. All three were genuinely tested against real runs,
not dismissed on theory, and all three are now ruled out:

1. **Contact softening/stiffening (`solimp`/`solref`/`condim`).** Neither
   repo actually solves this problem (`Adaptive-Grasp-in-Dynamic-Environment`
   has the same blind bang-bang close with no drift compensation;
   `robomimic` has no MuJoCo models or gripper-closing code of its own — the
   relevant contact params live in robosuite/mimicgen, out of scope). A
   direct sweep — 24 configurations across the cube geom, the finger pad
   geoms, and both together, spanning stiffer/softer `solref`, tighter/looser
   `solimp`, and `condim` 4/6 — never once flipped the real failure point
   (3.67cm) to a success, and drift-during-closing showed no reliable,
   monotonic response to softening. Stacking moderate softening across both
   surfaces *simultaneously* silently broke the previously-working 0.0cm and
   3.0cm cases — a regression a naive "soften everything to be safe" fix
   could easily introduce without noticing.
2. **Pre-impact velocity matching** (command the arm's TCP to track the
   object's estimated velocity through the close+settle window, instead of
   holding still) — motivated by "your end-effector must match the
   conveyor's velocity vector at contact." Tested with both an oracle
   (ground-truth velocity) and the real KF estimate, in both an isolated
   sweep and the full closed-loop pipeline: `contact_verified` was identical
   with matching on vs. off at every offset from 0 to 6cm, and matching the
   full 3D velocity (not just the closing axis) actively flipped a
   boundary-case success into a failure. The diagnosis: the drag during
   closing is caused by the *fingers'* own closing motion relative to the
   object as the pads converge, not by the arm base/TCP's translational
   velocity relative to the belt — matching whole-arm velocity doesn't touch
   that degree of freedom at all.
3. **Force/width-plateau grasp commit** (`deoxys_control`'s actual pattern:
   libfranka's native `grasp()` closes at bounded force and stops on
   contact/stall, rather than driving to a fixed closed position) —
   the single most promising lead, since it directly targets "one finger
   drags a not-quite-centered object." Tested two ways: (a) a kinematic
   plateau detector (freeze the commanded width once tendon length stops
   decreasing), which false-triggered on the position servo's own natural
   deceleration as it approaches any target — including in free air — and
   broke the previously-reliable 0cm case; (b) genuine MuJoCo contact-force
   detection (`mj_contactForce` on the object body, swept over six
   thresholds from 0.01N to 0.5N), which also broke the 0cm case at every
   threshold. Root cause: for an object this light (0.05kg), *any* stopping
   point at or near first contact leaves the fingers only lightly touching —
   not squeezed enough for friction to hold through a lift. Full closure
   (`ctrl=0`) turns out to be necessary, not merely a naive default; the
   real failure mode is the object being *pushed out from* an
   incompletely-centered grip during that full closure, not the closure
   itself being too aggressive.

**Updated conclusion:** every control-layer and contact-physics lever tried
— tolerance/debounce tuning, commit-sequence redesign, contact-parameter
tuning, velocity matching, and contact-triggered closure — either does
nothing or actively regresses previously-working cases. This is now strong,
convergent evidence (not a single dead end) that (a) from the "Conclusion,
not yet fixed" paragraph above — improving segmentation/targeting precision
itself, or widening the physical margin — is the only remaining path, and
that further control-strategy tuning at the current precision floor is not
a productive direction.
