# Cube Tilt (#27) + CLOSE Trigger Robustness — Design Spec

## 1. Purpose

Closes [issue #27](https://github.com/VivekSai07/GAUGE/issues/27) (cube
tips over during conveyor travel, reaching 77.03° by grasp time under the
current rendezvous timing — worse than the ~32° originally reported, since
rendezvous takes longer than the old pursuit approach) and, as a direct
consequence of verifying that fix end-to-end, also closes most of the
practical impact of [issue #36](https://github.com/VivekSai07/GAUGE/issues/36)
(camera blind during WAIT): the same closed-loop verification found and
fixed a second, previously-masked bug in the grasp CLOSE trigger. Together
they take the grasp from working at 1 tested speed (0.08 m/s,
`grasp_error_m` 0.0135) to working at all 6 previously-tested speeds
(0.04–0.12 m/s, `grasp_error_m` 0.0084–0.0119) — including two speeds
(0.04, 0.05 m/s) that failed outright under the shipped Round 6 rendezvous
design.

## 2. Root cause 1: center-of-mass actuation tips the cube

`sim/conveyor_scene.py` drives the conveyor object with two `<velocity>`
actuators on its free joint. The joint (and therefore the body frame
origin the actuators push) was placed at the cube's geometric center,
which is also its center of mass. The platform's friction reaction acts
2cm below that, at the base — so every tick of drive force created a
tipping couple. Measured directly (arm held still, object driven for
3650 steps at the shipped friction=3.0): **77.03°** of tilt.

Two approaches were evaluated:

- **Decouple contact friction** (a MuJoCo `<pair>` element giving
  cube-platform contact its own, lower friction than cube-finger contact).
  Verified to eliminate tilt in isolation (0.00–0.05° across pair-friction
  values 0.1–1.0) — but broke the grasp in the full closed-loop episode at
  *every* value tested (`contact_verified: False`, `object_height_gain_m`
  ≈ 0). Lower friction lets the cube travel measurably farther for the
  same drive force (0.453m → 0.540–0.575m over the same step count in
  isolation), shifting the object's arrival timing enough that the
  rendezvous logic — implicitly tuned against the old, friction=3.0
  dynamics — missed it. Rejected.
- **Drive from the base** (this fix): move the free joint's origin to the
  cube's base instead of its center, offsetting the geom up within the
  body frame so its world position (and mass/inertia, computed from the
  geom) is unchanged. The velocity actuators now push the body frame
  origin at the base, co-located with the friction reaction that opposes
  it, removing the moment arm at its source rather than reducing friction
  to blunt its effect. Verified: tilt 77.03° → **0.17°**, with the cube's
  steady travel velocity clean and constant (0.0559 m/s, no transient
  bump), and — critically — **no** friction value needed changing, so
  the grip strength established for finger-vs-object contact (Round 3)
  is untouched.

`ConveyorSceneEnv.get_object_ground_truth()` continues to return the
object's true center: it now reads `data.geom_xpos[geom_id]` instead of
`data.xpos[body_id]`, so the body-frame reparameterization is invisible to
every caller (perception ground-truth checks, the integration test,
`experiments/yolo_precision/generate_dataset.py`) — none of them needed to
change.

## 3. Root cause 2: the CLOSE trigger was accidentally tuned to the tipping bug

Fixing root cause 1 alone (base-drive, no other changes) still failed the
full episode: `contact_verified: False`, `grasp_error_m` 0.0443, TCP
closing **4.3cm ahead** of the object along its direction of travel — a
false-early trigger, not a small miss.

Tracing the CLOSE phase (`run_conveyor_demo.py`) found the crossing check
`np.dot(predicted - ee_pos, closing_axis)` used `tcp_rot[:, 1]` — the
gripper's closing axis — as the projection direction. At the arm's
commit-time posture that axis points mostly along world-**X** (~0.93)
while the object travels almost entirely along **Y**. The trigger was
therefore dominated by a near-constant, low-information X component and
only weakly sensitive to the Y-crossing it exists to detect — a real
design gap, not something the base-drive change introduced. Under the
old, tipping dynamics the resulting drag happened to keep the object's
Y-position in range when the (nearly X-driven) trigger fired; under the
now-correct dynamics it doesn't, and the trigger fires early.

**Fix:** project onto the object's own direction of travel
(`vel_horizontal / speed`, already computed for the rendezvous point above
this block) instead of the gripper's closing axis. This measures the
thing the CLOSE phase actually needs to know — has the object, moving
along its own path, reached the TCP — independent of whatever the wrist
happens to be oriented to. Falls back to the old closing-axis behavior
only in the degenerate case `speed <= 1e-3` (shouldn't occur in practice,
since WAIT is only entered from a CONFIRMED track with `speed > 1e-3`, but
kept as a defensive fallback rather than an unguarded division).

## 4. Verification

Full `uv run pytest -v`: **57/57 pass**, unchanged.

Full closed-loop episode, `configs/conveyor.yaml`'s speed (0.08 m/s):

| | Round 6 (shipped) | base-drive only | base-drive + travel-axis trigger |
|---|---|---|---|
| tilt at grasp time | 77.03° | 0.17° | 0.17° |
| `grasp_error_m` | 0.0135 | 0.0443 | **0.0100** |
| `contact_verified` | True | False | **True** |
| `object_height_gain_m` | 0.0893 | -0.0014 | 0.0885 |

6-speed sweep (0.04–0.12 m/s), all with both fixes applied:

| speed (m/s) | `grasp_error_m` | `contact_verified` |
|---|---|---|
| 0.04 | 0.0105 | True |
| 0.05 | 0.0102 | True |
| 0.06 | 0.0101 | True |
| 0.08 | 0.0100 | True |
| 0.10 | 0.0119 | True |
| 0.12 | 0.0084 | True |

All 6 pass, including 0.04/0.05 which failed under the shipped Round 6
design (documented there as a known 4-of-6 limitation, and the origin of
issue #36). This is a side effect of fixing the trigger's projection axis,
not a change targeted at #36 directly — the camera is still 0% detection
during WAIT, so #36 stays open as a documented limitation, but its
observed impact (grasp failures at slow speeds) is resolved by this round.

## 5. Files changed

- `sim/conveyor_scene.py`: object body/joint moved to the cube's base;
  geom offset to compensate; `get_object_ground_truth()` reads
  `geom_xpos` instead of body `xpos`.
- `run_conveyor_demo.py`: CLOSE-phase crossing check projects onto
  `vel_horizontal`-derived travel axis instead of `tcp_rot[:, 1]`.

No changes to `perception/`, `tracking/`, `control/mpc.py`, or
`tests/test_integration_conveyor.py` (the acceptance test, still
unmodified, still passing).

## 6. Documentation follow-up (post-merge)

- Close #27 with these numbers.
- Update #36 to note the reduced (but not eliminated) impact — camera
  coverage during WAIT is still 0%, but it no longer causes grasp
  failures at the previously-tested speeds; the issue stays open as a
  sensing-coverage gap, re-scoped rather than closed.
- `README.md` and `docs/PROJECT_METRICS.md`: new Round 7 entry with the
  table above.
