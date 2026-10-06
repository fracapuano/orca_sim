<p align="center">
  <img src="https://huggingface.co/datasets/fracapuano/blogs/resolve/main/orca_sim.png" alt="orca_sim header" width="600"/>
</p>


`orca_sim` provides simulation environments for the ORCA hand, including a
bimanual ORCA-Panda manipulation scene in Genesis.
You can start building your ORCA hand today at [orcahand.com](https://www.orcahand.com/).

## Install

We recommend using python 3.11 inside of a virtual environment.
You can create a virtual environment using `uv` ([how to install uv](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
uv venv orca --python 3.11
source orca/bin/activate
uv pip install orca_sim
```

Alternatively, you can use `conda` ([how to install conda](https://www.anaconda.com/docs/getting-started/miniconda/install)):
```bash
conda create -n orca python=3.11 -y
conda activate orca
python -m pip install orca_sim
```
As we are continuously iterating on `orca_sim`, you can fetch the latest `main` building this package from source, so to be in the loop with the latest developments.

```bash
git clone https://github.com/orcahand/orca_sim
cd orca_sim && uv pip install -e .
```

> [!WARNING] 
We are still iterating (a lot!) on this package. If you need stability, consider sticking to the Pypi package (`pip install orca_sim`).

## Getting started

`orca_sim` follows the [Gymnasium](https://gymnasium.farama.org/) API, and uses [Mujoco](https://mujoco.readthedocs.io/en/stable/overview.html) for physics simulation and rendering.
You can instantiate an environment with one hand (or both) via:

```python
from orca_sim import OrcaHandRight  # or OrcaHandLeft, OrcaHandCombined

env = OrcaHandRight()
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```
The 'extended' version of different hands contain additional bodies (incl. inertial properties) such as the camera mount, the U2D2 board and fans.

### Hands versioning

By default, any environment defaults to the latest fully-supported embodiment files.

```python
from orca_sim import OrcaHandRight, OrcaHandRightExtended

env = OrcaHandRight()  # latest version of the standard right hand
extended_env = OrcaHandRightExtended()  # latest version of the extended right hand
```

`orca_sim` stored different hand versions under versioned `scenes/` and `models/` directories. You can still pin an older version explicitly when needed:

```python
from orca_sim import OrcaHandCombinedExtended

env = OrcaHandCombinedExtended(version="v1")  # loads the v1 hand
```

See our [`random_policy.py`](random_policy.py) example to see how to instantiate and interface an ORCA hand.

## Panda + fixed v2 hand: native MuJoCo cube stacking

This local experiment mounts the corrected **native v2 right-hand MJCF** on
the Franka Emika Panda from `orca_arm`'s Menagerie assets. It uses the ORCA
`cube_stacking.xml` scene, including its two **5.6 cm / 30.72 g cubes** and target
marker. The composer extends its table to one continuous **1.10 × 0.80 m**
bench with a **0.325 m** surface height; the separate robot table is removed.
The user's saved base is **(0, -0.224, 0.325) m**, with **90° yaw**, facing
across the long side of the bench. The task objects are now directly ahead:
red cube **(0.438, 0.201, 0.353) m**, blue cube **(0, 0.201, 0.353) m**, and target
**(0, 0.201, 0.326) m**, underneath the blue cube. These world-frame positions
are stored in `src/orca_sim/assets/mujoco/tabletop_layout.json` and used by
both MJCF body poses and reset keyframes. Both cubes are 20% smaller in every
linear dimension than the original 7 cm cubes. Density is preserved: mass
scales by 0.8³ and inertia by 0.8⁵. Their centres are lowered by 7 mm to keep
their bottoms on the table; saved XY positions and friction are unchanged.
The single default home is the user's saved arm pose
in `src/orca_sim/assets/mujoco/home_pose.json`, including the editable hand rest
pose. All seven arm angles and all 17 saved hand angles are preserved exactly;
the rollout never replaces the saved home. No older arm home is used as a fallback.
The scene originated in the primary checkout and is resized only in this
worktree; the primary checkout and its local edits are not modified.

No Genesis runtime or Genesis model conversion is used. The assembly retains
the fixed hand's original meshes, inertias, actuators, collision defaults and
self-collision exclusions. No further hand-asset changes were needed.

```bash
python -m pip install -e '.[panda,render,dev]'
python scripts/mujoco_panda_grasp.py --video --check
python scripts/mujoco_panda_grasp.py --stack --video --check \
  --output-dir renders/mujoco_panda_front_stacking
# Interactive simulation; macOS requires mjpython:
.venv/bin/mjpython scripts/mujoco_panda_grasp.py --viewer --check
python -m pytest tests/test_mujoco_panda.py -q
```

The 20-second rollout starts at home, approaches above the red cube, descends
with an open hand, closes the fingers, lifts, and holds. The approach keeps
the thumb abducted until closure, with a 0.3 rad wrist target and a higher
grasp frame so the thumb does not push the cube sideways during descent.
These are trajectory targets, not changes to the saved resting hand pose.
Default hand-frame yaw follows the robot's base yaw minus 90°; `--yaw` can
override it in world-frame degrees. Arm IK runs on scratch
data; only actuator targets drive the live robot. The cube is a free body, with
no pose resets during the rollout, welds, mocap attachment, external holding
forces, or optional preload controller. Native hand torque limits remain ±1 Nm.
The historical 7 cm-cube front-workspace rollout holds the red cube **13.57 cm above its initial
height**, with thumb/index/middle contacts. With `--stack`, the 40-second
rollout then transfers, levels a cube face, lowers onto the blue cube,
unloads the grip, opens the fingers while retreating, and waits. During the
final three seconds the red cube is supported by the blue cube, the blue cube
by the table, and neither the hand nor the table supports the red cube.
The final centre-to-centre height is **70.00 mm**, with **5.80 mm** horizontal
offset. No negative-distance hand–cube contacts, sampled skin-vertex overlap,
arm/hand–table penetration or joint-limit violations were detected in this
run. These are measured diagnostics, not a mathematical proof of zero overlap.
No MuJoCo warnings occurred. Final-three-second peak cube speed is
**0.0026 mm/s**, with **0.000022 rad/s** peak angular speed.
This is a scripted, state-feedback stacking demo, not a learned policy or a
guarantee for arbitrary objects or teleoperation commands. The earlier
single-seed IK issue is addressed by a scratch-data approach seed, joint-limit
avoidance and deterministic restarts; these never replace the live home pose.
Placement levels a cube face while allowing its in-plane yaw to remain free,
avoiding an unnecessary 90° wrist turn just to match the world's X/Y axes.
It tests continuous IK paths and uses simulated position/orientation feedback
to compensate for compliant-grasp deflection. On release,
commands approach the measured finger angles to remove the squeezing preload
before opening and retreating together; the wrist stays steady while the
fingers open. Before withdrawal, the controller checks the complete Cartesian
retreat with scratch-data IK. If pulling backward would hit an arm limit, it
tries steeper paths up to a vertical 18 cm withdrawal; an unreachable set of
paths fails explicitly. The selected displacement is recorded as
`retreat_vector_m` in the report. This checks reachability, not a general
collision-free motion-planning guarantee. No gains, force caps, friction or
hand MJCF contact settings are changed by this controller.

To avoid the large initial turn and retain the saved home's side-facing hand
heading, use the optional sideways mode:

```bash
python scripts/mujoco_panda_grasp.py --sideways --stack --video --check \
  --output-dir renders/mujoco_panda_sideways_stacking_20261004
```

This continues the home arm's IK branch along a Cartesian translation, holding
the initial `hand_frame` rotation through approach, descent and lift. That
frame is **upstream of the ORCA wrist**: the hand's wrist still bends toward
the normal 0.3 rad grasp target and the fingers close. The mode uses a slightly
different grasp offset, `(-0.03, 0.1762, -0.10)` m in the hand frame, without
moving the cubes or changing the robot home or MJCF/contact parameters.
`--offset` can override it; `--yaw` and `--vertical` conflict with `--sideways`.

During placement this mode retains the measured grasp orientation rather than
rotating the held cube to level a face. The lowering height accounts for the
tilted cube's complete vertical extent; after release the cube settles onto
the blue cube through ordinary contacts. The same final three-second stability
checks apply. Before the cube resize, the validated October 4 side-facing run
holds a **13.59 cm** lift, with **0.73°** maximum approach hand-frame rotation
relative to home. This does not mean that the downstream wrist or fingers are
immobile, or guarantee a grasp for other layouts. The default mode remains
available unchanged when `--sideways` is omitted.

For an open-palm landing followed by a finger wrap, use the contact-gated mode:

```bash
python scripts/mujoco_panda_grasp.py --palm-first --stack --video --check \
  --output-dir renders/mujoco_panda_palm_first_small_cubes_20261004
```

`--palm-first` implies `--sideways`. The open hand approaches above the cube,
then descends until the actual native palm mesh contacts the cube's top face.
Bounded vertical admittance adjusts the arm target to rest against that face,
with a nominal 2 N palm-load target. Finger closure requires **0.6 continuous
seconds** of palm-top contact with a filtered load between 0.2 and 5 N;
without this contact the demo stops by 15.5 s and leaves the fingers open.
The ring and little fingers then spread to establish opposing contacts while
the fingers and thumb wrap over three seconds. A four-second lift and
three-second hold precede transfer, release and the usual stack checks.

This mode defaults to `--grip 1.0`, `--wrist 0.06` rad and a hand-frame offset
of `(0.01, 0.15, -0.048)` m for the 5.6 cm cubes. The vertical offset follows
the cube half-height. It keeps the saved home, base and horizontal cube layout,
hand MJCF, actuator gains, torque caps and friction unchanged. The 2 N target
is for resting the palm, **not a global force cap**: contact loads can be much
higher during closure and the loaded grasp. The report includes first palm
contact, closure start, qualifying rest duration, raw resting-load range and
peak palm-top load across the sequence. These contact signals come from the
simulator; this is a scripted feedback controller for the saved layout, not
a general grasp planner or a guarantee for arbitrary teleoperation targets.
With both cubes resized to 5.6 cm, the palm-first physics run holds an
**18.13 cm** lift and finishes with a stable **56.00 mm** centre-to-centre
stack height and **9.97 mm** lateral offset. The arm, hand and saved XY layout
are unchanged; older videos retain their original 7 cm cubes.

Outputs under `renders/mujoco_panda_grasp/` include `home.png`, `rollout.mp4`
(whole-scene view plus contact close-up), `scene.xml`, `trace.json` and
`report.json`. The generated MJCF has an `orcapanda_home` keyframe. `--check`
requires a >10 cm lift held with thumb plus finger contacts, no table support,
bounded penetration/joint-limit error, and no MuJoCo warnings. Contact depths
are checked each 1 ms; rendered-skin vertex penetration every 20 ms is only a
lower-bound diagnostic, not a full mesh intersection proof.
The new stack command writes these artifacts under `renders/mujoco_panda_front_stacking/`;
the older short-side rollout remains under `renders/mujoco_panda_stacking/`.
Its check additionally requires sustained cube–cube and blue–table contacts,
no hand support, a stack height matching the cubes' actual half-heights
(currently 56 ± 5 mm), <15 mm lateral offset, and low
linear/angular velocity throughout the last three seconds, checked every
physics step. The native Panda scene now uses a 1 ms timestep to resolve the
stiff cube contacts during settling; no source hand MJCF/contact parameters
are altered. Failed controller
runs retain a partial trace and `failure.json` for diagnosis.

For external IK/retargeting, `step` accepts **24 joint-position targets in
actuator order**, not `qpos` order (the latter also contains the free cubes):

```python
from orca_sim import OrcaPandaCubeStackingMujoco

env = OrcaPandaCubeStackingMujoco()
obs, info = env.reset()
print(env.actuator_joints)  # 7 Panda joints followed by 17 named hand joints.
targets = env.data.ctrl.copy()
targets[env.actuator_joints.index("right_i-mcp")] = 0.5
obs, reward, terminated, truncated, info = env.step(targets)
env.close()
```

The Gym ID is `OrcaPandaCubeStackingMujoco-v0` after `orca_sim.register_envs()`.
Genesis installation is not required; `panda_path` can also point directly to
the no-hand Panda MJCF and its meshes instead of installing `orca_arm`.

### Interactive base, hand-rest and cube-layout sliders

```bash
python -m orca_sim.pose_editor
# Keep the separate hand-rest editor available:
python -m orca_sim.pose_editor --mode hand
# Move cubes relative to the saved robot base:
python -m orca_sim.pose_editor --mode objects
```

Open the printed `http://127.0.0.1:.../` address. This local editor renders the
actual MuJoCo model while physics is paused. The default **layout** mode has
**base X / Y** sliders (metres) and **yaw** (degrees, counterclockwise viewed
from above). Z is fixed at the shared tabletop height, **0.325 m**; X/Y ranges
leave 12 cm clearance to the bench edges. **All seven arm and 17 hand joint
angles stay locked**, including server-side. The whole robot moves as one
assembly; the shared table and cubes stay fixed. Moving the static base
recompiles its MJCF specification so collision structures update too.
The `panda` extra requires MuJoCo 3.3+ for this model-editing API (tested here
with MuJoCo 3.11).

Choose **Whole robot**, **Top view** or **Hand close-up**; drag the image to
orbit and scroll to zoom. **Save layout** updates the default home file;
**Revert unsaved layout** restores the last saved base position and yaw.
In `--mode hand`, the base position/yaw and arm angles are instead locked.
The 17 hand-joint sliders cover the wrist, thumb and four fingers, grouped by
digit and labelled in degrees.

The layout and hand save buttons update `src/orca_sim/assets/mujoco/home_pose.json`;
existing versions are backed up alongside it as timestamped `.bak` files.
**Revert unsaved hand edits** restores the last saved hand pose, never an old
arm configuration. Restarting the editor loads this canonical file; refreshing
its page preserves unsaved preview edits and labels them as unsaved.
Robot contacts deeper than 1 mm block saving; this is a check of
the model's enabled collision geometry, not a guarantee for disabled pairs
or for subsequent motion. New environments load this default automatically;
the generated MJCF reset keyframe also contains the named hand angles in the
correct qpos and actuator order. To use an explicit alternative file:

```bash
python scripts/mujoco_panda_grasp.py --video --check \
  --start-pose path/to/pose.json
```

Or pass `start_pose="path/to/start_pose.json"` to
`OrcaPandaCubeStackingMujoco(...)`. Version-2 files include `hand_qpos`, a mapping
from each of the 17 native joint names to radians. Legacy version-1 files are
still accepted explicitly and initialize an open hand. The optional
`base_yaw` is in radians about world Z; old files without it mean zero yaw.
The generated MJCF stores the base position/orientation in the fixed root
body, so loading it or resetting a newly created environment preserves the
layout. Restart an already running simulator after saving. The rollout holds the
saved hand pose during its initial settle phase, then interpolates to its
approach preshape. Custom base poses must fit the shared bench at its surface
height; arbitrary edited home poses still need a new rollout validation.
The editor binds only to loopback;
stop it with Ctrl-C. Use `--save-path PATH` or `--port NUMBER` if needed.

In `--mode objects`, each cube has **Forward / Left** sliders and numeric
inputs in metres, relative to the saved robot base orientation. The robot's
base, arm and hand remain locked. In **Robot top view**, forward is up and
left is left; zero is the robot's mounting origin. Cube height stays on the
table, and the green stacking target follows the blue cube. Slider ranges
keep the complete object footprints on the table. Cube overlap and enabled
robot contacts deeper than 1 mm block saving.

**Save cube layout** writes only
`src/orca_sim/assets/mujoco/tabletop_layout.json` (world coordinates), with a
timestamped backup; it never changes the robot home file. **Revert unsaved
cube edits** restores the last saved positions. New environments and demo
runs load this layout automatically, including their reset keyframes;
restart an existing simulator to load a newly saved layout. You can also pass
`object_layout="path/to/layout.json"` to `OrcaPandaCubeStackingMujoco(...)`.
This paused editor checks placement, not IK reachability or motion safety:
an edited layout still needs a fresh rollout validation before collecting
demonstrations.

## Standalone v2 hand grasp validation (native MuJoCo)

This isolates the **actual v2 hand MJCF** from the arms and Genesis. A fixed
forearm closes around a free 5 cm cube; a small support retracts 25 cm, leaving
the hand to hold the cube under gravity for five seconds. The support collides
with the cube, not the fingers. There is no cube weld, pose reset after closure,
external cube force, enlarged collision box, IK, or arm motion. Native actuator
gains, torque limits, masses/inertias and friction coefficients are retained.
The v2 MJCF now supplies firmer hand contacts and elliptic friction, as described
below; this is a native-MuJoCo asset test, not a new Genesis validation.

From this checkout (Python 3.11; validated with MuJoCo 3.11.0):

```bash
python -m pip install -e '.[render,dev]'
python scripts/standalone_hand_grasp.py --suite --video \
  --output-dir renders/standalone_hand/validated
python -m pytest tests/test_standalone_hand_grasp.py -q
```

Genesis and `orca_arm` are **not required** for this experiment. Omit `--video`
to run without an OpenGL context. The suite writes `report.json`, per-trial
metrics and traces, scene XMLs, and (when requested) videos:

- `right_comparison.mp4` / `left_comparison.mp4`: historical skin masks **and
  soft-contact settings** on the left, corrected native MJCF on the right,
  with identical commands.
- `right_open_control.mp4` / `left_open_control.mp4`: without closing, the cube
  falls when the support retracts.

To inspect the contact region from all sides after generating the suite:

```bash
python scripts/render_standalone_grasp_orbit.py
python scripts/render_standalone_grasp_orbit.py --diagnostic
```

These replay both saved scenes to the recorded 5.002 s sample, verify the cube
pose and penetration against the saved trace, then **pause physics** while the
camera orbits. The close-up videos and eight still viewpoints are saved under
`validated/orbit` and `validated/orbit_diagnostic`. The diagnostic view makes
only the rendered cube transparent; amber markers identify sampled skin vertices
more than 0.1 mm inside the cube. They are geometric-overlap markers, not contact
force markers. Transparency also reveals fingers *behind* the cube, which must
not be mistaken for penetration. No physical settings or poses are changed.

The original assets **can hold the cube**, but their ten white finger-skin geoms
per hand had `contype="0" conaffinity="0"`. Internal structures supplied contact
while the cube entered the visible skin by approximately **5.2 mm**. Both v2
hand-body XMLs now enable collisions on these existing skin meshes. The initial
mask-only fix was insufficient under heavier squeeze. The current fix also
sets hand-local `solref="0.004 1"`, `solimp="0.999 0.9999 0.001"`, `priority="1"`
and uses `cone="elliptic" impratio="10"`. This reduces soft-contact penetration
and regularization slip; it does not replace meshes, inflate the cube, raise
friction coefficients, or change masses, gains, motor limits or the 2 ms timestep.
Keep the timestep at or below 2 ms. The hand defaults live in side-specific
classes, inherited through the body root's `childclass`. Include each hand's
`.mjcf` definition along with its `_body.xml` (do not ship the body alone).
The priority makes hand contact parameters win over lower-priority objects;
custom material/contact pairs may need their own explicit settings.
These are numerical rigid-contact defaults, not a measured silicone-compliance
calibration. The existing 0.5 mm contact margin is unchanged.
See MuJoCo's [contact and slip guidance](https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip).
The Genesis importer continues to separate visual skins and collision copies;
it uses its existing contact configuration. Its generated model remains
byte-for-byte unchanged by this native-MuJoCo correction.

The corrected left and right hands pass the nominal 80 g trial, 40 g and 120 g
trials, and four selected collision-free 3 mm position perturbations each.
Open-hand controls drop the cube. The original-mask trials retain the cube but
fail the visual-penetration check. This is a **bounded smoke test**, not a
guarantee for arbitrary objects, initial poses, or controller commands.

### Stronger squeezing: contacts versus commands

**For a model-only test, use the progressive comparison below.** It does not
use the optional `GraspPreload` controller. That controller changes the command
and therefore must not be used as evidence that the MJCF itself tolerates the
same increasing grip demand.

Teleoperation acceptance concerns **collision coverage and contact stability**
under unmodified native joint commands. Do not treat lower gains, lower effort
limits, preload balancing, grasp welds, or artificial holding forces as contact
fixes. The standalone stress command now defaults to progressive native control;
`--check` checks retention/penetration/finite state, while `--check-motion`
separately asks whether the object stays nearly immobile. These are different
requirements. Thumb, fingers and palm loads are reported separately so the test
cannot quietly claim palm coverage from a fingertip-only grasp.

```bash
python scripts/progressive_grip_comparison.py --video
python scripts/standalone_squeeze_stress.py --video --check
python -m pytest tests/test_progressive_grip.py -q
```

Both hands receive exactly the same open-loop commands in three contact cases:
the original released masks/soft contacts, the previous mask-only correction,
and the corrected native MJCF. Compiled masses, geometry, friction coefficients,
servo gains and limits are asserted equal. A SHA-256 of **every physics-step
command vector** must match between the three versions, not just their final
targets. Nothing observes slip and reduces the command. The schedule is:

- 0–5 s: acquire the same nominal grasp and withdraw the support.
- 5–15 s: continuously increase MCP/PIP targets to the previous strong pose.
- 15–25 s: continue increasing all ten MCP/PIP targets to their native upper
  limits; wrist, thumb opposition and finger abduction stay fixed.
- 25–40 s: hold those maximum targets with the unchanged native +/-1 Nm motors.

The output directory `renders/standalone_hand/progressive_grip` contains both
hands' before/after videos, measured force/penetration plots, state and command
traces, checkpoints and a report. Force means summed normal contact load, not
net cube force or per-finger force. The native position servos saturate; larger
targets must not be described as unbounded or necessarily monotonic force.
There is no artificial immobility requirement: distinguish a repositioned cube
from penetration or complete loss of the grasp.

In the 40 s progressive run (80 g cube), both historical configurations lose
all force-bearing contacts around 17.2–17.8 s. Original released skin meshes
penetrate by about 6 mm; the mask-only fix still permits about 3.4 mm. The
corrected MJCF retains the cube in both hands with no measured skin-vertex or
raw-contact penetration. Right-hand load is approximately 26 / 62 / 135 / 235 /
280 N at 5 / 10 / 15 / 20 / 25 s; it remains about 279 N at 40 s. The left hand
holds about 305 N at 40 s. Peak sampled loads are 355 N right and 383 N left.
These are summed normal contact loads, not a hardware force rating.

The cube repositions up to 8.6 mm / 10.4 degrees in the right hand and 10.0 mm /
18.5 degrees in the left. Contacts briefly rearrange early in closure, so this
is **retention under increasing demand**, not proof of an immobile object or
uninterrupted thumb opposition. No native gain, torque limit, contact setting,
or geometry was changed to obtain this new comparison; only the common command
schedule was extended. The 1.37–1.39x optional preload limit is not used here.

The previous mask-only model dropped the cube under heavier MCP/PIP closure:
approximately 25 mm translation / 12 degrees rotation by 7 s, up to 3.7 mm skin
penetration, then a drop near 12 s. These measurements and videos remain in
`renders/standalone_hand/squeeze_stress`. They are not the corrected result.

The **contact fix** and a separate, optional control experiment must not be
conflated:

1. **Native MJCF contacts.** With the *identical* larger-curl command, both hands
   now retain the cube with no sampled skin/contact penetration. The cube still
   moves about 7.3–7.5 mm: changing finger posture can physically move an object.
2. **Optional controller experiment, not a contact fix or teleoperation requirement.** `orca_sim.mujoco_grasp.GraspPreload`
   latches the settled hand's joint positions and position-servo preloads, then
   scales those preloads together. It includes thumb opposition/abduction, not
   just finger flexion. It caps the common multiplier at the first torque or
   joint-target limit, preserving 5% torque headroom instead of independently
   saturating motors and changing the grasp's torque pattern. It does not read
   or control the cube pose, add constraints, change gains, or apply cube forces.

At 80 g, a request for 4x preload is **explicitly capped to 1.37–1.39x**, because
the thumb is already near its native 1 Nm limit. Actual summed normal contact
load rises from approximately 26 N to 36 N. Over the 18.5 s rollout the right
hand moves 0.23 mm / 0.34 degrees, the left 0.26 mm / 0.32 degrees. Both 60 s
runs stay below 1 mm / 1.3 degrees. All selected 40/80/120 g and collision-free
3 mm offset trials pass. This is not evidence for an uncapped 4x-force grasp,
arbitrary objects, or zero motion forever.

Reproduce the historical optional-controller experiment and its causal controls:

```bash
python scripts/standalone_squeeze_stress.py --profile preload --video --check-motion
python scripts/standalone_squeeze_stress.py --validation-suite --video
# This checkout's final saved run (same suite, separate evidence directory):
python scripts/standalone_squeeze_stress.py --validation-suite --video \
  --output-dir renders/standalone_hand/squeeze_verified
python -m pytest tests/test_standalone_squeeze_stress.py -q
# Historical failure, using enabled skins but the former soft contacts:
python scripts/standalone_squeeze_stress.py --profile flexion \
  --contact-model legacy --output-dir renders/standalone_hand/legacy_repro --check
```

The historical command intentionally exits nonzero. The validation suite writes
26 trials, hashes, state/command traces, and `before_vs_fixed.mp4` and
`fixed_contacts_raw_vs_preload.mp4` to `renders/standalone_hand/squeeze_fixed`.
It tests contact-only, controller-only, combined, longer holds and perturbations;
negative controls must reproduce their failure to pass the suite. The original
2 mm / 5 degree **peak** palm-relative-motion limits remain unchanged. The
optional-controller tests apply the immobility criterion only to the explicit
preload command. Separate model-only tests require retention and no deep
penetration, not immobility under a command that moves the fingers.
Raw contact depth, motor torque and finite state are checked at **every 2 ms
physics step**; the mesh-vertex probe and pose traces remain sampled at 20 ms.

Use after acquiring and settling an opposing-contact grasp in native MuJoCo:

```python
from orca_sim.mujoco_grasp import GraspPreload

squeeze = GraspPreload(model, data, "right")  # latch once, not every step
# Ramp 1 -> 4 over time; this example is the final requested multiplier.
data.ctrl[:] = squeeze.command(4.0, other_targets=data.ctrl)
mujoco.mj_step(model, data)
print(squeeze.applied_multiplier)  # actual common multiplier, after limiting
```

Release/open using ordinary joint targets and discard/re-latch the controller
for a new grasp. This is a static squeeze helper, not a grasp planner, slip
feedback controller, Genesis controller, or compensation for arm acceleration.

Checks require thumb plus another digit in contact, no support contact, less
than 5 mm drift from 4.0 s onward (after a 0.5 s load-transfer settling period),
finite state, and no MuJoCo warnings. Penetration limits are 1 mm for skin
vertices and 2 mm for raw solver contacts. Nominal corrected trials have no
sampled skin vertices inside the cube; small compliant-contact overlap remains.
The independent skin-vertex probe samples every 20 ms and is a **lower bound**,
not a complete triangle-intersection test or proof of zero penetration.

Single-trial reproduction:

```bash
python scripts/standalone_hand_grasp.py --side right --video --check
python scripts/standalone_hand_grasp.py --side left --cube-mass 0.12 --check
python scripts/standalone_hand_grasp.py --variant skin_disabled --contact-model legacy --video
python scripts/standalone_hand_grasp.py --variant open_control --video
```

`skin_disabled` restores only the original ten skin masks **in memory**; source
files are never modified by the experiment. Add `--contact-model legacy` to
also restore the earlier soft-contact/cone configuration, as the comparison
suite does. `--body-file /path/to/original.xml --contact-model legacy` can
include an archived original body verbatim. Reports record its SHA-256.
`--check` exits nonzero on a failed grasp smoke test; `--suite` exits
nonzero if any positive trial fails or either negative control fails to expose
the expected defect/drop. Intersecting initial fixtures are rejected before
stepping. Saved scene XMLs use absolute include paths for local inspection.

## Genesis bimanual ORCA-Panda cube grasp

The Genesis environment composes two 7-DOF Franka arms from the no-hand Panda
MJCF bundled by `orca_arm` with the native v2 left/right ORCA-hand MJCFs in
[`src/orca_sim/models/v2/mjcf`](src/orca_sim/models/v2/mjcf). It does not load
the URDF-derived hand bodies from `bimanual_orcapanda.xml`. The environment adds
a grounded workbench asset and a dynamic 5 cm, 80 g cube, exposes all 48
actuated joints through the Gymnasium API, and includes a deterministic
side-grasp/lift validation. The implementation follows the [Genesis user
guide](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/index.html)
and currently pins Genesis 1.3.3 for repeatability.

The robot, workbench, and cube are all imported through Genesis's MJCF loader;
no URDF visual model is used. The renderer selects the detailed MJCF visual
meshes, smooth normals, authored materials, a neutral studio floor, and a
three-quarter 1280 × 720 camera with balanced ambient and directional light.

Use Python 3.11 and install the optional simulator dependencies:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[genesis,dev]'
```

Run the end-to-end grasp validation headlessly and save an image at each phase
(ready, open, grasp, and lift):

```bash
python scripts/genesis_grasp_cube.py \
  --render-mode rgb_array \
  --output-dir renders/genesis
```

Add `--video` to record the continuously stepped rollout rather than only the
four checkpoints:

```bash
python scripts/genesis_grasp_cube.py \
  --render-mode rgb_array \
  --output-dir renders/genesis \
  --video renders/genesis_grasp.mp4
```

The command exits non-zero unless the cube is finite, rises at least 12 cm,
remains in an opposing pinch with at least two hand contacts, has low final
linear speed, never enters the rendered cube, and every lift inverse-kinematics
solve is within tolerance. Use `--render-mode human` for the interactive viewer
or `--render-mode none` for the fastest validation. The first run is slower
because Genesis imports the meshes and compiles its simulation kernels.

The grasp is solved relative to the object rather than hard-coded to one world
pose. For example, this translates both the cube and grasp target by 1 cm:

```bash
python scripts/genesis_grasp_cube.py \
  --render-mode rgb_array \
  --cube-offset 0.01 -0.01 \
  --video renders/genesis_grasp_offset.mp4
```

The environment can also be used directly:

```python
import gymnasium as gym
from orca_sim import register_envs

register_envs()
env = gym.make("BimanualOrcaPandaCubeGenesis-v0")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

Actions are absolute joint-position targets in asset order. Observations are the
48 robot positions, 48 robot velocities, 7-value cube pose, and 6-value cube
velocity (109 values total). A normal reset places both arms in a mirrored,
elbow-up ready configuration with open hands facing the shared workspace. This
keeps the wrists away from singular and joint-limit regions and gives IK and
random policies a well-defined starting posture. The scripted validation first
approaches its calibrated right-hand pre-grasp from this same reset pose.
The cached Genesis conversion swaps the source arm mounts: the left robot
occupies the negative-y mount and the right robot the positive-y mount, so both
hands point inward without modifying the upstream asset.

### Visualize a random policy

Use the smooth bounded random-policy example for a live rollout:

```bash
python scripts/genesis_random_policy.py --render-mode human --seed 0
```

Or record the same policy headlessly:

```bash
python scripts/genesis_random_policy.py \
  --render-mode rgb_array \
  --duration 20 \
  --output renders/genesis_random_policy.mp4
```

The policy draws random arm and hand targets around the reset configuration and
low-pass filters them before calling `env.step`. Sampling a completely unrelated
48-joint target from `env.action_space` on every frame is possible, but is not a
useful visualization: it commands discontinuous full-range motion against a
high-gain position controller.

### Contact and inertial treatment

The Panda inertias in the Menagerie MJCF are preserved. The native v2 hand MJCF
does not prescribe link inertias, so Genesis integrates them from the native
collision meshes at 650 kg/m³. The white skin meshes remain the rendered
geometry and receive transparent, coincident collision copies so a cube cannot
pass through the visible fingertip before reaching an internal bone mesh. At
load time, `orca_sim` composes this model in a cache and resolves every mesh to
an absolute path; loading does not modify source files. The native source skin
collision masks were corrected separately as described in the standalone test
above; this does not change the generated Genesis model.

The workbench is one packaged MJCF asset with a 0.75 × 1.15 m wooden top, four
grounded steel legs, aprons, lower rails, and feet. The top has coincident
visual and collision geoms so Genesis retains its authored wood material rather
than drawing the generic collision surface. Its frame is visual-only so it
cannot introduce irrelevant robot–furniture contacts; the cube-contact surface
remains at the same calibrated height as the validated grasp.

The cube has coincident 5 cm visual and support geometry plus a 0.5 mm numerical
contact skin for the ORCA links. This replaces the former 10 mm proxy. The skin's
bottom is aligned exactly with the rendered cube, so it cannot make the object
float on the workbench. The contact solver is configured to keep all raw overlap
inside that submillimetre skin, and validation separately requires zero overlap
into the rendered cube.

The closure uses the hand's physically opposing thumb configuration, stops when
the thumb and a finger form a real pinch, and adds only a bounded preload under
the hand's 0.5 N·m effort limit. It retains that contact target throughout the
lift; it never replaces it with a fully closed target. Collision meshes use a
256-face decimation target where Genesis can simplify them safely, while the
detailed MJCF visual meshes remain untouched. This reduces runtime collision
work without changing the rendered hand.

The validated rigid-contact configuration uses a 2.5 ms control step with five
physics substeps, 150 constraint iterations, five no-slip iterations, box-box
contact detection, and torsional friction. The scripted trajectory closes the
right hand gradually and lifts through five Cartesian wrist waypoints rather
than teleporting bodies.

### Isolate the added hand-skin contacts

Run a controlled A/B experiment and record a synchronized close-up video:

```bash
python scripts/genesis_skin_collision_ablation.py
```

Outputs are in `renders/skin_collision_ablation`: `comparison.mp4` (OFF on the
left, ON on the right), the individual videos, `report.json`, and saved action
and state traces. The script first records the successful controller's targets,
then replays exactly those targets in both trials. It uses the same built scene
and disables only collision pairs involving the active right hand's added skin
geoms. The idle left robot is unchanged (one of its skin meshes is merged with
other geometry by the loader and cannot be independently switched). Visuals,
mass/inertia, original colliders, friction, cube contact skin, timestep, and
solver settings stay identical. Runtime checks verify the initial states and
inertias match, that disabled skin contacts disappear, and that the ON replay
reproduces the reference result. This diagnostic uses Genesis 1.3.3 internals
and must be revalidated before changing the simulator version.

This measures the added skin contacts' contribution within the current setup;
it does not compare raw original assets or Genesis against MuJoCo. The command
stream is generated by the skin-enabled grasp, so results describe that fixed
trajectory, not the best achievable grasp for each variant. Solver contact
depths must not be interpreted as an independent visual-intersection measure.

After generating the comparison, inspect either replay in the live viewer:

```bash
python scripts/genesis_skin_collision_ablation.py --view off
python scripts/genesis_skin_collision_ablation.py --view on
```

The viewer holds the final command until closed. Left-drag orbits, Option-drag
pans, scrolling zooms, and Z resets the camera. `--no-video` runs the experiment
without rendering.

## Sample task: in-hand cube orientation

`orca_sim` now also ships a task-level example that augments the right hand with a free-floating cube whose one target face is colored red:

```python
from orca_sim import OrcaHandRightCubeOrientation

env = OrcaHandRightCubeOrientation(version="v2", render_mode="human")
obs, info = env.reset(seed=0)
```

By default, the task resets to a palm-up open-hand pose with the cube resting on the palm and the red face pointing downward. You can also randomize the initial cube orientation while keeping it unsolved:

```python
env = OrcaHandRightCubeOrientation(
    version="v2",
    initial_red_face="random",
    cube_pos_xy_jitter=0.01,
)
obs, info = env.reset(seed=0)
```

If you want to keep the environment completely deterministic while you are still building it, you can use the nominal reset directly and add randomization later:

```python
env = OrcaHandRightCubeOrientation(version="v2")

nominal = env.nominal_reset_options()
obs, info = env.reset(options=nominal)

randomized = env.sample_randomized_reset_options(
    seed=0,
    initial_red_face="random",
    cube_pos_xy_jitter=0.01,
)
obs, info = env.reset(options=randomized)
```

The implementation is intentionally split so it doubles as a porting template:

- The task scene lives in [`src/orca_sim/scenes/v2/scene_right_cube_orientation.xml`](src/orca_sim/scenes/v2/scene_right_cube_orientation.xml) and composes the existing hand MJCF with a single task cube.
- The nominal palm-up hand pose and in-palm cube spawn are now authored into the task-specific scene/model files, so opening the XML directly in MuJoCo shows the intended setup.
- The task logic lives in [`src/orca_sim/task_envs.py`](src/orca_sim/task_envs.py), including reset-time cube randomization and optional hand-pose overrides for custom MJCF layouts.
