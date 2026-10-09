<p align="center">
  <img src="https://huggingface.co/datasets/fracapuano/blogs/resolve/main/orca_sim.png" alt="orca_sim header" width="600"/>
</p>


`orca_sim` provides simulation environments for the ORCA hand.
This is Francesco Capuano's maintained fork, with native MuJoCo hand-contact
fixes and a Panda palm-first cube-stacking task. It is not a Genesis port.
The original ORCA hand project is at [orcahand.com](https://www.orcahand.com/).

## Install

Use Python 3.11 in a fresh virtual environment. During review, the full stack
lives on `fracapuano/mujoco-06-fork-packaging`; `main` has not been changed.
Install this Git checkout, **not** the upstream PyPI release:

```bash
git clone --branch fracapuano/mujoco-06-fork-packaging https://github.com/fracapuano/orca_sim.git
cd orca_sim
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[panda,render,dev]' -c requirements/mujoco-validation.txt
```

Or, after cloning, use standard Python/pip:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[panda,render,dev]' -c requirements/mujoco-validation.txt
```
The import/package name remains `orca_sim`; do not install the upstream and
fork distributions into the same environment. The local version identifier
`0.1.0+fracapuano.2` distinguishes this cleaned review stack from the earlier
mixed experimental branch. No release has been published to PyPI.

Extras: `panda` installs the pinned bare Panda MJCF/meshes from `orca_arm`;
`render` adds video/image dependencies; `dev` adds pytest. For hand-only use,
install `-e '.[dev]'` with the same constraints. No Genesis, Torch, or
`orca_core` dependency is needed. Interactive rendering requires a desktop
graphics context; headless physics does not.

The validation constraints record the tested Python dependencies rather than
claiming identical contacts on every supported MuJoCo version. The Panda
asset dependency is fixed at `d92e47106d282edde6230170aa6e70429a5dccea`.
Fork publishing is disabled by a repository-identity guard in the inherited
PyPI workflow. Merging these review branches does not publish a package.

## Review stack and ownership

Each branch below adds one commit on the preceding branch. Review the PRs
bottom-up in this order; the final branch contains the complete native task.

| Layer | Branch after `fracapuano/` | What belongs here |
| --- | --- | --- |
| 1 | `mujoco-01-hand-contacts` | Native left/right v2 MJCF collision masks, contact/solver response, isolated squeeze regression; **no limit/gain changes**. |
| 2 | `mujoco-02-panda-home` | Native MJCF composition, Gym action interface, canonical arm/hand/base home, shared bench and baseline task fixture. |
| 3 | `mujoco-03-cube-layout` | Saved cube/target positions, 20% smaller cubes and density-preserving mass/inertia. |
| 4 | `mujoco-04-palm-first-stacking` | Palm-contact gate, wrap/lift/stack controller, installed rollout command and dynamic regressions. |
| 5 | `mujoco-05-scene-editors` | Local base/hand/cube sliders, collision-aware saves and reload tests. |
| 6 | `mujoco-06-fork-packaging` | Fork install/review guide, validation constraints, version and publication safeguard. |

The first PR targets this fork's `main`; each later PR targets its immediate
predecessor, keeping its diff focused. Merge bottom-up using merge commits
to preserve ancestry, then retarget the next PR to `main`. Squashing or
rebasing a lower layer requires rebasing its descendants before continuing.
Do not delete a base branch while another open PR still targets it.

All changes in this stack belong in **`orca_sim`**. `orca_arm` supplies
unchanged, pinned Panda assets; its preassembled OrcaPanda MJCF/URDF is not
used by this native task. `orca_core` is not modified or required. Genesis
environments, conversion code, abandoned force adapters and old experiment
scripts are deliberately excluded. The previous `fracapuano/genesis-sim`
branch remains an unchanged archive, not the base of these PRs.

### Validate or reproduce the demonstration

```bash
python -m pytest -q
orca-sim-stack --palm-first --stack --check --output-dir renders/validation
orca-sim-stack --palm-first --stack --video --check --output-dir renders/video
```

Tests include both native hands, the saved scene/home, editor round trips
and full dynamic rollouts. The isolated contact test compares identical
progressively increasing closure commands with legacy versus corrected
contact response, keeping skins enabled in both to isolate solver stiffness.
A separate test verifies the previously disabled skin masks. It is not a
claim that every object and arbitrary teleoperation trajectory is stable.

The saved home and cube layout are version-controlled JSON, not browser
state. Meshes, MJCF includes, scene XML, JSON defaults, editor HTML and the
rollout implementation are package data/code in the wheel. For a non-editable
installation use an explicit editor `--save-path`, then pass that saved file
to the environment. Do not depend on writable `site-packages`.

## Getting started

### Local bimanual MuJoCo setup

The local `fracapuano/mujoco-bimanual` branch adds
`OrcaBimanualPandaCubeStackingMujoco`: two native Panda arms, the corrected
native right **and left** ORCA hands, and the same 5.6 cm cubes. It does not
convert URDFs or use the older preassembled bimanual model.

The right robot's home and the cube positions remain exactly as saved in the
single-arm task. The second base is at `(1.0, -0.224, 0.325)` m; both bases
have 90-degree yaw. A mirrored left-arm posture points its hand inward.
The native hands are **swapped between those unchanged arms**: the arm
called `right` (base x=0) carries the native **left** hand, and the arm called
`left` (base x=1) carries the native **right** hand. This puts the thumbs on
the requested opposite side without moving/reorienting either Panda or
changing the mounting transforms. Native hand joint/body/mesh names retain
their true handedness. `env.robots[side].hand_side` exposes that assignment.
The shared bench is widened to 1.80 × 0.80 m, at the unchanged 0.325 m height.
Its separate `assets/mujoco/bimanual_home_pose.json` records both homes;
the single-arm JSON and scene remain untouched. The 1 m base separation
provides clearance for the full fingertip meshes, not just joint centres.

From this checkout with the `panda,render` extras installed:

```bash
orca-sim-bimanual --video --check
# On macOS: orbit/zoom the live scene; close the viewer to stop.
mjpython scripts/mujoco_bimanual.py --viewer --hold
# Exercise both arms and hands in the viewer:
mjpython scripts/mujoco_bimanual.py --viewer --check
```

The 12-second motion demo steps MuJoCo while moving each arm through an
independent Cartesian IK path, closing/opening both hands, and returning
home. This validates the bimanual assembly/control interface, **not a
coordinated two-hand grasp or stacking policy**. The existing single-arm
stacking command still selects its original environment. The old layout
sliders also still edit only the single-arm task.

```python
from orca_sim import OrcaBimanualPandaCubeStackingMujoco

env = OrcaBimanualPandaCubeStackingMujoco()
env.reset()
# Action order: right Panda 7 + LEFT hand 17, left Panda 7 + RIGHT hand 17.
action = env.data.ctrl.copy()
left = env.robots["left"]
target = env.data.site(left.hand_site).xpos + [0, 0.03, 0.02]
action[left.actuator_indices[:7]] = env.solve_ik("left", target)
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

`env.robots[side]` exposes actuator indices, joint names, arm/hand qpos
addresses, and the hand-frame site. `env.side_action(side, targets)` replaces
one robot's 24 targets while preserving the other's command. IK operates
only on scratch data; it is not a collision-aware trajectory planner.
Here `side` and the existing `right_hand_frame`/`left_hand_frame` site names
identify the **arm**, not the native hand's handedness. The version-1 home
JSON keeps its arm-scoped hand-target keys; the builder maps their joint
suffixes to the mounted hand without changing the saved rest angles.
Collision ownership follows the kinematic tree, not hand-name prefixes.
Inter-robot collisions remain enabled, intersecting initial layouts are
rejected, and no gains, torque limits, contact masks or hand meshes are
weakened to make the motion pass. Custom homes/layouts can be supplied via
`start_pose=` and `object_layout=` (a dict or JSON path).

#### Three-camera observations

The bimanual MJCF contains exactly three named RGB cameras:

- `front`: the shared elevated view **from behind both robots**, toward the table.
- `right_wrist`: under the right arm's palm (currently the native **left** hand).
- `left_wrist`: under the left arm's palm (currently the native **right** hand).

Both wrist cameras attach to the native carpal body, so they follow the arm
**and** the hand's wrist joint. Their optical centres sit under the palm heel,
behind the grasp contact patch, looking toward the fingers and 30 degrees
down toward the work surface (80-degree vertical field of view). These are
ideal virtual cameras, not additional physical camera housings: they add no
mass, collision geometry or change to the hand's contacts. Objects/fingers
can naturally occlude the view during a grasp; no meshes are hidden.

```python
from orca_sim import OrcaBimanualPandaCubeStackingMujoco

env = OrcaBimanualPandaCubeStackingMujoco()
try:
    env.reset()
    env.step(env.data.ctrl.copy())  # Replace with your 48 IK/retargeter targets.
    rgb = env.render_cameras()     # Same physics instant; does not step.
    # rgb["front"], rgb["right_wrist"], rgb["left_wrist"]: uint8 (480, 640, 3)
    wrist = env.render_camera("right_wrist", width=1280, height=720)
finally:
    env.close()
```

Rendering needs working OpenGL. `render_cameras()` works independently of
`render_mode`; the default state-vector observation/action interface is
unchanged. The three independent image arrays share one reused offscreen
renderer. Render sizes may be up to 1280×720. Do not call `step()` concurrently
with capture. The existing `render()` free-view camera remains available for
orbiting/debugging; it is not a fourth model camera. In the MuJoCo viewer,
select `front`, `right_wrist` or `left_wrist` from the camera selector.

```bash
# Save three raw home PNGs and a labelled side-by-side preview:
python -m orca_sim.demos.bimanual_cameras
# Step the existing 12-second arm/hand motion and record synchronized views:
python -m orca_sim.demos.bimanual_cameras --video
```

Outputs are in `renders/bimanual_cameras/`: `cameras.png`, the three named PNGs,
the exported MJCF, `report.json`, and (with `--video`) `cameras.mp4`.
This camera demo preserves the saved cube layout and is **not** a grasp demo.

#### Coordinated red-on-blue stacking demo

```bash
python -m orca_sim.demos.bimanual_stack --video --check
# After reinstalling the package, the equivalent entry point is:
orca-sim-bimanual-stack --video --check
```

This is one continuously stepped MuJoCo episode: the **right arm picks red
while the left arm picks blue**, starting their approaches together. Each
hand independently waits for palm contact before wrapping its fingers and
lifting. Once both pickups are verified, the left places blue on the **X**
and clears the workspace, then the right stacks red **on blue** and releases
it. Center placement remains sequential to keep the arms clear of each other.
The roughly 69-second demo ends with five seconds of stable stacking with
both hands released. The report includes each cube's initial lift time.

The demo explicitly loads `assets/mujoco/bimanual_stack_layout.json`: red at
`(0.30, 0.201)`, blue at `(0.70, 0.201)`, and the X at their midpoint
`(0.50, 0.201)` metres. The X is a non-colliding visual marker on the tabletop.
This opt-in layout does **not** overwrite the saved homes, the default cube
layout, or the single-arm stacking scene. Arm identities still follow their
unchanged bases, not the native handedness of the swapped hand assemblies.

Only the existing 48 joint-position actuators are driven. Cubes remain free
dynamic bodies: no welds, object-state writes, external forces, contact-mask
changes or gain/torque-limit changes. Palm contact gates finger closure;
measured cube pose guides transfer/placement. Shared center access is
serialized. The stacking loop does not run the extra collision audit, hand-skin
mesh scans, or joint-limit/torque diagnostic tracking; finger self-contact no
longer triggers an audit abort. MuJoCo collision physics and the hand/cube
contact feedback used by the controller are unchanged. This is a scripted
controller for this layout, not a general-purpose collision-avoidance planner.

`renders/bimanual_stack/` contains `rollout.mp4` (with `--video`), exported
`scene.xml`, `reset.json`, `report.json`, and state/control `trace.json`.
The command starts from home and no longer writes diagnostic stage checkpoints.
`--check` exits unsuccessfully unless the released stack passes the physical
support, position and velocity checks, with neither robot supporting a cube.
Reports declare `validation_scope: "task_outcome_only"`: success is not a
collision-free or mesh-penetration certification, and removed diagnostics are
omitted rather than reported as zero. The video uses the closer
`front` camera with orbit-view close-ups of both pickups; both wrist cameras
are also present in the exported model.

#### Shared cube-position distribution

Cube sampling belongs to the **environment's `reset()`**, not the scripted
controller. Both data collection and policy evaluation must construct the
same environment configuration. Randomization is opt-in; omitting
`cube_spawn_regions` preserves the existing fixed layout.

The editable starting preset is
`src/orca_sim/assets/mujoco/bimanual_cube_spawn_regions.json`:

```json
{
  "version": 1,
  "red_cube": {"x": [0.25, 0.35], "y": [0.151, 0.251]},
  "blue_cube": {"x": [0.65, 0.75], "y": [0.151, 0.251]},
  "min_gap_m": 0.02,
  "max_attempts": 200
}
```

Bounds are cube **centre coordinates in world/table XY, in metres**: these
10 × 10 cm rectangles extend 5 cm each way from the demonstrated pickup locations.
Increasing X goes from the right-arm base toward the left-arm base;
increasing Y goes forward from the bases into the task workspace.
Copy this JSON into your experiment configuration and change the bounds to
choose other regions; a dictionary is also accepted. Equal lower/upper bounds
fix an axis. Height remains 0.353 m, orientation remains upright with zero
yaw, and the robot homes and stack target remain unchanged.

```python
from orca_sim.mujoco_bimanual import (
    OrcaBimanualPandaCubeStackingMujoco, STACK_LAYOUT, CUBE_SPAWN_REGIONS,
)

# Use these SAME kwargs for expert collection and autonomous rollouts.
env = OrcaBimanualPandaCubeStackingMujoco(
    object_layout=STACK_LAYOUT,
    cube_spawn_regions=CUBE_SPAWN_REGIONS,  # or your saved JSON path/dict
    target_marker="x",
    frame_skip=1,
)
obs, info = env.reset(seed=42)
# Run your policy, or construct a NEW BimanualStack(env) after this reset.
# env.reset() advances the RNG; env.reset(seed=42) repeats this initial state.
```

The stacking entry point delegates to exactly that reset sampler:

```bash
python -m orca_sim.demos.bimanual_stack --seed 42 --check \
  --cube-spawn-regions src/orca_sim/assets/mujoco/bimanual_cube_spawn_regions.json \
  --output-dir renders/randomized_stack_seed42
```

To plot the regions on the actual table, without stepping or changing physics:

```bash
python -m orca_sim.demos.spawn_regions
```

This saves `renders/cube_spawn_regions/distributions.png`: a scene overview
and top-down detail. Filled rectangles show the uniform cube-centre regions;
grey outlines show one cube footprint. Cubes are hidden only in these debug
renders so they cannot obscure the smaller regions. The X stays fixed and the
ordinary training cameras are unaffected. Pass `--cube-spawn-regions` to plot
a different configuration. These are spatial distributions, not grasp-success
heatmaps.

Each proposal draws both cubes independently and uniformly inside their
rectangles. The environment rejects overlapping cubes, insufficient
edge-to-edge XY clearance (`min_gap_m`), and intersections with robots/other
obstacles. Out-of-table bounds are invalid. The accepted distribution is thus
the joint uniform proposal **conditioned on valid initial placement**; when
regions overlap it is not two independent uniform marginals. Exhausting
`max_attempts` raises an error, never silently returns a fallback layout.
There is no IK, grasp-success or policy-dependent rejection. Reachability
and collision-free trajectories are separate controller concerns.

`info` contains the sampled `object_layout`, normalized `cube_spawn_regions`,
and `spawn_attempts`. The public `env.cube_spawn_regions` is a detached copy;
pass a changed configuration to a new environment to change the distribution.
The demo saves `reset.json`, the configuration/seed in `report.json`, and an
exported `scene.xml` containing the actual sampled initial layout. Keep those
alongside the simulator version and robot configuration for reproducibility.
Sampling changes free-cube positions **only at reset**, without recompiling
the model or changing contact settings, actuator limits or saved keyframes.

This preset defines an evaluation distribution, **not a validated region of
guaranteed grasp success**. The fixed-layout expert remains sensitive to
position changes. Before removing the stacking-loop diagnostics, an initial
MuJoCo 3.11.0 smoke check of the **previous 4 × 4 cm regions** with seeds 0,
1 and 2 produced no fully passing episodes: the old audit stopped seeds 0/1 on
thumb/middle-finger self-contact (not proof that the physical grasp could not finish);
seed 2 lifted both cubes and left a supported stack without unintended
contacts, but its 13.3 mm stack offset exceeded the unchanged 12 mm limit.
Those historical audit results are not a success-rate estimate for the current
runner; broader expert validation is still needed before collecting a training
dataset. The 10 × 10 cm preset
has a larger support and is not a validated grasp-success region either.
Keep failures in evaluation statistics. Filtering successful demonstrations
changes the collected data distribution and must not redefine policy resets.

#### One-episode LeRobot smoke collection

`python -m orca_sim.demos.collect_lerobot_smoke` has three explicit stages:

```bash
# Run in the validated simulator environment (MuJoCo, NumPy, Pillow).
python -m orca_sim.demos.collect_lerobot_smoke record \
  --output-dir renders/lerobot_smoke/recording

# Run with LeRobot 0.5.0 and its dataset/video dependencies available.
python -m orca_sim.demos.collect_lerobot_smoke package \
  --recording-dir renders/lerobot_smoke/recording \
  --dataset-dir renders/lerobot_smoke/dataset

# Explicit network write, using the logged-in Hugging Face account.
python -m orca_sim.demos.collect_lerobot_smoke upload \
  --dataset-dir renders/lerobot_smoke/dataset
```

The default repository is `fracapuano/orcapanda_cube_stacking_scripted`; override
`--repo-id` on both package and upload for a different destination. Upload creates
a **private** dataset and refuses to overwrite an existing repository. Record and
package require fresh output directories. No upload happens unless requested.
Recording retains bounded, lossless PNG staging, then LeRobot writes its standard
v3.0 Parquet/video/metadata structure with H.264 videos. All three cameras use the
same pre-action state at 20 Hz, with native rendering settings unchanged.

This is one fixed-layout smoke episode, not the randomized 200-demo collector.
The 48D state and action have explicit joint names; actions are absolute target
angles in radians. The expert still updates at 1 kHz, so the 20 Hz action feature
is subsampled, **not** a command held unchanged for 50 ms. Full 1 kHz controls and
sampled full simulator states are retained in `provenance/`. A 20 Hz learned
policy's equivalent behavior has not been established. The dataset card records
these limitations, hand assignment, layout and task-success report.

Before upload, validation checks all numeric sample alignment, episode boundaries,
video frame counts/timestamps, and LeRobot decoding of each camera. The source
runtime and packaging runtime may differ: set `PYTHONPATH` to this checkout's
`src` when using a separate Python interpreter. Do not silently switch the
simulator's numerical dependencies just to install the dataset writer.

#### Independent collision-audited grasp/lift tests

These separate, explicitly diagnostic tests are not invoked by the stacking loop.

The right arm's attached hand targets red in the saved layout. The saved blue cube is
still at x=0, about 1.09 m horizontally from the left base; the left
palm-first IK path does **not** reach it. Do not treat the earlier free-space
motion demo as proof of a grasp from that layout.

An explicit **test fixture**, `tests/fixtures/bimanual_reach_layout.json`,
moves only blue to `(0.700, 0.201, 0.353)` m. It is not a new scene default:
the saved single-arm/bimanual homes and object layout remain unchanged.
From this checkout:

```bash
# Original saved layout, right hand only:
python -m orca_sim.demos.bimanual_grasp --side right --video --check
# Same proposed reachable layout for BOTH independent episodes:
python -m orca_sim.demos.bimanual_grasp --side both --video --check \
  --object-layout tests/fixtures/bimanual_reach_layout.json \
  --output-dir renders/bimanual_reach_test
```

Each 28-second episode resets the same scene, approaches with the selected
arm, rests its palm on its cube, wraps the fingers, lifts about 18 cm and
checks the last five seconds of holding. The other robot keeps its original
home **commands** and remains dynamic, not welded or frozen. The `--side`
option selects an **arm**: right-arm/native-left-hand targets red;
left-arm/native-right-hand targets blue. This is sequential independent validation, not a
simultaneous/coordinated grasp or stacking policy.

The policy folds the distal thumb before descending, curls distal finger
joints before proximal joints, delays thumb opposition, and keeps the index
partly open to avoid thumb/index contact. The native left hand needs a
different lateral grasp offset. With the hands exchanged, the approach's
small tilt is mirrored for the opposite handedness; it blends from the
unchanged saved arm home. These are controller targets only: no
MJCF mesh, contact mask, gain, torque cap, cube friction or mass is changed.

Every 1 ms physics step is audited. Only the active hand's contacts with its
assigned cube and cube/table support are allowed. Robot/table, inter-robot,
self-contact and wrong-cube contact abort the run, including force-bearing
contacts at positive collision margins. Reports also check loaded
thumb/finger contacts, sustained lift, other-cube movement, penetration,
native force/joint limits, numerical warnings and absence of external forces
or equality attachments. Videos have a close-up inset; `report.json`,
`trace.json` and the exact generated `scene.xml` are saved per side.
These checks establish the recorded trajectories, not safety for arbitrary
teleoperation commands. Run the physics regressions with:

```bash
python -m pytest tests/test_bimanual_grasp.py -q
```

### Saved cube layout and dimensions

The native Panda task loads `assets/mujoco/tabletop_layout.json`: red cube
`(0.438, 0.201, 0.353)` m, blue cube `(0, 0.201, 0.353)` m, and target marker
`(0, 0.201, 0.326)` m. The source scene keeps a generic object layout; the
builder applies these saved world coordinates to both MJCF bodies and reset
keyframes. Robot home/base settings remain in their separate JSON file.

Both cubes are **5.6 cm** per side (20% smaller than the original 7 cm).
Density is unchanged: mass is **30.72 g**, and each diagonal inertia is
**0.00001605632 kg m²**. Centres are 7 mm lower so both bottoms remain at the
0.325 m tabletop. Friction and contact settings are unchanged by this resize.
Scene validation and lift-height measurement use the actual cube dimensions.

### Native Panda assembly and canonical home

Install `python -m pip install -e '.[panda,dev]' 'mujoco==3.11.0'` to use
`OrcaPandaCubeStackingMujoco`. The builder combines the pinned bare Panda
MJCF from `orca_arm` with the corrected native v2 right-hand MJCF from this
package; it does not use `orca_arm`'s preassembled OrcaPanda or any URDF.

The canonical `assets/mujoco/home_pose.json` preserves the selected seven arm
angles, seventeen hand angles, base `(0, -0.224, 0.325)` m and 90-degree yaw.
The builder scopes the source defaults, retains native gains and limits,
and writes the same pose into a reset keyframe. A shared 1.10 × 0.80 m bench
has its top at 0.325 m. The native task uses a 1 ms timestep. No external
holding force, grasp equality or object attachment is added.

```python
from orca_sim import OrcaPandaCubeStackingMujoco

env = OrcaPandaCubeStackingMujoco()
obs, info = env.reset()
print(env.actuator_joints)  # Seven arm + seventeen hand position targets.
obs, reward, terminated, truncated, info = env.step(env.data.ctrl.copy())
env.close()
```

The action order is actuator order, not `qpos` order (which also contains the
free cubes). Custom IK/retargeting can supply these actions without a scripted
policy. `python -m pytest tests/test_mujoco_panda.py -q` verifies preserved
mechanics, canonical reset/export, scratch-data IK and native physics stepping.

### Palm-first grasp and stacking rollout

Install this checkout with `pip install -e '.[panda,render,dev]'`, then run:

```bash
orca-sim-stack --palm-first --stack --video --check
# Headless validation (no rendering dependencies needed):
python -m orca_sim.demos.panda_grasp --palm-first --stack --check
# Interactive viewer on macOS, from the checkout:
mjpython scripts/mujoco_panda_grasp.py --palm-first --stack --viewer --check
```

The installed CLI and source wrapper run the same implementation. Outputs go
to `renders/mujoco_panda_grasp/` by default; use `--output-dir` to keep runs
separate. The report contains source/control hashes, contact depths, support
contacts, lift height and post-release stack stability; `--check` exits
unsuccessfully if the physical checks fail.

The controller keeps the saved home hand orientation during the approach,
rests the open palm on the cube, then wraps the fingers, lifts, transfers,
releases and retreats. Closure requires 0.6 seconds of continuous top contact
with filtered palm load between 0.2 and 5 N. A bounded vertical adjustment
targets a gentle 2 N rest load; this is **not** a force cap during the grasp.
If no stable contact appears by 15.5 seconds, the demo stops with the fingers
open instead of attempting a blind grasp.

All motion uses native position-actuator commands and `mj_step`. IK runs on
scratch data: there are no cube attachments, external applied forces or
live-state teleports. This is a deterministic demonstration for the saved
layout, not a general-purpose grasp planner. The same environment also
accepts arbitrary 24-joint actions from an IK/retargeting pipeline.

The controller tests are `tests/test_palm_grasp.py`,
`tests/test_panda_stacking.py` and `tests/test_panda_demo.py`. The last includes
full-physics grasp, sideways-approach and palm-first stacking regressions.

### Interactive layout and hand-rest editors

With the `panda,render` extras installed, run one editor at a time and open
the loopback URL printed in the terminal:

```bash
orca-sim-pose-editor --mode layout   # Base X/Y and yaw; arm/hand home locked.
orca-sim-pose-editor --mode hand     # 17 hand joints; arm/base locked.
orca-sim-pose-editor --mode objects  # Cube forward/left offsets from the robot.
```

The chosen arm home remains locked in every mode. These are paused
MuJoCo previews, not teleoperation: sliders update kinematics and collisions,
and no simulation time advances. Moving the static base recompiles the model
so its collision structures match the visible geometry. Cube heights stay on
the tabletop and the target marker follows the blue cube.

Save rejects detected intersecting contacts, writes atomically, and makes a
timestamped backup before overwriting an existing JSON. By default it updates
the package's `home_pose.json` or `tabletop_layout.json`; commit those files
to preserve the new defaults. Restart a rollout after saving to load changes.
For a read-only installation or a separate experiment, use
`--save-path /absolute/path/pose.json` (or `layout.json` in objects mode).
Pass that file to the environment as `start_pose=` or `object_layout=`.
The demo CLI accepts `--start-pose`; its cube layout uses the package default.

The local server binds only to `127.0.0.1`, checks request origin and a
session token, and does not expose arbitrary filesystem or command access.
Tests in `test_pose_editor.py` and `test_cube_editor.py` cover saved/reloaded
physics, locked fields, coordinate transforms and collision rejection.

### Native v2 hand contact fixes

Both native v2 hand MJCFs now collide with their ten finger-skin geometries.
Previously these visible surfaces had both collision masks disabled. A scoped
hand contact class uses `solref="0.004 1"`, `solimp="0.999 0.9999 0.001"` and
`priority="1"`. The MJCF also selects elliptic friction and `impratio="10"`;
these two solver options are scene-wide, not per-hand properties. Use a
timestep of at most 2 ms with this contact time constant.

Joint limits, motor gains, ±1 Nm actuator caps, friction coefficients, meshes
and inertias are unchanged. There is no grasp attachment or force-limiting
adapter. This is independent of the Panda arm and any task scene.

`python -m pytest tests/test_hand_contacts.py -q` checks both hands, compares
unchanged mechanics, and steps an identical 40-second increasing-closure
experiment with corrected versus legacy contacts. The free cube is supported
only during acquisition; the support retracts before the strong squeeze.
Vertex-overlap measurements are a lower-bound diagnostic, not proof of zero
intersection for arbitrary objects or actions. Physics validation uses
MuJoCo 3.11.0.

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
