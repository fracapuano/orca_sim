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
