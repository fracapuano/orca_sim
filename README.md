<p align="center">
  <img src="https://huggingface.co/datasets/fracapuano/blogs/resolve/main/orca_sim.png" alt="orca_sim header" width="600"/>
</p>


`orca_sim` provides simulation environments for the ORCA hand.
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
