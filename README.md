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

## Genesis bimanual ORCA-Panda cube grasp

The Genesis environment uses the two 7-DOF Franka arms and two 17-DOF ORCA
hands from [`fracapuano/orca_arm`](https://github.com/fracapuano/orca_arm). It
adds a grounded workbench asset and a dynamic 5 cm, 80 g cube, exposes all 48
actuated joints through the Gymnasium API, and includes a deterministic
side-grasp/lift validation. The implementation follows the [Genesis user
guide](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/index.html)
and currently pins Genesis 1.3.3 for repeatability.

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

The command exits non-zero unless the cube is finite, rises at least 12 cm,
remains in contact with at least four hand collision geometries, has low final
linear speed, and every lift inverse-kinematics solve is within tolerance. Use
`--render-mode human` for the interactive viewer or `--render-mode none` for the
fastest validation. The first run is slower because Genesis imports the meshes
and compiles its simulation kernels.

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

The Panda inertias in the upstream MJCF are preserved. The ORCA hand's authored
CAD inertias do not match its metre-scale collision geometry, which causes
unstable contact in Genesis. At load time, `orca_sim` generates a cached MJCF
that removes only the 38 hand-link inertials and asks Genesis to integrate them
from the collision meshes at 650 kg/m³. It also resolves every mesh to an
absolute path; the upstream asset itself is never modified.

The workbench combines a collidable 0.75 × 1.15 m top with a packaged MJCF frame
containing four grounded steel legs, aprons, lower rails, and feet. The frame is
visual-only so it cannot introduce irrelevant robot–furniture contacts; the
cube-contact surface remains at the same calibrated height as the validated
grasp.

The validated rigid-contact configuration uses a 2.5 ms control step with five
physics substeps, 150 constraint iterations, five no-slip iterations, box-box
contact detection, and torsional friction. The scripted trajectory closes the
right hand gradually and lifts through five Cartesian wrist waypoints rather
than teleporting bodies.

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
