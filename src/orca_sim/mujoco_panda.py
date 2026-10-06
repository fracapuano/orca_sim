"""Native Panda + corrected ORCA v2 hand in the ORCA cube-stacking scene.

No Genesis conversion, geometry proxies, force overrides, or object welds.
The MJCF builder scopes defaults to preserve both components' native mechanics.
"""

from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

PACKAGE = Path(__file__).resolve().parent
HOME_POSE = PACKAGE / "assets/mujoco/home_pose.json"
TABLETOP_LAYOUT = PACKAGE / "assets/mujoco/tabletop_layout.json"
BASE_XY_LIMITS = ((-0.28, 0.58), (-0.28, 0.28))
TABLE_HEIGHT = 0.325
# Keep layout validation and editors in sync with the MJCF's cube geometry.
CUBE_HALF_SIZE = float(
    ET.parse(PACKAGE / "scenes/cube_stacking.xml")
    .getroot()
    .find(".//geom[@name='red_cube_geom']")
    .get("size")
    .split()[2]
)
CUBE_REST_Z = TABLE_HEIGHT + CUBE_HALF_SIZE
# Full cube support, including the blue cube's 4 cm-radius target marker.
OBJECT_XY_LIMITS = ((-0.36, 0.66), (-0.36, 0.36))
HAND_JOINT_NAMES = tuple(
    actuator.get("joint")
    for actuator in ET.parse(PACKAGE / "models/v2/mjcf/orcahand_right.mjcf")
    .getroot()
    .find("actuator")
)
TOWER_ROTATION = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]])


def start_configuration(value=None):
    """Load the user's canonical home; joint angles and base yaw are radians.

    Explicit version-1 files remain readable and describe an open hand.
    The default never falls back to a historical arm configuration.
    """
    defaults = json.loads(HOME_POSE.read_text())
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text())
    value = defaults if value is None else dict(value)
    if value.get("version", 2) not in (1, 2):
        raise ValueError("Unsupported start-pose version")
    arm = np.asarray(value.get("arm_qpos", defaults["arm_qpos"]), dtype=float)
    base = np.asarray(value.get("base_pos", defaults["base_pos"]), dtype=float)
    # Old explicit files always mean zero yaw, even if the default was rotated.
    yaw = np.asarray(value.get("base_yaw", 0.0), dtype=float)
    if yaw.shape != () or not np.isfinite(yaw) or not -np.pi <= yaw <= np.pi:
        raise ValueError("Base yaw must be a finite angle between -pi and pi radians")
    hand = value.get(
        "hand_qpos",
        dict.fromkeys(HAND_JOINT_NAMES, 0.0)
        if value.get("version") == 1
        else defaults["hand_qpos"],
    )
    if not isinstance(hand, dict) or set(hand) != set(HAND_JOINT_NAMES):
        raise ValueError("Expected all 17 named hand joint angles")
    hand_values = np.asarray([hand[name] for name in HAND_JOINT_NAMES], dtype=float)
    if hand_values.shape != (17,) or not np.isfinite(hand_values).all():
        raise ValueError("Expected 17 finite hand joint angles")
    if (
        arm.shape != (7,)
        or base.shape != (3,)
        or not np.isfinite(np.r_[arm, base]).all()
    ):
        raise ValueError(
            "Expected seven finite joint angles and three base coordinates"
        )
    if not (
        BASE_XY_LIMITS[0][0] <= base[0] <= BASE_XY_LIMITS[0][1]
        and BASE_XY_LIMITS[1][0] <= base[1] <= BASE_XY_LIMITS[1][1]
        and abs(base[2] - TABLE_HEIGHT) < 1e-12
    ):
        raise ValueError(
            "Base must sit on the shared tabletop at z=0.325 m, with mounting clearance"
        )
    return {
        "version": 2,
        "arm_qpos": arm.tolist(),
        "base_pos": base.tolist(),
        "base_yaw": float(yaw),
        "hand_qpos": dict(zip(HAND_JOINT_NAMES, hand_values.tolist())),
    }


def base_quaternion(yaw):
    """Upright base rotation about world Z, in MuJoCo's wxyz order."""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def tabletop_configuration(value=None):
    """Load fixed-height, world-frame task positions on the shared bench."""
    if value is None:
        value = TABLETOP_LAYOUT
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text())
    names = ("red_cube", "blue_cube", "stack_target")
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError("Expected red_cube, blue_cube and stack_target positions")
    result = {}
    for name in names:
        pos = np.asarray(value[name], dtype=float)
        if pos.shape != (3,) or not np.isfinite(pos).all():
            raise ValueError(f"Expected three finite coordinates for {name}")
        height = TABLE_HEIGHT + 0.001 if name == "stack_target" else CUBE_REST_Z
        if abs(pos[2] - height) > 1e-12:
            raise ValueError(f"{name} must stay on the tabletop at z={height:.3f} m")
        if any(
            not lo - 1e-12 <= pos[i] <= hi + 1e-12
            for i, (lo, hi) in enumerate(OBJECT_XY_LIMITS)
        ):
            raise ValueError(f"{name} must fit fully on the tabletop")
        result[name] = pos.tolist()
    return result


def panda_source() -> Path:
    try:
        from orca_arm import ASSETS_DIR
    except ImportError as exc:
        raise RuntimeError(
            "Install orca_arm (or pass panda_path) for Panda meshes"
        ) from exc
    return Path(ASSETS_DIR) / "franka_emika_panda" / "panda_nohand.xml"


def scene_xml(
    panda_path: Path | None = None, start_pose=None, object_layout=None
) -> str:
    """Compose source MJCFs without modifying their files or hand parameters."""
    source = Path(panda_path or panda_source()).resolve()
    config = start_configuration(start_pose)
    base = np.asarray(config["base_pos"])
    panda = ET.parse(source).getroot()
    native = ET.parse(PACKAGE / "models/v2/mjcf/orcahand_right.mjcf").getroot()
    root = ET.parse(PACKAGE / "scenes/cube_stacking.xml").getroot()
    # Native Panda task layout, separate from the released standalone-hand scene.
    layout = tabletop_configuration(object_layout)
    for name in ("red_cube", "blue_cube", "stack_target"):
        root.find(f"./worldbody/body[@name='{name}']").set(
            "pos", " ".join(map(str, layout[name]))
        )
    root.set("model", "native_orcapanda_cube_stacking")
    for include in root.findall("include"):
        include.set("file", str((PACKAGE / "scenes" / include.get("file")).resolve()))
    # One continuous bench supports both the robot and the original cubes.
    # Keep the source scene unchanged for standalone-hand consumers.
    table = root.find("./worldbody/body[@name='table']")
    table.set("pos", "0.15 0 0.30")
    table.find("geom[@name='tabletop']").set("size", "0.55 0.40 0.025")
    for geom, (x, y) in zip(
        table.findall("geom")[1:],
        ((-0.50, -0.35), (0.50, -0.35), (-0.50, 0.35), (0.50, 0.35)),
    ):
        geom.set("pos", f"{x} {y} -0.1625")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    option = copy.deepcopy(native.find("option"))
    # Resolve the stiff, face-on cube contacts during post-release settling.
    # Leave all native hand/cube contact and actuator parameters untouched.
    option.set("timestep", "0.001")
    option.set("integrator", "implicitfast")
    root.append(option)
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")

    defaults = ET.SubElement(root, "default")
    defaults.append(copy.deepcopy(panda.find("./default/default[@class='panda']")))
    hand_defaults = copy.deepcopy(native.find("default"))
    hand_defaults.set("class", "native_hand")
    defaults.append(hand_defaults)

    assets = ET.SubElement(root, "asset")
    meshdir = source.parent / panda.find("compiler").get("meshdir", "")
    for asset in panda.find("asset"):
        asset = copy.deepcopy(asset)
        if asset.tag == "mesh":
            mesh_path = meshdir / asset.get("file")
            if not mesh_path.exists():
                mesh_path = source.parent / asset.get("file")
            asset.set("file", str(mesh_path.resolve()))
            asset.set("scale", "1 1 1")  # Scene's ORCA meshes use millimetres.
        elif asset.tag == "material":
            asset.set("name", "panda_" + asset.get("name"))
        assets.append(asset)
    for asset in native.find("asset"):
        asset = copy.deepcopy(asset)
        if asset.tag == "mesh":
            asset.set(
                "file", str((PACKAGE / "models/v2/mjcf" / asset.get("file")).resolve())
            )
            asset.set("scale", "0.001 0.001 0.001")
        assets.append(asset)

    world = ET.SubElement(root, "worldbody")
    arm = copy.deepcopy(panda.find("./worldbody/body[@name='link0']"))
    arm.set("pos", " ".join(map(str, base)))
    arm.set("quat", " ".join(map(str, base_quaternion(config["base_yaw"]))))
    for geom in arm.iter("geom"):
        if "material" in geom.attrib:
            geom.set("material", "panda_" + geom.get("material"))
    attachment = arm.find(".//body[@name='attachment']")
    attachment.attrib.pop("quat", None)
    tower = (
        ET.parse(PACKAGE / "models/v2/mjcf/orcahand_right_body.xml")
        .getroot()
        .find("body")
    )
    tower.set("pos", "0 0 0.0575")
    tower.set("quat", "0 0 0.7071067811865476 0.7071067811865476")
    ET.SubElement(tower, "site", name="hand_frame", size="0.003", rgba="0 0 0 0")
    attachment.append(tower)
    world.append(arm)
    actuators = ET.SubElement(root, "actuator")
    actuators.extend(copy.deepcopy(list(panda.find("actuator"))))
    for actuator in native.find("actuator"):
        actuator = copy.deepcopy(actuator)
        actuator.set("class", "native_hand")
        actuators.append(actuator)
    contacts = copy.deepcopy(panda.find("contact"))
    contacts.extend(copy.deepcopy(list(native.find("contact"))))
    root.append(contacts)
    # Keep a usable home pose in the generated MJCF as well as the Gym reset.
    cube_qpos = []
    for name in ("red_cube", "blue_cube"):
        body = root.find(f"./worldbody/body[@name='{name}']")
        cube_qpos.extend(float(x) for x in body.get("pos").split())
        cube_qpos.extend([1.0, 0.0, 0.0, 0.0])
    keyframes = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframes,
        "key",
        name="orcapanda_home",
        # Tree/qpos ordering differs from actuator ordering for this hand.
        qpos=" ".join(
            map(
                str,
                [
                    *cube_qpos,
                    *config["arm_qpos"],
                    *[
                        config["hand_qpos"][joint.get("name")]
                        for body in tower.iter("body")
                        for joint in body.findall("joint")
                    ],
                ],
            )
        ),
        ctrl=" ".join(
            map(
                str,
                [
                    *config["arm_qpos"],
                    *[config["hand_qpos"][name] for name in HAND_JOINT_NAMES],
                ],
            )
        ),
    )
    return ET.tostring(root, encoding="unicode")


class OrcaPandaCubeStackingMujoco(gym.Env):
    """24 native joint-position actions: 7 Panda + 17 ORCA, in actuator order."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        *,
        panda_path=None,
        frame_skip=5,
        start_pose=None,
        object_layout=None,
    ):
        if render_mode not in (None, "rgb_array", "human"):
            raise ValueError("Unsupported render mode")
        if not isinstance(frame_skip, int) or frame_skip < 1:
            raise ValueError("frame_skip must be a positive integer")
        self.start_pose = start_configuration(start_pose)
        self.object_layout = tabletop_configuration(object_layout)
        self.xml = scene_xml(panda_path, self.start_pose, self.object_layout)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.frame_skip, self.render_mode = frame_skip, render_mode
        self.actuator_joints = tuple(
            self.model.joint(int(j)).name for j in self.model.actuator_trnid[:, 0]
        )
        self.qadr = self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]]
        self.dadr = self.model.jnt_dofadr[self.model.actuator_trnid[:, 0]]
        self.arm_qadr, self.arm_dadr = self.qadr[:7], self.dadr[:7]
        self.action_space = spaces.Box(
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
            dtype=np.float64,
        )
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (self.model.nq + self.model.nv,), dtype=np.float64
        )
        if np.any(
            np.asarray(self.start_pose["arm_qpos"]) < self.action_space.low[:7]
        ) or np.any(
            np.asarray(self.start_pose["arm_qpos"]) > self.action_space.high[:7]
        ):
            raise ValueError("Starting arm angles exceed native joint limits")
        hand = np.array(
            [self.start_pose["hand_qpos"][name] for name in self.actuator_joints[7:]]
        )
        if np.any(hand < self.action_space.low[7:]) or np.any(
            hand > self.action_space.high[7:]
        ):
            raise ValueError("Starting hand angles exceed native joint limits")
        self._renderer = self._viewer = None
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.04, 0, 0.48]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = (
            1.85,
            135,
            -22,
        )
        self.reset()

    def _obs(self):
        return np.concatenate((self.data.qpos.copy(), self.data.qvel.copy()))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(
            self.model, self.data, self.model.key("orcapanda_home").id
        )
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.model.nu,) or not np.isfinite(action).all():
            raise ValueError(f"Expected {self.model.nu} finite actuator targets")
        self.data.ctrl[:] = np.clip(
            action, self.action_space.low, self.action_space.high
        )
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        mujoco.mj_forward(self.model, self.data)
        cube = self.data.body("red_cube").xpos.copy()
        info = {
            "red_cube_position": cube,
            "lift_height_m": float(cube[2] - self.object_layout["red_cube"][2]),
        }
        return self._obs(), float(info["lift_height_m"]), False, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=720, width=1280)
            self._renderer.update_scene(self.data, camera=self.camera)
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer

                self._viewer = viewer.launch_passive(self.model, self.data)
                self._viewer.cam.lookat[:] = self.camera.lookat
                self._viewer.cam.distance = self.camera.distance
                self._viewer.cam.azimuth = self.camera.azimuth
                self._viewer.cam.elevation = self.camera.elevation
            self._viewer.sync()

    def close(self):
        for handle in (self._viewer, self._renderer):
            if handle is not None:
                handle.close()
        self._renderer = self._viewer = None


def inverse_kinematics(env, position, rotation=TOWER_ROTATION, seed=None):
    """Damped 6D Jacobian IK on scratch data; never teleports the simulation."""
    m = env.model
    d = mujoco.MjData(m)
    d.qpos[:] = env.data.qpos
    if seed is not None:
        d.qpos[env.arm_qadr] = seed
    site = m.site("hand_frame").id
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    target_quat = np.empty(4)
    mujoco.mju_mat2Quat(target_quat, np.asarray(rotation).ravel())
    for _ in range(500):
        mujoco.mj_forward(m, d)
        current_quat = np.empty(4)
        mujoco.mju_mat2Quat(current_quat, d.site_xmat[site])
        rotational_error = np.empty(3)
        mujoco.mju_subQuat(rotational_error, target_quat, current_quat)
        # mju_subQuat returns the rotation vector in the current local frame.
        rotational_error = d.site_xmat[site].reshape(3, 3) @ rotational_error
        error = np.r_[np.asarray(position) - d.site_xpos[site], rotational_error]
        if np.linalg.norm(error[:3]) < 1e-5 and np.linalg.norm(error[3:]) < 1e-4:
            return d.qpos[env.arm_qadr].copy()
        mujoco.mj_jacSite(m, d, jp, jr, site)
        jac = np.vstack((jp[:, env.arm_dadr], jr[:, env.arm_dadr]))
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), error)
        # Use the redundant joint to move away from limits. Clipping alone can
        # strand a valid target at a limit with a small but persistent error.
        nullspace = np.eye(7) - np.linalg.pinv(jac, rcond=1e-5) @ jac
        centre = (env.action_space.low[:7] + env.action_space.high[:7]) / 2
        dq += nullspace @ (
            0.005
            * min(1.0, np.linalg.norm(error) * 20)
            * (centre - d.qpos[env.arm_qadr])
        )
        dq *= min(1.0, 0.15 / max(np.max(np.abs(dq)), 1e-12))
        d.qpos[env.arm_qadr] = np.clip(
            d.qpos[env.arm_qadr] + dq,
            env.action_space.low[:7] + 0.001,
            env.action_space.high[:7] - 0.001,
        )
    if seed is None:
        # The saved home can lie in another IK basin. Restarts affect scratch
        # data only; the live robot still starts at exactly its saved keyframe.
        rng = np.random.default_rng(17)
        solutions = []
        for candidate in rng.uniform(
            env.action_space.low[:7] + 0.01,
            env.action_space.high[:7] - 0.01,
            size=(24, 7),
        ):
            try:
                solution = inverse_kinematics(env, position, rotation, seed=candidate)
                d.qpos[env.arm_qadr] = solution
                mujoco.mj_forward(m, d)
                if not any(
                    c.dist < -0.001
                    and any(
                        m.body(m.geom_bodyid[g]).name.startswith("link")
                        for g in (c.geom1, c.geom2)
                    )
                    for c in d.contact
                ):
                    solutions.append(solution)
            except RuntimeError:
                continue
        if solutions:
            return min(
                solutions, key=lambda q: np.linalg.norm(q - env.data.qpos[env.arm_qadr])
            )
    raise RuntimeError(
        f"IK failed: position error {np.linalg.norm(error[:3]):.4f} m, orientation error {np.linalg.norm(error[3:]):.4f} rad"
    )
