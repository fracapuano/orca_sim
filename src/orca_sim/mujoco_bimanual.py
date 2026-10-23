"""Two native Panda arms with corrected left/right ORCA hands in one MuJoCo world.

Only assembly transforms and names change. Meshes, native actuator parameters,
hand contacts and inter-robot collisions are preserved. The single-arm scene
and its saved defaults are never edited by this builder.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from gymnasium import spaces

from orca_sim import mujoco_panda as single
from orca_sim.cube_sampling import CUBES, sample_xy, separated, spawn_configuration

SIDES = ("right", "left")
# Arm identities/homes stay fixed; only the native hands exchange mounts.
HAND_FOR_ARM = {"right": "left", "left": "right"}
CAMERA_NAMES = ("front", "right_wrist", "left_wrist")
HOME_POSE = single.PACKAGE / "assets/mujoco/bimanual_home_pose.json"
STACK_LAYOUT = single.PACKAGE / "assets/mujoco/bimanual_stack_layout.json"
CUBE_SPAWN_REGIONS = single.PACKAGE / "assets/mujoco/bimanual_cube_spawn_regions.json"
TABLE_CENTER = np.array([0.5, 0.0, 0.30])
TABLE_HALF_SIZE = np.array([0.9, 0.40, 0.025])
BASE_XY_LIMITS = ((-0.28, 1.28), (-0.28, 0.28))
OBJECT_XY_LIMITS = ((-0.36, 1.36), (-0.36, 0.36))


def _within_table(position, limits, height, label):
    pos = np.asarray(position, dtype=float)
    if pos.shape != (3,) or not np.isfinite(pos).all():
        raise ValueError(f"{label} needs three finite coordinates")
    if abs(pos[2] - height) > 1e-12 or any(
        not lo - 1e-12 <= pos[i] <= hi + 1e-12 for i, (lo, hi) in enumerate(limits)
    ):
        raise ValueError(f"{label} must fit on the tabletop at z={height} m")
    return pos.tolist()


def start_configuration(value=None):
    """Load independent named homes; angle units are radians, positions metres."""
    if value is None:
        value = HOME_POSE
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text())
    if not isinstance(value, dict) or set(value) != {"version", *SIDES}:
        raise ValueError("Expected version, right and left bimanual home entries")
    if value["version"] != 1:
        raise ValueError("Unsupported bimanual home version")
    result = {"version": 1}
    for side in SIDES:
        config = value[side]
        if not isinstance(config, dict) or set(config) != {
            "arm_qpos",
            "hand_qpos",
            "base_pos",
            "base_yaw",
        }:
            raise ValueError(f"Incomplete {side} robot home")
        arm = np.asarray(config["arm_qpos"], dtype=float)
        if arm.shape != (7,) or not np.isfinite(arm).all():
            raise ValueError(f"Expected seven finite {side} arm angles")
        names = tuple(
            n.replace("right_", side + "_", 1) for n in single.HAND_JOINT_NAMES
        )
        hand = config["hand_qpos"]
        if not isinstance(hand, dict) or set(hand) != set(names):
            raise ValueError(f"Expected all 17 named {side} hand angles")
        angles = np.asarray([hand[n] for n in names], dtype=float)
        if angles.shape != (17,) or not np.isfinite(angles).all():
            raise ValueError(f"Expected finite {side} hand angles")
        yaw = np.asarray(config["base_yaw"], dtype=float)
        if yaw.shape != () or not np.isfinite(yaw) or not -np.pi <= yaw <= np.pi:
            raise ValueError(f"Invalid {side} base yaw")
        result[side] = dict(
            base_pos=_within_table(
                config["base_pos"], BASE_XY_LIMITS, single.TABLE_HEIGHT, side
            ),
            base_yaw=float(yaw),
            arm_qpos=arm.tolist(),
            hand_qpos=dict(zip(names, angles.tolist())),
        )
    if (
        np.linalg.norm(
            np.subtract(result["right"]["base_pos"][:2], result["left"]["base_pos"][:2])
        )
        < 0.24
    ):
        raise ValueError("Robot bases need at least 0.24 m mounting separation")
    return result


def tabletop_configuration(value=None):
    if value is None:
        value = single.TABLETOP_LAYOUT
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text())
    if not isinstance(value, dict) or set(value) != {
        "red_cube",
        "blue_cube",
        "stack_target",
    }:
        raise ValueError("Expected both cubes and the stack target")
    return {
        name: _within_table(
            pos,
            OBJECT_XY_LIMITS,
            single.TABLE_HEIGHT + 0.001
            if name == "stack_target"
            else single.CUBE_REST_Z,
            name,
        )
        for name, pos in value.items()
    }


def _rename(tree, names):
    """Rewrite this assembly's object names/references, not meshes or classes."""
    for node in tree.iter():
        for attr in ("name", "joint", "body1", "body2", "site"):
            if node.get(attr) in names:
                node.set(attr, names[node.get(attr)])


def _camera_xyaxes(direction, up):
    """MuJoCo cameras look along -Z; image right/up are local +X/+Y."""
    z = -np.asarray(direction, dtype=float)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    return " ".join(map(str, np.concatenate((x, np.cross(z, x)))))


def _add_cameras(root, hands):
    # Replace the inherited single-hand topdown/angled cameras. The bimanual
    # MJCF has exactly three task cameras; the UI's free camera is not a sensor.
    for parent in root.iter():
        for camera in parent.findall("camera"):
            parent.remove(camera)
    # "front" is the shared task camera, physically BEHIND the robot bases.
    eye = np.array([0.5, -1.5, 1.4])
    ET.SubElement(
        root.find("worldbody"),
        "camera",
        name="front",
        mode="fixed",
        pos=" ".join(map(str, eye)),
        xyaxes=_camera_xyaxes(np.array([0.5, 0.025, 0.46]) - eye, [0, 0, 1]),
        fovy="43",
    )
    for arm in SIDES:
        hand = hands[HAND_FOR_ARM[arm]]
        # Native carpal frame: +Y points out of the palm, +Z toward fingers.
        # Place the optical centre under the heel of the palm, behind the
        # contact patch, not inside the palm mesh or the grasped cube.
        palm = next(
            b
            for b in hand.iter("body")
            if b.find("joint[@name='" + HAND_FOR_ARM[arm] + "_wrist']") is not None
        )
        ET.SubElement(
            palm,
            "camera",
            name=arm + "_wrist",
            mode="fixed",
            pos="0.005 0.0418770780631086 -0.0568000006705525",
            xyaxes=_camera_xyaxes([0, 0.5, np.sqrt(0.75)], [0, -1, 0]),
            fovy="80",
        )


def scene_xml(
    panda_path=None, start_pose=None, object_layout=None, target_marker="disk"
):
    if target_marker not in ("disk", "x"):
        raise ValueError("target_marker must be 'disk' or 'x'")
    home = start_configuration(start_pose)
    objects = tabletop_configuration(object_layout)
    # Reuse the tested single-arm composition and its source/default handling.
    root = ET.fromstring(single.scene_xml(panda_path))
    root.set("model", "native_bimanual_orcapanda_cube_stacking")
    table = root.find("./worldbody/body[@name='table']")
    table.set("pos", " ".join(map(str, TABLE_CENTER)))
    table.find("geom[@name='tabletop']").set(
        "size", " ".join(map(str, TABLE_HALF_SIZE))
    )
    for geom, (x, y) in zip(
        table.findall("geom")[1:],
        ((-0.85, -0.35), (0.85, -0.35), (-0.85, 0.35), (0.85, 0.35)),
    ):
        geom.set("pos", f"{x} {y} -0.1625")
    for name, pos in objects.items():
        root.find(f"./worldbody/body[@name='{name}']").set(
            "pos", " ".join(map(str, pos))
        )
    if target_marker == "x":
        marker = root.find("./worldbody/body[@name='stack_target']")
        marker.remove(marker.find("geom"))
        marker.find("site").set("rgba", "0 0 0 0")
        for sign in (-1, 1):
            ET.SubElement(
                marker,
                "geom",
                name=f"stack_target_x_{'a' if sign < 0 else 'b'}",
                type="box",
                size="0.065 0.004 0.0003",
                quat=f"0.9238795325 0 0 {sign * 0.3826834324}",
                rgba="0.035 0.035 0.035 1",
                contype="0",
                conaffinity="0",
            )

    right = root.find("./worldbody/body[@name='link0']")
    right_attachment = right.find(".//body[@name='attachment']")
    left = copy.deepcopy(right)
    left_attachment = left.find(".//body[@name='attachment']")
    left_attachment.remove(left_attachment.find("body[@name='right_tower']"))
    native = ET.parse(single.PACKAGE / "models/v2/mjcf/orcahand_left.mjcf").getroot()
    hand = (
        ET.parse(single.PACKAGE / "models/v2/mjcf/orcahand_left_body.xml")
        .getroot()
        .find("body")
    )
    for attr in ("pos", "quat"):
        hand.set(attr, right_attachment.find("body[@name='right_tower']").get(attr))
    ET.SubElement(hand, "site", name="left_hand_frame", size="0.003", rgba="0 0 0 0")
    left_attachment.append(hand)

    defaults = copy.deepcopy(native.find("default"))
    defaults.set("class", "native_left_hand")
    root.find("default").append(defaults)
    assets = root.find("asset")
    for asset in native.find("asset"):
        asset = copy.deepcopy(asset)
        if asset.tag == "mesh":
            asset.set(
                "file",
                str((single.PACKAGE / "models/v2/mjcf" / asset.get("file")).resolve()),
            )
            asset.set("scale", "0.001 0.001 0.001")
        assets.append(asset)

    # Panda visual/collision meshes and defaults are shared, but all movable
    # objects and actuators have side-specific names. Hands keep native names.
    panda_names = {
        n.get("name")
        for n in left.iter()
        if n.get("name") and not n.get("name").startswith("left_")
    }
    right_map = {name: "right_panda_" + name for name in panda_names}
    right_map["hand_frame"] = "right_hand_frame"
    left_map = {name: "left_panda_" + name for name in panda_names}
    _rename(right, right_map)
    _rename(left, left_map)
    actuator = root.find("actuator")
    arm_actuators = [copy.deepcopy(a) for a in list(actuator)[:7]]
    for side, body, mapping in (("right", right, right_map), ("left", left, left_map)):
        config = home[side]
        body.set("pos", " ".join(map(str, config["base_pos"])))
        body.set("quat", " ".join(map(str, single.base_quaternion(config["base_yaw"]))))
        for original in arm_actuators:
            mapping[original.get("name")] = side + "_panda_" + original.get("name")
    for a in list(actuator)[:7]:
        _rename(a, right_map)
    for a in arm_actuators:
        _rename(a, left_map)
        actuator.append(a)
    for a in native.find("actuator"):
        a = copy.deepcopy(a)
        a.set("class", "native_left_hand")
        actuator.append(a)
    right_contacts = root.find("contact")
    panda_contacts = [
        copy.deepcopy(c) for c in right_contacts if c.get("body1") in panda_names
    ]
    _rename(right_contacts, right_map)
    for c in panda_contacts:
        _rename(c, left_map)
        right_contacts.append(c)
    right_contacts.extend(copy.deepcopy(list(native.find("contact"))))

    # Exchange complete native hand assemblies, not Panda bases or arm poses.
    # Site names and 24-target action blocks continue to identify the ARM;
    # native hand body/joint/mesh names continue to identify handedness.
    attachments = {"right": right_attachment, "left": left_attachment}
    hands = {s: a.find(f"body[@name='{s}_tower']") for s, a in attachments.items()}
    for side in SIDES:
        attachments[side].remove(hands[side])
    for side in SIDES:
        hand_side = HAND_FOR_ARM[side]
        hand = hands[hand_side]
        hand.find(f"site[@name='{hand_side}_hand_frame']").set(
            "name", side + "_hand_frame"
        )
        attachments[side].append(hand)
    original_actuators = list(actuator)
    for a in original_actuators:
        actuator.remove(a)
    for side in SIDES:
        actuator.extend(
            a for a in original_actuators if a.get("joint").startswith(side + "_panda_")
        )
        hand_side = HAND_FOR_ARM[side]
        actuator.extend(
            a
            for a in original_actuators
            if a.get("joint").startswith(hand_side + "_")
            and not a.get("joint").startswith(hand_side + "_panda_")
        )

    # No inter-arm/hand exclusions: robots must physically collide if commanded
    # into each other. Nominal motion is checked separately, not collision-masked.
    root.findall("worldbody")[-1].append(left)
    _add_cameras(root, hands)

    positions = {
        f"{side}_panda_joint{i + 1}": q
        for side in SIDES
        for i, q in enumerate(home[side]["arm_qpos"])
    }
    for side in SIDES:
        positions.update(
            (HAND_FOR_ARM[side] + "_" + name.removeprefix(side + "_"), angle)
            for name, angle in home[side]["hand_qpos"].items()
        )
    qpos = []
    for name in ("red_cube", "blue_cube"):
        qpos.extend([*objects[name], 1, 0, 0, 0])
    for robot in (right, left):
        # Match MJCF body traversal, which differs from hand actuator ordering.
        qpos.extend(
            positions[j.get("name")]
            for b in robot.iter("body")
            for j in b.findall("joint")
        )
    for keyframes in root.findall("keyframe"):
        root.remove(keyframes)
    keyframes = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframes,
        "key",
        name="orcapanda_home",
        qpos=" ".join(map(str, qpos)),
        ctrl=" ".join(str(positions[a.get("joint")]) for a in actuator),
    )
    return ET.tostring(root, encoding="unicode")


class OrcaBimanualPandaCubeStackingMujoco(single.OrcaPandaCubeStackingMujoco):
    """48 targets: right arm + native left hand, left arm + native right hand.

    Shares the single-arm native stepping/rendering implementation. Homes and
    assembly are independent. IK is kinematic, not a collision-aware planner.
    """

    camera_names = CAMERA_NAMES

    def __init__(
        self,
        render_mode=None,
        *,
        panda_path=None,
        frame_skip=5,
        start_pose=None,
        object_layout=None,
        target_marker="disk",
        cube_spawn_regions=None,
    ):
        if render_mode not in (None, "rgb_array", "human"):
            raise ValueError("Unsupported render mode")
        if not isinstance(frame_skip, int) or frame_skip < 1:
            raise ValueError("frame_skip must be a positive integer")
        self.start_pose = start_configuration(start_pose)
        self.object_layout = tabletop_configuration(object_layout)
        self._fixed_object_layout = copy.deepcopy(self.object_layout)
        self._cube_spawn_regions = spawn_configuration(
            cube_spawn_regions, OBJECT_XY_LIMITS
        )
        self.xml = scene_xml(
            panda_path, self.start_pose, self.object_layout, target_marker
        )
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.frame_skip, self.render_mode = frame_skip, render_mode
        self.actuator_joints = tuple(
            self.model.joint(int(j)).name for j in self.model.actuator_trnid[:, 0]
        )
        self.qadr = self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]]
        self.dadr = self.model.jnt_dofadr[self.model.actuator_trnid[:, 0]]
        self.action_space = spaces.Box(
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
            dtype=np.float64,
        )
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (self.model.nq + self.model.nv,), dtype=np.float64
        )
        self.robots = {}
        self.hand_assignment = HAND_FOR_ARM.copy()
        for side, offset in (("right", 0), ("left", 24)):
            indices = np.arange(offset, offset + 24)
            self.robots[side] = SimpleNamespace(
                model=self.model,
                data=self.data,
                actuator_indices=indices,
                arm_qadr=self.qadr[indices[:7]],
                arm_dadr=self.dadr[indices[:7]],
                hand_qadr=self.qadr[indices[7:]],
                hand_side=self.hand_assignment[side],
                hand_site=side + "_hand_frame",
                actuator_joints=self.actuator_joints[offset : offset + 24],
                action_space=spaces.Box(
                    self.action_space.low[indices],
                    self.action_space.high[indices],
                    dtype=np.float64,
                ),
            )
        # Ownership follows the kinematic tree, not the native hand's prefix.
        self.body_robots = []
        for body in range(self.model.nbody):
            owner = None
            ancestor = body
            while ancestor:
                name = self.model.body(ancestor).name
                owner = next((s for s in SIDES if name == s + "_panda_link0"), None)
                if owner is not None:
                    break
                ancestor = int(self.model.body_parentid[ancestor])
            self.body_robots.append(owner)
        self.geom_robots = tuple(self.body_robots[b] for b in self.model.geom_bodyid)
        home_ctrl = self.model.key("orcapanda_home").ctrl
        if np.any(home_ctrl < self.action_space.low) or np.any(
            home_ctrl > self.action_space.high
        ):
            raise ValueError("Starting angles exceed native joint limits")
        self._renderer = self._viewer = None
        self._camera_renderer = None
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.5, 0.04, 0.61]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 2.55, 95, -22
        self.reset()
        contacts = self.contact_metrics()
        if (
            max(contacts["inter_robot_depth_m"], contacts["robot_table_depth_m"])
            > 0.0001
        ):
            raise ValueError("Starting robots intersect each other or the table")

    @property
    def cube_spawn_regions(self):
        """A detached, JSON-serializable copy of the reset distribution (or None)."""
        return copy.deepcopy(self._cube_spawn_regions)

    def reset(self, *, seed=None, options=None):
        """Restore home and sample cube XY using Gymnasium's per-env RNG.

        Same configuration and seed give the same reset for a scripted expert
        or an autonomous policy. Repeated reset() calls advance the RNG stream.
        Cube height/orientation, robot homes and stack target are not randomized.
        No IK/controller-success rejection is applied to this distribution.
        """
        super().reset(seed=seed, options=options)
        self.object_layout = copy.deepcopy(self._fixed_object_layout)
        config = self._cube_spawn_regions
        attempts = 0
        if config is not None:
            m, d = self.model, self.data
            geoms = {name: m.geom(name + "_geom").id for name in CUBES}
            cube_ids = set(geoms.values())
            sizes = {name: m.geom_size[g] for name, g in geoms.items()}
            addresses = {
                name: m.jnt_qposadr[m.body_jntadr[m.body(name).id]] for name in CUBES
            }
            for attempts in range(1, config["max_attempts"] + 1):
                xy = sample_xy(self.np_random, config)
                if not separated(xy, sizes, config["min_gap_m"]):
                    continue
                for name in CUBES:
                    adr = addresses[name]
                    d.qpos[adr : adr + 2] = xy[name]
                mujoco.mj_forward(m, d)
                # Reject active contacts with robots/obstacles, including
                # positive collision margins. The table is intended support.
                invalid = False
                for contact in d.contact:
                    g1, g2 = int(contact.geom1), int(contact.geom2)
                    if not cube_ids.intersection((g1, g2)):
                        continue
                    other = g2 if g1 in cube_ids else g1
                    if m.body(
                        m.geom_bodyid[other]
                    ).name != "table" and contact.dist <= max(
                        0.0, contact.includemargin
                    ):
                        invalid = True
                        break
                if not invalid:
                    for name in CUBES:
                        self.object_layout[name][:2] = xy[name].tolist()
                    break
            else:
                # Do not silently fall back to a different distribution or
                # leave an invalid candidate as a usable episode.
                super().reset()
                raise ValueError(
                    f"No valid cube placement after {attempts} proposals; "
                    "adjust cube_spawn_regions, min_gap_m or the robot layout"
                )
        return self._obs(), dict(
            object_layout=copy.deepcopy(self.object_layout),
            cube_spawn_regions=self.cube_spawn_regions,
            spawn_attempts=attempts,
        )

    def render_camera(self, name, *, width=640, height=480):
        """Independent RGB uint8 image from a named camera; never steps physics.

        Available regardless of render_mode. The ordinary render()/interactive
        orbit camera is unchanged. A single offscreen context serves all three
        cameras. Call close() to release it. Requires the render extra/OpenGL.
        """
        if name not in self.camera_names:
            raise ValueError(
                f"Unknown camera {name!r}; choose from {self.camera_names}"
            )
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (width, height)):
            raise ValueError("Camera width/height must be positive integers")
        if not (
            0 < width <= self.model.vis.global_.offwidth
            and 0 < height <= self.model.vis.global_.offheight
        ):
            raise ValueError(
                "Camera size exceeds the 1280x720 offscreen buffer or is nonpositive"
            )
        renderer = self._camera_renderer
        if renderer is not None and (renderer.width, renderer.height) != (
            width,
            height,
        ):
            renderer.close()
            self._camera_renderer = renderer = None
        if renderer is None:
            self._camera_renderer = renderer = mujoco.Renderer(
                self.model, width=width, height=height
            )
        renderer.update_scene(self.data, camera=name)
        return renderer.render().copy()

    def render_cameras(self, *, width=640, height=480):
        """Three synchronized RGB images from the current state (no stepping).

        Names identify arms, not native hand handedness. As with step()/render(),
        callers must not mutate the environment concurrently during capture.
        """
        return {
            name: self.render_camera(name, width=width, height=height)
            for name in self.camera_names
        }

    def close(self):
        if self._camera_renderer is not None:
            self._camera_renderer.close()
            self._camera_renderer = None
        super().close()

    def contact_metrics(self):
        """Report unwanted inter-robot/table contacts, without masking any pairs."""
        inter = table = 0.0
        count = 0
        for c in self.data.contact:
            names = [
                self.model.body(self.model.geom_bodyid[g]).name
                for g in (c.geom1, c.geom2)
            ]
            sides = {self.geom_robots[g] for g in (c.geom1, c.geom2)} - {None}
            if len(sides) == 2:
                inter = max(inter, -float(c.dist))
                count += int(c.dist < 0)
            elif sides and "table" in names:
                table = max(table, -float(c.dist))
        return dict(
            inter_robot_depth_m=inter,
            inter_robot_contacts=count,
            robot_table_depth_m=table,
        )

    def solve_ik(self, side, position, rotation=None, seed=None):
        """Return seven arm targets without changing either robot's live state."""
        if side not in self.robots:
            raise ValueError("side must be 'right' or 'left'")
        robot = self.robots[side]
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Expected a finite 3D target position")
        if rotation is None:
            rotation = self.data.site(robot.hand_site).xmat.reshape(3, 3).copy()
        rotation = np.asarray(rotation, dtype=float)
        if (
            rotation.shape != (3, 3)
            or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1)
        ):
            raise ValueError("Expected a proper 3x3 rotation matrix")
        if seed is not None:
            seed = np.asarray(seed, dtype=float)
            if seed.shape != (7,) or not np.isfinite(seed).all():
                raise ValueError("Expected seven finite IK seed angles")
        return single.inverse_kinematics(
            robot,
            position,
            rotation,
            seed,
            site_name=robot.hand_site,
            collision_prefix=side + "_",
            collision_body_ids={
                i for i, owner in enumerate(self.body_robots) if owner == side
            },
        )

    def side_action(self, side, targets, *, action=None):
        """Replace one robot's 24 targets, preserving the other robot's command."""
        if side not in self.robots:
            raise ValueError("side must be 'right' or 'left'")
        targets = np.asarray(targets, dtype=float)
        if targets.shape != (24,) or not np.isfinite(targets).all():
            raise ValueError("Expected 24 finite side targets")
        result = np.array(
            self.data.ctrl if action is None else action, dtype=float, copy=True
        )
        if result.shape != (48,) or not np.isfinite(result).all():
            raise ValueError("Expected 48 finite combined targets")
        result[self.robots[side].actuator_indices] = targets
        return result
