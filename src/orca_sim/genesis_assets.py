"""Build the Genesis OrcaPanda from native MJCF components.

The Panda arm comes from the no-hand MuJoCo Menagerie model bundled by
``orca_arm``.  The hand bodies, meshes, joints, and actuators come directly
from the v2 MJCF release bundled by ``orca_sim``.  In particular, this module
does not consume the URDF-derived hand subtrees in ``bimanual_orcapanda.xml``.
"""

from __future__ import annotations

import copy
import hashlib
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_HAND_DENSITY = 650.0
LEFT_ARM_ROOT = "left_panda_link0"
RIGHT_ARM_ROOT = "right_panda_link0"

_PACKAGE_DIR = Path(__file__).resolve().parent
_NATIVE_HAND_MJCF_DIR = _PACKAGE_DIR / "models" / "v2" / "mjcf"
_NATIVE_HAND_OPTIONS = _PACKAGE_DIR / "models" / "v2" / "assets" / "options.xml"
_HAND_SPECS = (
    {
        "side": "left",
        "base_pos": "0 0.45 0",
        "mount_quat": "0.7071067811865476 0.7071067811865476 0 0",
    },
    {
        "side": "right",
        "base_pos": "0 -0.45 0",
        "mount_quat": "0 0 0.7071067811865476 0.7071067811865476",
    },
)
_HAND_MATERIAL_PREFIX = "orca_"


def resolve_panda_nohand_mjcf() -> Path:
    """Return the Menagerie Panda MJCF without an end effector."""
    try:
        from orca_arm import ASSETS_DIR
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "The Genesis environment needs the Panda MJCF assets from orca_arm. "
            "Install this project with `pip install -e '.[genesis]'`."
        ) from exc

    source = (
        Path(ASSETS_DIR).expanduser().resolve()
        / "franka_emika_panda"
        / "panda_nohand.xml"
    )
    if not source.is_file():
        raise FileNotFoundError(f"No-hand Panda MJCF not found: {source}")
    return source


def _native_hand_paths(side: str) -> tuple[Path, Path]:
    definition = _NATIVE_HAND_MJCF_DIR / f"orcahand_{side}.mjcf"
    body = _NATIVE_HAND_MJCF_DIR / f"orcahand_{side}_body.xml"
    for path in (definition, body, _NATIVE_HAND_OPTIONS):
        if not path.is_file():
            raise FileNotFoundError(
                f"Native ORCA-hand MJCF component not found: {path}"
            )
    return definition, body


def _set_valid_home_ref(joint: ET.Element) -> None:
    joint_range = joint.get("range")
    if not joint_range:
        return
    lower, upper = (float(value) for value in joint_range.split())
    home = min(max(0.0, lower), upper)
    if home != 0.0:
        joint.set("ref", f"{home:.12g}")


def _namespace_panda_subtree(
    source_body: ET.Element,
    *,
    side: str,
    base_pos: str,
) -> ET.Element:
    """Clone one Panda arm and namespace every kinematic identifier."""
    body = copy.deepcopy(source_body)
    body.set("pos", base_pos)

    for child_body in body.iter("body"):
        name = child_body.get("name")
        if name == "attachment":
            child_body.set("name", f"{side}_panda_link8")
            child_body.set("pos", "0 0 0.107")
            child_body.attrib.pop("quat", None)
        elif name and re.fullmatch(r"link[0-7]", name):
            child_body.set("name", f"{side}_panda_{name}")

    for joint in body.iter("joint"):
        name = joint.get("name")
        if name and re.fullmatch(r"joint[1-7]", name):
            joint.set("name", f"{side}_panda_{name}")
            _set_valid_home_ref(joint)

    for site in body.iter("site"):
        name = site.get("name")
        if name:
            site.set("name", f"{side}_panda_{name}")
    return body


def _append_native_materials(root: ET.Element) -> dict[str, str]:
    asset = root.find("asset")
    options_asset = ET.parse(_NATIVE_HAND_OPTIONS).getroot().find("asset")
    if asset is None or options_asset is None:
        raise ValueError("Missing MJCF asset section while composing native hands")

    material_names: dict[str, str] = {}
    for material in options_asset.findall("material"):
        name = material.get("name")
        if name is None:
            continue
        clone = copy.deepcopy(material)
        namespaced_name = f"{_HAND_MATERIAL_PREFIX}{name}"
        clone.set("name", namespaced_name)
        asset.append(clone)
        material_names[name] = namespaced_name
    return material_names


def _materialize_native_body_defaults(
    tower: ET.Element,
    *,
    material_names: dict[str, str],
    hand_density: float,
) -> tuple[set[str], int]:
    """Make the standalone hand defaults explicit and add skin collision geoms."""
    body_names = {
        body.get("name", "") for body in tower.iter("body") if body.get("name")
    }
    density = f"{hand_density:g}"
    skin_collision_count = 0

    for body in tower.iter("body"):
        for joint in body.findall("joint"):
            defaults = {
                "type": "hinge",
                "limited": "true",
                "damping": "0.1",
                "armature": "0.001",
                "margin": "0.01",
                "frictionloss": "0.001",
            }
            for name, value in defaults.items():
                joint.attrib.setdefault(name, value)

        # Iterate over a snapshot because skin proxies are appended below.
        for geom in list(body.findall("geom")):
            if geom.get("mesh"):
                geom.attrib.setdefault("type", "mesh")
            geom.attrib.setdefault("friction", "1 0.005 0.001")
            geom.attrib.setdefault("condim", "3")
            geom.attrib.setdefault("margin", "0.0005")
            geom.attrib.setdefault("contype", "1")
            geom.attrib.setdefault("conaffinity", "1")

            material = geom.get("material")
            if material in material_names:
                geom.set("material", material_names[material])

            is_visual = geom.get("contype") == "0" and geom.get("conaffinity") == "0"
            if is_visual:
                geom.set("density", "0")
            else:
                geom.attrib.pop("mass", None)
                geom.set("density", density)
                geom.set("contype", "6")
                geom.set("conaffinity", "6")
                geom.set("solref", "0.001 1")
                geom.set("solimp", "0.99 0.999 0.001")

            mesh_name = geom.get("mesh", "")
            if is_visual and "Skin" in mesh_name:
                collision = copy.deepcopy(geom)
                collision.set(
                    "name", f"genesis_{mesh_name}_collision_{skin_collision_count}"
                )
                collision.set("material", material_names["invisible"])
                collision.set("rgba", "0 0 0 0")
                collision.set("group", "3")
                collision.set("density", density)
                collision.set("contype", "6")
                collision.set("conaffinity", "6")
                collision.set("solref", "0.001 1")
                collision.set("solimp", "0.99 0.999 0.001")
                body.append(collision)
                skin_collision_count += 1

    return body_names, skin_collision_count


def _append_native_hand(
    root: ET.Element,
    panda_body: ET.Element,
    *,
    side: str,
    mount_quat: str,
    material_names: dict[str, str],
    hand_density: float,
) -> tuple[set[str], int, list[ET.Element]]:
    definition_path, body_path = _native_hand_paths(side)
    definition_root = ET.parse(definition_path).getroot()
    body_root = ET.parse(body_path).getroot()

    asset = root.find("asset")
    source_asset = definition_root.find("asset")
    if asset is None or source_asset is None:
        raise ValueError(f"Native {side} hand MJCF is missing its asset section")
    for mesh in source_asset.findall("mesh"):
        clone = copy.deepcopy(mesh)
        file_name = clone.get("file")
        if file_name:
            path = Path(file_name)
            if not path.is_absolute():
                path = definition_path.parent / path
            clone.set("file", str(path.resolve()))
        clone.set("scale", "0.001 0.001 0.001")
        asset.append(clone)

    tower = body_root.find("body")
    if tower is None or tower.get("name") != f"{side}_tower":
        raise ValueError(f"Native {side} hand body does not contain {side}_tower")
    tower = copy.deepcopy(tower)
    tower.set("pos", "0 0 0.0575")
    tower.set("quat", mount_quat)
    body_names, skin_collisions = _materialize_native_body_defaults(
        tower,
        material_names=material_names,
        hand_density=hand_density,
    )

    attachment = next(
        (
            body
            for body in panda_body.iter("body")
            if body.get("name") == f"{side}_panda_link8"
        ),
        None,
    )
    if attachment is None:
        raise ValueError("Panda source MJCF is missing its attachment body")
    attachment.append(tower)

    actuator_root = definition_root.find("actuator")
    if actuator_root is None:
        raise ValueError(f"Native {side} hand MJCF is missing its actuators")
    hand_actuators: list[ET.Element] = []
    for actuator in actuator_root:
        clone = copy.deepcopy(actuator)
        clone.attrib.setdefault("ctrllimited", "true")
        clone.attrib.setdefault("forcelimited", "true")
        clone.attrib.setdefault("forcerange", "-1 1")
        clone.attrib.setdefault("kp", "2.0")
        hand_actuators.append(clone)
    return body_names, skin_collisions, hand_actuators


def _clone_panda_actuators(
    source_actuators: list[ET.Element], side: str
) -> list[ET.Element]:
    actuators: list[ET.Element] = []
    for source in source_actuators:
        joint = source.get("joint", "")
        match = re.fullmatch(r"joint([1-7])", joint)
        if match is None:
            continue
        clone = copy.deepcopy(source)
        joint_name = f"{side}_panda_joint{match.group(1)}"
        clone.set("joint", joint_name)
        clone.set("name", f"act_{joint_name}")
        actuators.append(clone)
    if len(actuators) != 7:
        raise ValueError("Panda source MJCF does not expose seven arm actuators")
    return actuators


def prepare_bimanual_orcapanda_mjcf(
    source_path: str | Path,
    output_path: str | Path,
    *,
    hand_density: float = DEFAULT_HAND_DENSITY,
    swap_arm_mounts: bool = True,
) -> Path:
    """Compose a Genesis-ready bimanual robot from native MJCF sources.

    ``source_path`` must be the no-hand Panda MJCF.  The released v2 hand MJCF
    components are loaded from this package and attached directly to two cloned
    Panda arms.  The hand skin visual meshes also receive invisible coincident
    collision geoms so contacts occur at the rendered surface.
    """
    if hand_density <= 0:
        raise ValueError("hand_density must be positive")

    source = Path(source_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source MJCF not found: {source}")

    tree = ET.parse(source)
    root = tree.getroot()
    if root.tag != "mujoco":
        raise ValueError(f"Expected a MuJoCo MJCF root, got {root.tag!r}")
    root.set("model", "native_bimanual_orcapanda")

    worldbody = root.find("worldbody")
    source_body = root.find("./worldbody/body[@name='link0']")
    actuator_root = root.find("actuator")
    asset = root.find("asset")
    if (
        worldbody is None
        or source_body is None
        or actuator_root is None
        or asset is None
    ):
        raise ValueError("Expected the no-hand Panda MJCF body, assets, and actuators")
    source_actuators = [copy.deepcopy(actuator) for actuator in actuator_root]
    worldbody.remove(source_body)
    actuator_root.clear()

    for removable in ("contact", "keyframe"):
        element = root.find(removable)
        if element is not None:
            root.remove(element)

    material_names = _append_native_materials(root)
    native_hand_body_names: set[str] = set()
    skin_collision_count = 0

    for spec in _HAND_SPECS:
        side = spec["side"]
        panda_body = _namespace_panda_subtree(
            source_body,
            side=side,
            base_pos=spec["base_pos"],
        )
        body_names, side_skin_collisions, hand_actuators = _append_native_hand(
            root,
            panda_body,
            side=side,
            mount_quat=spec["mount_quat"],
            material_names=material_names,
            hand_density=hand_density,
        )
        native_hand_body_names.update(body_names)
        skin_collision_count += side_skin_collisions
        worldbody.append(panda_body)
        actuator_root.extend(_clone_panda_actuators(source_actuators, side))
        actuator_root.extend(hand_actuators)

    if swap_arm_mounts:
        left_root = root.find(f"./worldbody/body[@name='{LEFT_ARM_ROOT}']")
        right_root = root.find(f"./worldbody/body[@name='{RIGHT_ARM_ROOT}']")
        if left_root is None or right_root is None:
            raise ValueError("The composed MJCF is missing a Panda arm root")
        left_position = left_root.get("pos", "0 0 0")
        right_position = right_root.get("pos", "0 0 0")
        left_root.set("pos", right_position)
        right_root.set("pos", left_position)

    compiler = root.find("compiler")
    mesh_dir = source.parent
    if compiler is not None and compiler.get("meshdir"):
        configured = Path(compiler.get("meshdir", "."))
        mesh_dir = (
            configured if configured.is_absolute() else source.parent / configured
        )

    for mesh in root.findall(".//asset/mesh"):
        file_name = mesh.get("file")
        if file_name:
            path = Path(file_name)
            if not path.is_absolute():
                path = mesh_dir / path
                # The vendored no-hand Panda retains Menagerie's historical
                # ``meshdir=assets`` although its mesh files sit beside the
                # MJCF inside the installed package.
                if not path.exists():
                    sibling_path = source.parent / file_name
                    if sibling_path.exists():
                        path = sibling_path
            mesh.set("file", str(path.resolve()))

    # Keep authored Panda inertias and use scene-wide collision groups for its
    # collision meshes. Native hand properties were made explicit above.
    panda_collision_count = 0
    native_collision_count = 0
    for body in root.findall(".//body"):
        is_native_hand = body.get("name", "") in native_hand_body_names
        for geom in body.findall("geom"):
            is_visual = geom.get("class", "").endswith("/visual") or (
                geom.get("contype") == "0" and geom.get("conaffinity") == "0"
            )
            if is_visual:
                continue
            if is_native_hand:
                native_collision_count += 1
            else:
                geom.set("contype", "5")
                geom.set("conaffinity", "5")
                panda_collision_count += 1

    if (
        len(native_hand_body_names) == 0
        or native_collision_count == 0
        or skin_collision_count == 0
        or panda_collision_count == 0
    ):
        raise ValueError("Failed to compose the expected native ORCA-hand layout")

    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def _fingerprint(paths: list[Path], hand_density: float) -> str:
    digest = hashlib.sha256(f"{hand_density:g}:native-v2-v2".encode())
    for path in paths:
        stat = path.stat()
        digest.update(f"{path}:{stat.st_mtime_ns}:{stat.st_size}".encode())
    return digest.hexdigest()[:16]


def build_genesis_bimanual_orcapanda_mjcf(
    *,
    hand_density: float = DEFAULT_HAND_DENSITY,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return a cached native-MJCF bimanual ORCA-Panda asset."""
    source = resolve_panda_nohand_mjcf()
    native_paths = [source, _NATIVE_HAND_OPTIONS]
    for side in ("left", "right"):
        native_paths.extend(_native_hand_paths(side))
    fingerprint = _fingerprint(native_paths, hand_density)
    root = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else Path(tempfile.gettempdir()) / "orca_sim" / "genesis"
    )
    destination = root / f"native-bimanual-orcapanda-{fingerprint}.xml"
    if destination.is_file():
        return destination
    return prepare_bimanual_orcapanda_mjcf(
        source,
        destination,
        hand_density=hand_density,
    )
