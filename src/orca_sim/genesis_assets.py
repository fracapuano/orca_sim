"""Genesis-specific preparation of the upstream bimanual ORCA-Panda asset."""

from __future__ import annotations

import hashlib
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_HAND_DENSITY = 650.0
HAND_BODY_PREFIX = "orcahand_"
LEFT_ARM_ROOT = "left_panda_link0"
RIGHT_ARM_ROOT = "right_panda_link0"


def resolve_bimanual_orcapanda_mjcf() -> Path:
    """Return the MJCF shipped by ``fracapuano/orca_arm``."""
    try:
        from orca_arm import BIMANUAL_ORCAPANDA_MJCF_PATH
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "The Genesis environment needs the bimanual assets from orca_arm. "
            "Install this project with `pip install -e '.[genesis]'`."
        ) from exc

    source = Path(BIMANUAL_ORCAPANDA_MJCF_PATH).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Bimanual ORCA-Panda MJCF not found: {source}")
    return source


def prepare_bimanual_orcapanda_mjcf(
    source_path: str | Path,
    output_path: str | Path,
    *,
    hand_density: float = DEFAULT_HAND_DENSITY,
    swap_arm_mounts: bool = True,
) -> Path:
    """Prepare the upstream MJCF for physically consistent Genesis import.

    The upstream Panda inertias are already authored in SI units and are preserved.
    The ORCA-hand inertias came from millimetre-scale CAD exports and are not
    consistent with the metre-scale collision meshes.  Genesis rightfully reports
    this mismatch.  Removing only the hand inertials lets the simulator integrate
    mass and inertia from the collision geometry at a documented material density.

    Mesh paths are made absolute because the generated XML lives in a cache rather
    than beside the source assets.
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

    if swap_arm_mounts:
        left_root = root.find(f"./worldbody/body[@name='{LEFT_ARM_ROOT}']")
        right_root = root.find(f"./worldbody/body[@name='{RIGHT_ARM_ROOT}']")
        if (left_root is None) != (right_root is None):
            raise ValueError("The MJCF contains only one expected Panda arm root")
        if left_root is not None and right_root is not None:
            left_position = left_root.get("pos", "0 0 0")
            right_position = right_root.get("pos", "0 0 0")
            left_root.set("pos", right_position)
            right_root.set("pos", left_position)

    # Genesis rewrites every MJCF collision bitmask when an explicit MuJoCo
    # <exclude> is present.  The source only excludes each fixed base from its
    # adjacent first link, which Genesis already handles with adjacent-pair
    # filtering.  Removing it preserves the scene-wide contact groups below.
    contact = root.find("contact")
    if contact is not None:
        root.remove(contact)

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
            mesh.set("file", str(path.resolve()))

    hand_bodies = 0
    removed_inertials = 0
    collision_geoms = 0
    density = f"{hand_density:g}"
    for body in root.findall(".//body"):
        is_hand = body.get("name", "").startswith(HAND_BODY_PREFIX)
        if is_hand:
            hand_bodies += 1

        inertial = body.find("inertial")
        if is_hand and inertial is not None:
            body.remove(inertial)
            removed_inertials += 1

        for geom in body.findall("geom"):
            is_visual = geom.get("class", "").endswith("/visual") or geom.get(
                "density"
            ) == "0" or (
                geom.get("contype") == "0" and geom.get("conaffinity") == "0"
            )
            if is_visual:
                continue
            if is_hand:
                geom.attrib.pop("mass", None)
                geom.set("density", density)
                # Bit 2: manipulation-object contact skins. Bit 4: robot/table
                # and robot self-collision. The skin is only 0.5 mm and exists
                # to absorb the solver's finite contact tolerance.
                geom.set("contype", "6")
                geom.set("conaffinity", "6")
                geom.set("solref", "0.001 1")
                geom.set("solimp", "0.99 0.999 0.001")
                collision_geoms += 1
            else:
                # Bit 1: exact scene/object collision. Bit 4: robot/table and
                # robot self-collision. Panda links do not touch contact skins.
                geom.set("contype", "5")
                geom.set("conaffinity", "5")

    if hand_bodies == 0 or removed_inertials == 0 or collision_geoms == 0:
        raise ValueError(
            "The MJCF does not contain the expected ORCA-hand body/inertial/geom layout"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def build_genesis_bimanual_orcapanda_mjcf(
    *,
    hand_density: float = DEFAULT_HAND_DENSITY,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return a cached Genesis-ready bimanual ORCA-Panda MJCF path."""
    source = resolve_bimanual_orcapanda_mjcf()
    stat = source.stat()
    fingerprint = hashlib.sha256(
        f"{source}:{stat.st_mtime_ns}:{stat.st_size}:{hand_density:g}:v11".encode()
    ).hexdigest()[:16]
    root = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else Path(tempfile.gettempdir()) / "orca_sim" / "genesis"
    )
    destination = root / f"bimanual_orcapanda-{fingerprint}.xml"
    if destination.is_file():
        return destination
    return prepare_bimanual_orcapanda_mjcf(
        source,
        destination,
        hand_density=hand_density,
    )
