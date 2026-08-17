from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from orca_sim.genesis_assets import prepare_bimanual_orcapanda_mjcf
from orca_sim.genesis_env import (
    CUBE_COLLISION_PADDING,
    CUBE_MJCF_PATH,
    CUBE_POSITION,
    TABLE_POSITION,
    TABLE_SIZE,
    WORKBENCH_MJCF_PATH,
)


def test_prepare_mjcf_preserves_panda_and_recomputes_hand_inertias(
    tmp_path: Path,
) -> None:
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    (mesh_dir / "part.stl").write_bytes(b"mesh placeholder")
    source = tmp_path / "source.xml"
    source.write_text(
        """\
<mujoco model="test">
  <compiler meshdir="meshes"/>
  <asset><mesh name="part" file="part.stl"/></asset>
  <worldbody>
    <body name="left_panda_link0" pos="0 0.45 0"/>
    <body name="right_panda_link0" pos="0 -0.45 0"/>
    <body name="panda_link">
      <inertial mass="2" pos="0 0 0" diaginertia="1 1 1"/>
      <geom mesh="part" class="panda/visual"/>
      <geom mesh="part" class="panda/collision"/>
    </body>
    <body name="orcahand_right_palm">
      <inertial mass="8" pos="0 0 0" diaginertia="100 100 100"/>
      <geom mesh="part" density="0" contype="0" conaffinity="0"/>
      <geom mesh="part" mass="8"/>
    </body>
  </worldbody>
  <contact><exclude body1="left_panda_link0" body2="panda_link"/></contact>
</mujoco>
""",
        encoding="utf-8",
    )

    output = prepare_bimanual_orcapanda_mjcf(
        source,
        tmp_path / "generated" / "genesis.xml",
        hand_density=725.0,
    )
    root = ET.parse(output).getroot()

    mesh = root.find(".//asset/mesh")
    assert mesh is not None
    assert Path(mesh.get("file", "")).is_absolute()
    assert root.find(".//body[@name='panda_link']/inertial") is not None
    panda_visual, panda_collision = root.findall(
        ".//body[@name='panda_link']/geom"
    )
    assert panda_visual.get("contype") is None
    assert panda_collision.get("contype") == "5"
    assert panda_collision.get("conaffinity") == "5"

    left_root = root.find(".//body[@name='left_panda_link0']")
    right_root = root.find(".//body[@name='right_panda_link0']")
    assert left_root is not None and left_root.get("pos") == "0 -0.45 0"
    assert right_root is not None and right_root.get("pos") == "0 0.45 0"
    assert root.find("contact") is None

    hand = root.find(".//body[@name='orcahand_right_palm']")
    assert hand is not None
    assert hand.find("inertial") is None
    visual, collision = hand.findall("geom")
    assert visual.get("density") == "0"
    assert collision.get("density") == "725"
    assert "mass" not in collision.attrib
    assert collision.get("contype") == "6"
    assert collision.get("conaffinity") == "6"


def test_prepare_mjcf_rejects_non_orca_asset(tmp_path: Path) -> None:
    source = tmp_path / "source.xml"
    source.write_text("<mujoco><worldbody/></mujoco>", encoding="utf-8")
    with pytest.raises(ValueError, match="expected ORCA-hand"):
        prepare_bimanual_orcapanda_mjcf(source, tmp_path / "output.xml")


def test_workbench_asset_is_grounded_and_preserves_task_surface() -> None:
    model = mujoco.MjModel.from_xml_path(str(WORKBENCH_MJCF_PATH))
    workbench_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "workbench")

    np.testing.assert_allclose(model.body_pos[workbench_id], [TABLE_POSITION[0], 0, 0])
    np.testing.assert_allclose(TABLE_SIZE, [0.75, 1.15, 0.05])
    assert TABLE_POSITION[2] + TABLE_SIZE[2] / 2 == pytest.approx(0.30)

    surface_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "work_surface")
    visual_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "work_surface_visual"
    )
    np.testing.assert_allclose(model.geom_size[surface_id], TABLE_SIZE / 2)
    np.testing.assert_allclose(model.geom_size[visual_id], TABLE_SIZE / 2)
    assert model.geom_pos[surface_id, 2] == pytest.approx(TABLE_POSITION[2])
    assert model.geom_pos[visual_id, 2] == pytest.approx(TABLE_POSITION[2])
    assert model.geom_contype[surface_id] == 5
    assert model.geom_conaffinity[surface_id] == 5
    assert model.geom_contype[visual_id] == 0
    assert model.geom_conaffinity[visual_id] == 0
    np.testing.assert_allclose(
        model.mat_rgba[model.geom_matid[visual_id]], [0.38, 0.20, 0.09, 1]
    )

    for name in ("foot_xn_yn", "foot_xn_yp", "foot_xp_yn", "foot_xp_yp"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        bottom_z = model.geom_pos[geom_id, 2] - model.geom_size[geom_id, 2]
        assert bottom_z == pytest.approx(0.0)
        assert model.geom_contype[geom_id] == 0


def test_cube_asset_separates_support_and_hand_contact_margin() -> None:
    model = mujoco.MjModel.from_xml_path(str(CUBE_MJCF_PATH))
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "grasp_cube")
    np.testing.assert_allclose(model.body_pos[body_id], CUBE_POSITION)

    visual_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "grasp_cube_visual"
    )
    support_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "grasp_cube_support_collision"
    )
    margin_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "grasp_cube_hand_collision"
    )
    np.testing.assert_allclose(model.geom_size[visual_id], [0.025] * 3)
    np.testing.assert_allclose(model.geom_size[support_id], [0.025] * 3)
    np.testing.assert_allclose(model.geom_size[margin_id], [0.035, 0.035, 0.025])
    assert model.geom_pos[margin_id, 2] == pytest.approx(CUBE_COLLISION_PADDING)
    assert model.geom_contype[visual_id] == 0
    assert model.geom_contype[support_id] == 1
    assert model.geom_contype[margin_id] == 2
