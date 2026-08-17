from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from orca_sim.genesis_assets import (
    prepare_bimanual_orcapanda_mjcf,
    resolve_panda_nohand_mjcf,
)
from orca_sim.genesis_env import (
    CUBE_CONTACT_SKIN,
    CUBE_MJCF_PATH,
    CUBE_POSITION,
    TABLE_POSITION,
    TABLE_SIZE,
    WORKBENCH_MJCF_PATH,
)


def test_prepare_mjcf_composes_native_v2_hands_with_panda(
    tmp_path: Path,
) -> None:
    output = prepare_bimanual_orcapanda_mjcf(
        resolve_panda_nohand_mjcf(),
        tmp_path / "generated" / "genesis.xml",
        hand_density=725.0,
    )
    root = ET.parse(output).getroot()

    assert all(
        Path(mesh.get("file", "")).is_absolute()
        for mesh in root.findall(".//asset/mesh")
    )
    assert root.find(".//body[@name='left_panda_link1']/inertial") is not None
    panda_collision = root.find(
        ".//body[@name='left_panda_link1']/geom[@class='panda/collision']"
    )
    assert panda_collision is not None
    assert panda_collision.get("contype") == "5"
    assert panda_collision.get("conaffinity") == "5"

    left_root = root.find(".//body[@name='left_panda_link0']")
    right_root = root.find(".//body[@name='right_panda_link0']")
    assert left_root is not None and left_root.get("pos") == "0 -0.45 0"
    assert right_root is not None and right_root.get("pos") == "0 0.45 0"
    assert root.find("contact") is None

    assert root.find(".//body[@name='orcahand_right_palm']") is None
    hand = root.find(".//body[@name='right_I-FingerTipAssembly_ec49c16c']")
    assert hand is not None
    assert hand.find("inertial") is None
    skin = hand.find("geom[@mesh='right_I-FingerTipAssembly_I-DP-Skin']")
    assert skin is not None
    assert skin.get("density") == "0"
    assert skin.get("contype") == "0"
    collision = next(
        (
            geom
            for geom in hand.findall("geom")
            if geom.get("mesh") == skin.get("mesh") and geom.get("contype") == "6"
        ),
        None,
    )
    assert collision is not None
    assert collision.get("mesh") == skin.get("mesh")
    assert collision.get("density") == "725"
    assert collision.get("contype") == "6"
    assert collision.get("conaffinity") == "6"
    assert collision.get("solref") == "0.001 1"
    assert collision.get("solimp") == "0.99 0.999 0.001"

    model = mujoco.MjModel.from_xml_path(str(output))
    assert model.nq == 48
    assert model.nu == 48
    joint_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
    }
    assert "left_wrist" in joint_names
    assert "right_t-cmc" in joint_names
    assert not any(name.startswith("orcahand_") for name in joint_names)


def test_prepare_mjcf_rejects_non_panda_asset(tmp_path: Path) -> None:
    source = tmp_path / "source.xml"
    source.write_text("<mujoco><worldbody/></mujoco>", encoding="utf-8")
    with pytest.raises(ValueError, match="no-hand Panda MJCF"):
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


def test_cube_asset_uses_exact_support_and_submillimetre_hand_skin() -> None:
    model = mujoco.MjModel.from_xml_path(str(CUBE_MJCF_PATH))
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "grasp_cube")
    np.testing.assert_allclose(model.body_pos[body_id], CUBE_POSITION)

    visual_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "grasp_cube_visual")
    support_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "grasp_cube_support_collision"
    )
    hand_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "grasp_cube_hand_collision"
    )
    np.testing.assert_allclose(model.geom_size[visual_id], [0.025] * 3)
    np.testing.assert_allclose(model.geom_size[support_id], [0.025] * 3)
    np.testing.assert_allclose(
        model.geom_size[hand_id],
        [
            0.025 + CUBE_CONTACT_SKIN,
            0.025 + CUBE_CONTACT_SKIN,
            0.025 + CUBE_CONTACT_SKIN / 2,
        ],
    )
    assert model.geom_pos[hand_id, 2] == pytest.approx(CUBE_CONTACT_SKIN / 2)
    assert model.geom_contype[visual_id] == 0
    assert model.geom_contype[support_id] == 1
    assert model.geom_contype[hand_id] == 2
    assert model.geom_conaffinity[hand_id] == 2
    assert model.ngeom == 3
