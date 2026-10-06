"""Native MuJoCo release smoke tests: no Genesis, arm, or grasp attachment.

These are bounded grasp checks, not certification for arbitrary objects. The
rendered-mesh vertex probe is a lower bound on visual interpenetration.
"""

import mujoco
import numpy as np
import pytest

from scripts.standalone_hand_grasp import (
    FIXTURE_OFFSETS,
    MAX_CONTACT_DEPTH,
    MAX_SKIN_VERTEX_DEPTH,
    default_cube_position,
    run,
    scene_xml,
    skin_geoms,
    skin_vertex_depth,
    targets,
)


@pytest.mark.parametrize("side", ["left", "right"])
def test_skin_fix_changes_only_collision_masks(side):
    fixed = mujoco.MjModel.from_xml_string(scene_xml(side))
    original = mujoco.MjModel.from_xml_string(scene_xml(side, variant="skin_disabled"))
    skins = skin_geoms(fixed)
    assert len(skins) == 10
    assert fixed.ngeom == original.ngeom
    assert fixed.nq == original.nq == 24  # 17 hand joints + 7 cube coordinates
    assert fixed.nu == original.nu == 17
    np.testing.assert_array_equal(fixed.geom_contype[skins], 1)
    np.testing.assert_array_equal(fixed.geom_conaffinity[skins], 1)
    np.testing.assert_array_equal(original.geom_contype[skins], 0)
    np.testing.assert_array_equal(original.geom_conaffinity[skins], 0)
    others = np.setdiff1d(np.arange(fixed.ngeom), skins)
    for field in ("geom_contype", "geom_conaffinity"):
        np.testing.assert_array_equal(
            getattr(fixed, field)[others], getattr(original, field)[others]
        )
    for field in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "body_pos",
        "body_quat",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "geom_margin",
        "geom_gap",
        "geom_dataid",
        "geom_rgba",
        "mesh_vert",
        "mesh_face",
        "jnt_range",
        "jnt_pos",
        "jnt_axis",
        "dof_damping",
        "dof_armature",
        "dof_frictionloss",
        "qpos0",
        "actuator_trnid",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "actuator_ctrlrange",
    ):
        np.testing.assert_array_equal(
            getattr(fixed, field), getattr(original, field), err_msg=field
        )
    assert fixed.opt.timestep == original.opt.timestep == 0.002
    np.testing.assert_array_equal(fixed.opt.gravity, [0, 0, -9.81])
    np.testing.assert_array_equal(fixed.actuator_gainprm[:, 0], 2)
    np.testing.assert_array_equal(fixed.actuator_forcerange, np.tile([-1, 1], (17, 1)))


@pytest.mark.parametrize("side", ["left", "right"])
def test_control_targets_use_actuator_joint_names(side):
    model = mujoco.MjModel.from_xml_string(scene_xml(side))
    control = targets(model, side, 0.6)
    expected = {"wrist": 0, "t-cmc": 0.33, "t-abd": 0, "t-mcp": 0.18, "t-pip": 0.48}
    for digit in "imrp":
        expected.update({f"{digit}-abd": 0, f"{digit}-mcp": 0.87, f"{digit}-pip": 0.87})
    for actuator in range(model.nu):
        joint = model.joint(int(model.actuator_trnid[actuator, 0])).name
        assert control[actuator] == pytest.approx(
            expected[joint.removeprefix(side + "_")]
        )


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("mass", [0.04, 0.08, 0.12])
def test_hand_holds_free_cube_without_deep_skin_penetration(side, mass):
    result = run(side=side, cube_mass=mass)
    assert result["initial_skin_vertex_depth_m"] == 0
    assert result["initial_collision_depth_m"] == 0
    assert result["skin_collision_enabled"]
    assert not any(result["warnings"])
    assert result["held_after_support_removal"], result
    assert result["max_skin_vertex_depth_m"] <= MAX_SKIN_VERTEX_DEPTH, result
    assert result["max_collision_depth_m"] <= MAX_CONTACT_DEPTH, result
    assert result["passes_grasp_smoke_test"], result


@pytest.mark.parametrize("side", ["left", "right"])
def test_original_masks_reproduce_visible_penetration(side):
    result = run(side=side, variant="skin_disabled", contact_model="legacy")
    # Holding alone would miss the original defect: internal bone meshes hold
    # the cube while its surface passes through the white, non-colliding skin.
    assert result["held_after_support_removal"], result
    assert not result["skin_collision_enabled"]
    assert result["max_skin_vertex_depth_m"] > 0.003, result
    assert not result["passes_grasp_smoke_test"]


@pytest.mark.parametrize("side", ["left", "right"])
def test_open_hand_does_not_hold_cube(side):
    result = run(side=side, variant="open_control")
    assert not result["held_after_support_removal"]
    assert not result["passes_grasp_smoke_test"]
    assert result["cube_drop_m"] > 0.20
    assert result["final_contacts"] == 0
    assert not any(result["warnings"])


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("offset", FIXTURE_OFFSETS)
def test_nearby_collision_free_cube_positions(side, offset):
    position = np.asarray(default_cube_position(side)) + np.asarray(offset) * [
        -1 if side == "left" else 1,
        1,
        1,
    ]
    result = run(side=side, cube_pos=position)
    assert result["passes_grasp_smoke_test"], result


def test_vertex_probe_detects_mesh_inside_cube():
    model = mujoco.MjModel.from_xml_string(scene_xml())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cube = model.geom("cube_geom").id
    skins = skin_geoms(model)
    assert skin_vertex_depth(model, data, cube, skins) == 0
    # Move only the probe cube frame to one known world-space skin vertex.
    geom = skins[0]
    vertex = model.mesh_vert[model.mesh_vertadr[model.geom_dataid[geom]]]
    data.geom_xpos[cube] = (
        vertex @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    )
    assert skin_vertex_depth(model, data, cube, skins) == pytest.approx(0.025)


@pytest.mark.parametrize("side", ["left", "right"])
def test_intersecting_initial_fixture_is_rejected(side):
    position = list(default_cube_position(side))
    position[0] += -0.003 if side == "right" else 0.003
    with pytest.raises(ValueError, match="starts intersecting"):
        run(side=side, cube_pos=position)


@pytest.mark.parametrize(
    "kwargs", [{"hold_seconds": 0}, {"fraction": 1.1}, {"cube_mass": -1}]
)
def test_invalid_experiment_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        run(**kwargs)
