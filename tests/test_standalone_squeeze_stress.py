"""Contact and bounded-squeeze regressions; raw curl is not a force command."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from scripts.standalone_hand_grasp import scene_xml, targets
from scripts.standalone_squeeze_stress import (
    pose_delta,
    relative_pose,
    run,
    squeeze_target,
)
from orca_sim.mujoco_grasp import GraspPreload


@pytest.mark.parametrize("side", ["left", "right"])
def test_flexion_stress_preserves_opposition_and_other_joint_targets(side):
    model = mujoco.MjModel.from_xml_string(scene_xml(side))
    original_gains = model.actuator_gainprm.copy()
    original_limits = model.actuator_forcerange.copy()
    baseline = targets(model, side, 0.6)
    stronger = squeeze_target(model, side, "flexion", 1.0)
    changed = []
    for i, j in enumerate(model.actuator_trnid[:, 0]):
        name = model.joint(int(j)).name
        if name.endswith(("-mcp", "-pip")):
            assert stronger[i] > baseline[i]
            changed.append(name)
        else:
            assert stronger[i] == baseline[i]
    assert len(changed) == 10
    np.testing.assert_array_equal(model.actuator_gainprm, original_gains)
    np.testing.assert_array_equal(model.actuator_forcerange, original_limits)
    np.testing.assert_array_equal(
        squeeze_target(model, side, "baseline", 1.0), baseline
    )


def test_relative_pose_ignores_rigid_motion_of_whole_hand_and_cube():
    rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    p = np.array([[1.0, 2, 3], [1.1, 2.2, 3.3]])
    data = SimpleNamespace(xpos=p, xmat=np.array([np.eye(3), rotation]).reshape(2, 9))
    before = relative_pose(data, 1, 0)
    transform = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    data.xpos = p @ transform.T + [3, -2, 1]
    data.xmat = np.array([transform, transform @ rotation]).reshape(2, 9)
    after = relative_pose(data, 1, 0)
    np.testing.assert_allclose(after[0], before[0])
    np.testing.assert_allclose(after[1], before[1])
    distance, angle = pose_delta(*after, before)
    assert distance == pytest.approx(0, abs=1e-12)
    assert angle == pytest.approx(0)


def test_rotation_is_not_hidden_by_cube_symmetry_or_zero_translation():
    quarter_turn = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    distance, angle = pose_delta(np.zeros(3), quarter_turn, (np.zeros(3), np.eye(3)))
    assert distance == 0
    assert angle == pytest.approx(90)


@pytest.mark.parametrize("side", ["left", "right"])
def test_stronger_squeeze_retains_cube_without_significant_motion(side):
    # A squeeze scales the current preload, rather than commanding a different
    # finger shape. Keep the original 2 mm / 5 degree acceptance thresholds.
    result = run(side=side, profile="preload", multiplier=4)
    assert result["retained_at_end"], result
    assert result["passes_motion_check"], result
    assert 1.1 < result["applied_preload_multiplier"] < 4
    assert (
        result["end_ramp_total_normal_N"]
        > 1.2 * result["pre_squeeze_mean_total_normal_N"]
    )
    assert result["max_skin_vertex_depth_entire_rollout_m"] < 0.0001
    assert result["max_raw_contact_depth_entire_rollout_m"] < 0.0001
    assert result["max_raw_contact_depth_every_physics_step_m"] < 0.0001
    assert result["max_abs_motor_torque_every_physics_step_Nm"] <= 1
    assert result["external_force_arrays_zero"]
    assert result["equality_constraints"] == 0


@pytest.mark.parametrize("side", ["left", "right"])
def test_contacts_alone_stop_raw_curl_penetrating_and_dropping_cube(side):
    result = run(side=side, profile="flexion")
    assert result["retained_at_end"], result
    assert not any(result["warnings"])
    assert result["max_skin_vertex_depth_entire_rollout_m"] < 0.0001
    assert result["max_raw_contact_depth_entire_rollout_m"] < 0.0001
    # Do not claim that moving the fingers cannot move an object.
    assert not result["passes_motion_check"]


def test_legacy_soft_contacts_reproduce_failure():
    result = run(profile="flexion", contact_model="legacy")
    assert not result["retained_at_end"]
    assert result["max_skin_vertex_depth_m"] > 0.003


@pytest.mark.parametrize("side", ["left", "right"])
def test_hand_contact_defaults_are_scoped_and_preserve_mechanics(side):
    native = mujoco.MjModel.from_xml_string(scene_xml(side))
    legacy = mujoco.MjModel.from_xml_string(scene_xml(side, contact_model="legacy"))
    hand = [
        i
        for i in range(native.ngeom)
        if native.body(int(native.geom_bodyid[i])).name.startswith(side + "_")
    ]
    np.testing.assert_allclose(
        native.geom_solref[hand], np.tile([0.004, 1], (len(hand), 1))
    )
    np.testing.assert_allclose(
        native.geom_solimp[hand, :3], np.tile([0.999, 0.9999, 0.001], (len(hand), 1))
    )
    np.testing.assert_array_equal(native.geom_priority[hand], 1)
    assert native.geom_priority[native.geom("cube_geom").id] == 0
    assert native.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert native.opt.impratio == 10
    assert native.opt.timestep == legacy.opt.timestep == 0.002
    for field in (
        "body_mass",
        "body_inertia",
        "mesh_vert",
        "mesh_face",
        "geom_friction",
        "geom_size",
        "dof_damping",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
        "actuator_forcerange",
    ):
        np.testing.assert_array_equal(getattr(native, field), getattr(legacy, field))


@pytest.mark.parametrize("side", ["left", "right"])
def test_preload_uses_common_saturation_and_does_not_modify_state(side):
    model = mujoco.MjModel.from_xml_string(scene_xml(side))
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.1
    for i, j in enumerate(model.actuator_trnid[:, 0]):
        if model.joint(int(j)).name.endswith("-abd"):
            data.ctrl[i] = 0.01
    qpos, qvel, ctrl = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
    gains, limits = model.actuator_gainprm.copy(), model.actuator_forcerange.copy()
    control = GraspPreload(model, data, side)
    command = control.command(100)
    assert control.applied_multiplier == pytest.approx(4.75)
    np.testing.assert_allclose(command[control.indices], ctrl[control.indices] * 4.75)
    assert command[0] == 0.1  # native wrist not scaled
    for value, original in (
        (data.qpos, qpos),
        (data.qvel, qvel),
        (data.ctrl, ctrl),
        (model.actuator_gainprm, gains),
        (model.actuator_forcerange, limits),
    ):
        np.testing.assert_array_equal(value, original)
    np.testing.assert_allclose(control.command(1), ctrl)
    with pytest.raises(ValueError):
        control.command(float("nan"))
    with pytest.raises(ValueError):
        control.command(0.5)


def test_preload_honors_position_limit_before_individual_clipping():
    model = mujoco.MjModel.from_xml_string(scene_xml())
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.1
    actuator = model.actuator("right_t-cmc_actuator").id
    joint = model.joint("right_t-cmc").id
    data.qpos[model.jnt_qposadr[joint]] = 0.54
    data.ctrl[actuator] = 0.55
    control = GraspPreload(model, data, "right")
    command = control.command(100)
    assert command[actuator] == pytest.approx(model.actuator_ctrlrange[actuator, 1])
    assert control.applied_multiplier == pytest.approx(
        (model.actuator_ctrlrange[actuator, 1] - 0.54) / 0.01
    )


def test_preload_rejects_no_load_or_already_saturated_grasp():
    model = mujoco.MjModel.from_xml_string(scene_xml())
    data = mujoco.MjData(model)
    with pytest.raises(ValueError, match="No existing preload"):
        GraspPreload(model, data, "right")
    data.ctrl[:] = 1
    with pytest.raises(ValueError, match="headroom"):
        GraspPreload(model, data, "right")


@pytest.mark.parametrize(
    "kwargs", [{"fraction": 1.1}, {"duration": 6}, {"variant": "unknown"}]
)
def test_invalid_stress_arguments_rejected(kwargs):
    with pytest.raises(ValueError):
        run(**kwargs)
