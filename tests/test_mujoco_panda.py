from __future__ import annotations

import sys
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from orca_sim.mujoco_panda import (
    PACKAGE,
    TABLETOP_LAYOUT,
    OrcaPandaCubeStackingMujoco,
    inverse_kinematics,
)


@pytest.fixture
def env():
    e = OrcaPandaCubeStackingMujoco()
    yield e
    e.close()


def test_native_hand_parameters_preserved(env):
    source = mujoco.MjModel.from_xml_path(str(PACKAGE / "scenes/v2/scene_right.xml"))
    combined = env.model
    assert combined.nu == 24
    assert combined.nq == 38  # 24 hinges + two free cubes.
    assert combined.neq == 0
    assert combined.opt.timestep == 0.001
    assert combined.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert combined.opt.impratio == 10

    def excluded_pairs(model):
        return {
            (model.body(int(sig) >> 16).name, model.body(int(sig) & 65535).name)
            for sig in model.exclude_signature
        }

    assert excluded_pairs(source) <= excluded_pairs(combined)
    for i in range(source.nu):
        joint_name = source.joint(source.actuator_trnid[i, 0]).name
        j = env.actuator_joints.index(joint_name)
        for field in (
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_forcerange",
            "actuator_ctrlrange",
        ):
            np.testing.assert_array_equal(
                getattr(source, field)[i], getattr(combined, field)[j]
            )
    for i in range(1, source.nbody):
        name = source.body(i).name
        if not name.startswith("right_") or name == "right_mount":
            continue
        j = combined.body(name).id
        np.testing.assert_allclose(
            source.body_mass[i], combined.body_mass[j], atol=1e-12
        )
        np.testing.assert_allclose(
            source.body_inertia[i], combined.body_inertia[j], atol=1e-12
        )
    source_geoms = [
        i
        for i in range(source.ngeom)
        if source.body(source.geom_bodyid[i]).name.startswith("right_")
    ]
    dest_geoms = [
        i
        for i in range(combined.ngeom)
        if combined.body(combined.geom_bodyid[i]).name.startswith("right_")
    ]
    for field in (
        "geom_solref",
        "geom_solimp",
        "geom_priority",
        "geom_friction",
        "geom_contype",
        "geom_conaffinity",
        "geom_margin",
    ):
        np.testing.assert_allclose(
            getattr(source, field)[source_geoms], getattr(combined, field)[dest_geoms]
        )


def test_original_stacking_scene_objects_preserved(env):
    source = mujoco.MjModel.from_xml_path(str(PACKAGE / "scenes/cube_stacking.xml"))
    for name in ("tabletop", "red_cube_geom", "blue_cube_geom"):
        i, j = source.geom(name).id, env.model.geom(name).id
        for field in (
            "geom_size",
            "geom_friction",
            "geom_condim",
            "geom_solref",
            "geom_solimp",
            "geom_contype",
            "geom_conaffinity",
        ):
            if name == "tabletop" and field == "geom_size":
                continue  # Shared bench is intentionally enlarged.
            np.testing.assert_array_equal(
                getattr(source, field)[i], getattr(env.model, field)[j]
            )
    for model in (source, env.model):
        for name in ("red_cube", "blue_cube"):
            half = model.geom(name + "_geom").size
            np.testing.assert_allclose(half, [0.035 * 0.8] * 3)
            mass = model.body(name).mass[0]
            assert mass == pytest.approx(0.06 * 0.8**3)
            np.testing.assert_allclose(
                model.body(name).inertia, [mass * (2 * half[0]) ** 2 / 6] * 3
            )


def test_smaller_cubes_start_on_table_and_lift_reward_uses_new_height(env):
    m, d = env.model, env.data
    top = d.geom("tabletop").xpos[2] + m.geom("tabletop").size[2]
    for name in ("red_cube", "blue_cube"):
        assert d.body(name).xpos[2] == pytest.approx(0.353)
        assert d.body(name).xpos[2] - m.geom(name + "_geom").size[2] == pytest.approx(
            top
        )
    _, reward, _, _, info = env.step(d.ctrl.copy())
    assert reward == pytest.approx(0, abs=1e-5)
    assert info["lift_height_m"] == pytest.approx(0, abs=1e-5)


def test_shared_table_and_user_chosen_home(env):
    m, d = env.model, env.data
    task_top = d.geom("tabletop").xpos[2] + m.geom("tabletop").size[2]
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot_table") == -1
    assert d.body("link0").xpos[2] == pytest.approx(task_top)
    np.testing.assert_array_equal(m.geom("tabletop").size, [0.55, 0.40, 0.025])
    for body in ("link0", "red_cube", "blue_cube"):
        assert np.all(
            np.abs(d.body(body).xpos[:2] - d.geom("tabletop").xpos[:2])
            < m.geom("tabletop").size[:2]
        )
    np.testing.assert_array_equal(
        d.qpos[env.arm_qadr],
        [
            0.0,
            -0.5236162310840257,
            0.0,
            -3.0142041346841824,
            -1.5289618664364453,
            1.467775193447175,
            0.9790762686794061,
        ],
    )
    assert d.body("right_tower").xpos[2] > task_top + 0.20
    q = d.qpos.copy()
    env.step(d.ctrl + 0.01)
    env.reset()
    np.testing.assert_array_equal(d.qpos, q)


def test_ik_is_accurate_and_does_not_teleport(env):
    before = env.data.qpos.copy()
    target = env.data.site("hand_frame").xpos + [0.015, 0, 0.03]
    rotation = env.data.site("hand_frame").xmat.reshape(3, 3).copy()
    q = inverse_kinematics(env, target, rotation=rotation)
    np.testing.assert_array_equal(env.data.qpos, before)
    scratch = mujoco.MjData(env.model)
    scratch.qpos[:] = before
    scratch.qpos[env.arm_qadr] = q
    mujoco.mj_forward(env.model, scratch)
    np.testing.assert_allclose(scratch.site("hand_frame").xpos, target, atol=1e-5)
    np.testing.assert_allclose(
        scratch.site("hand_frame").xmat.reshape(3, 3), rotation, atol=1e-4
    )


def test_sideways_approach_keeps_home_orientation_and_joint_branch(env):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from mujoco_panda_grasp import translation_approach

    before = env.data.qpos.copy()
    start = env.data.site("hand_frame").xpos.copy()
    rotation = env.data.site("hand_frame").xmat.reshape(3, 3).copy()
    destination = start + [0.06, 0.10, -0.02]
    path = translation_approach(env, destination, rotation)
    np.testing.assert_array_equal(env.data.qpos, before)
    np.testing.assert_array_equal(path[0], before[env.arm_qadr])
    assert np.max(np.abs(np.diff(path, axis=0))) < 0.05
    scratch = mujoco.MjData(env.model)
    scratch.qpos[:] = before
    for amount, q in zip(np.linspace(0, 1, len(path)), path):
        scratch.qpos[env.arm_qadr] = q
        mujoco.mj_forward(env.model, scratch)
        np.testing.assert_allclose(
            scratch.site("hand_frame").xpos,
            start * (1 - amount) + destination * amount,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            scratch.site("hand_frame").xmat.reshape(3, 3), rotation, atol=1e-4
        )


@pytest.mark.parametrize("options", [{"yaw_degrees": 0}, {"horizontal": False}])
def test_sideways_rejects_conflicting_orientation_overrides(tmp_path, options):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from mujoco_panda_grasp import run

    with pytest.raises(ValueError, match="saved home orientation"):
        run(output_dir=tmp_path, sideways=True, **options)


def test_saved_workspace_layout_survives_reset_and_mjcf_export(env):
    layout = json.loads(TABLETOP_LAYOUT.read_text())
    for name, pos in layout.items():
        np.testing.assert_allclose(env.data.body(name).xpos, pos)
    np.testing.assert_allclose(layout["stack_target"][:2], layout["blue_cube"][:2])
    model = mujoco.MjModel.from_xml_string(env.xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("orcapanda_home").id)
    mujoco.mj_forward(model, data)
    for name, pos in layout.items():
        np.testing.assert_allclose(data.body(name).xpos, pos)
    env.step(env.data.ctrl)
    env.reset()
    for name, pos in layout.items():
        np.testing.assert_allclose(env.data.body(name).xpos, pos)


def test_arbitrary_position_actions_step_physics(env):
    before = env.data.time
    action = env.data.ctrl.copy()
    action[0] += 0.02
    obs, _, terminated, truncated, info = env.step(action)
    assert env.data.time > before
    np.testing.assert_array_equal(env.data.ctrl, action)
    assert np.isfinite(obs).all() and not terminated and not truncated
    assert "lift_height_m" in info
    assert not np.any(env.data.xfrc_applied)
    assert not np.any(env.data.qfrc_applied)


@pytest.mark.parametrize("action", [np.zeros(17), np.full(24, np.nan)])
def test_invalid_actions_rejected(env, action):
    with pytest.raises(ValueError):
        env.step(action)


@pytest.mark.slow
def test_approach_grasp_lift(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from mujoco_panda_grasp import run

    result = run(output_dir=tmp_path)
    assert result["success"]
    assert result["minimum_hold_lift_m"] > 0.1
    assert result["max_hand_torque_Nm"] <= 1.0 + 1e-10
    assert result["equalities"] == 0 and result["external_forces_zero"]
    assert not result["warnings"]


@pytest.mark.slow
def test_stack_release_and_retreat(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from mujoco_panda_grasp import run

    result = run(output_dir=tmp_path, stack=True)
    assert result["success"] and result["stack_success"]
    assert result["final_stack"]["stack_contacts"] >= 1
    assert result["final_stack"]["stack_height_m"] == pytest.approx(0.056, abs=0.005)
    assert result["equalities"] == 0 and result["external_forces_zero"]


@pytest.mark.slow
def test_sideways_home_orientation_grasp_and_stack(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from mujoco_panda_grasp import run

    result = run(output_dir=tmp_path, sideways=True, stack=True)
    assert result["success"] and result["grasp_success"] and result["stack_success"]
    assert result["max_approach_hand_frame_rotation_deg"] < 1.0
    assert result["placement_orientation_mode"] == "keep_grasp_orientation"
    assert result["minimum_hold_lift_m"] > 0.1
    assert result["equalities"] == 0 and result["external_forces_zero"]
    assert not result["warnings"]


@pytest.mark.slow
def test_palm_contact_precedes_wrap_lift_and_stack(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from mujoco_panda_grasp import run

    result = run(output_dir=tmp_path, palm_first=True, stack=True)
    assert result["success"] and result["stack_success"]
    assert result["palm_before_closure_verified"]
    assert result["palm_rest_duration_s"] >= 0.6
    assert (
        result["finger_closure_start_time_s"]
        >= result["palm_first_contact_time_s"] + 0.6
    )
    assert result["minimum_hold_lift_m"] > 0.15
    assert result["max_hand_torque_Nm"] <= 1.0 + 1e-10
    assert result["equalities"] == 0 and result["external_forces_zero"]
    assert not result["warnings"]
    trace = json.loads((tmp_path / "trace.json").read_text())
    closed = result["finger_closure_start_time_s"]
    env = OrcaPandaCubeStackingMujoco()
    try:
        bends = [
            i
            for i, name in enumerate(env.actuator_joints)
            if name.endswith(
                ("i-mcp", "i-pip", "m-mcp", "m-pip", "r-mcp", "r-pip", "p-mcp", "p-pip")
            )
        ]
        for sample in trace:
            if 5 < sample["t"] < closed:
                np.testing.assert_allclose(
                    np.array(sample["ctrl"])[bends], 0, atol=1e-12
                )
        assert all(
            set(s["digits"]) == {"i", "m", "r", "p", "t"}
            for s in trace
            if s["phase"] == "HOLD" and s["t"] > closed + 8
        )
    finally:
        env.close()
