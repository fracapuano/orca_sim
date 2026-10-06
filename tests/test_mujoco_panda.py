from __future__ import annotations

import json

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
            np.testing.assert_allclose(half, [0.035] * 3)
            mass = model.body(name).mass[0]
            assert mass == pytest.approx(0.06)
            np.testing.assert_allclose(
                model.body(name).inertia, [mass * (2 * half[0]) ** 2 / 6] * 3
            )


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
