import copy

import gymnasium as gym
import mujoco
import numpy as np
import pytest

from orca_sim import register_envs
from orca_sim import mujoco_bimanual as bim
from orca_sim import mujoco_panda as single
from orca_sim.contact_diagnostics import skin_geoms


@pytest.fixture
def env():
    e = bim.OrcaBimanualPandaCubeStackingMujoco()
    yield e
    e.close()


def test_named_48_target_assembly_and_unchanged_right_home(env):
    original = single.OrcaPandaCubeStackingMujoco()
    try:
        assert (env.model.nq, env.model.nv, env.model.nu) == (62, 60, 48)
        assert env.model.neq == 0
        assert len(set(env.actuator_joints)) == 48
        assert env.actuator_joints[0] == "right_panda_joint1"
        assert env.actuator_joints[24] == "left_panda_joint1"
        assert env.model.opt.timestep == 0.001
        assert env.model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
        assert env.model.opt.impratio == 10
        np.testing.assert_array_equal(env.data.ctrl[:24], original.data.ctrl)
        np.testing.assert_array_equal(
            env.data.site("right_hand_frame").xpos,
            original.data.site("hand_frame").xpos,
        )
        np.testing.assert_array_equal(
            env.data.site("right_hand_frame").xmat,
            original.data.site("hand_frame").xmat,
        )
        for name in ("red_cube", "blue_cube", "stack_target"):
            np.testing.assert_array_equal(
                env.data.body(name).xpos, original.data.body(name).xpos
            )
        for side in bim.SIDES:
            base = env.data.body(side + "_panda_link0").xpos
            assert base[2] == pytest.approx(0.325)
            assert env.data.ctrl[env.robots[side].actuator_indices[7:]].shape == (17,)
        np.testing.assert_allclose(env.model.geom("tabletop").size, [0.9, 0.4, 0.025])
        for name in ("table_leg_front_left", "table_leg_back_right"):
            assert env.data.geom(name).xpos[2] - env.model.geom(name).size[
                1
            ] == pytest.approx(0)
    finally:
        original.close()


def test_homes_face_one_another_with_correct_handed_assets(env):
    reflection = np.diag([-1, 1, 1])
    right, left = [env.data.site(s + "_hand_frame") for s in bim.SIDES]
    np.testing.assert_allclose(
        left.xpos, reflection @ right.xpos + [1, 0, 0], atol=1e-12
    )
    np.testing.assert_allclose(
        left.xmat.reshape(3, 3),
        reflection @ right.xmat.reshape(3, 3) @ reflection,
        atol=1e-12,
    )
    assert right.xmat.reshape(3, 3)[0, 1] > 0.99
    assert left.xmat.reshape(3, 3)[0, 1] < -0.99
    assert env.model.mesh("left_L-Carpals").id != env.model.mesh("right_R-Carpals").id
    assert env.contact_metrics() == dict(
        inter_robot_depth_m=0.0, inter_robot_contacts=0, robot_table_depth_m=0.0
    )


@pytest.mark.parametrize("side,hand", (("right", "left"), ("left", "right")))
def test_only_native_hands_exchange_mounts_and_ownership(env, side, hand):
    from orca_sim.demos.bimanual_grasp import ContactAudit

    m, d = env.model, env.data
    robot = env.robots[side]
    assert robot.hand_side == hand
    assert m.site(robot.hand_site).bodyid == m.body(hand + "_tower").id
    assert all(name.startswith(hand + "_") for name in robot.actuator_joints[7:])
    assert all(name.startswith(side + "_panda_") for name in robot.actuator_joints[:7])
    np.testing.assert_array_equal(
        d.qpos[robot.arm_qadr], env.start_pose[side]["arm_qpos"]
    )
    np.testing.assert_array_equal(
        d.body(side + "_panda_link0").xpos, env.start_pose[side]["base_pos"]
    )
    frame = d.site(robot.hand_site)
    thumb_side = frame.xmat.reshape(3, 3)[:, 0] * (1 if hand == "left" else -1)
    toward_base = d.body(side + "_panda_link0").xpos[:2] - frame.xpos[:2]
    toward_base /= np.linalg.norm(toward_base)
    assert thumb_side[:2] @ toward_base > 0.8
    audit = ContactAudit(env, side, "red_cube")
    assert audit.hand_side == hand
    assert audit.palm_mesh == hand + ("_L-Carpals" if hand == "left" else "_R-Carpals")
    for geom in skin_geoms(m):
        native_side = m.body(m.geom_bodyid[geom]).name.split("_")[0]
        owner = (
            side if native_side == hand else ("left" if side == "right" else "right")
        )
        assert env.geom_robots[geom] == owner
        assert audit.roles[geom] == owner + "_hand"


def test_saved_hand_targets_remain_with_their_arm_after_swap():
    home = bim.start_configuration()
    home["right"]["hand_qpos"]["right_i-mcp"] = 0.2
    home["left"]["hand_qpos"]["left_i-mcp"] = 0.4
    env = bim.OrcaBimanualPandaCubeStackingMujoco(start_pose=home)
    try:
        assert env.data.joint("left_i-mcp").qpos[0] == 0.2
        assert env.data.joint("right_i-mcp").qpos[0] == 0.4
    finally:
        env.close()


def test_swapped_hands_hold_saved_home_without_unintended_contacts():
    from orca_sim.demos.bimanual_grasp import ContactAudit

    env = bim.OrcaBimanualPandaCubeStackingMujoco(frame_skip=1)
    try:
        targets = env.data.ctrl.copy()
        audit = ContactAudit(env, "right", "red_cube")
        for _ in range(3000):
            env.step(targets)
            assert not audit.inspect()["forbidden"]
        assert not any(w.number for w in env.data.warning)
        assert not np.any(env.data.xfrc_applied)
        assert not np.any(env.data.qfrc_applied)
    finally:
        env.close()


@pytest.mark.parametrize("side", bim.SIDES)
def test_each_native_hand_preserves_mechanics_and_corrected_contacts(env, side):
    source = mujoco.MjModel.from_xml_path(
        str(single.PACKAGE / f"scenes/v2/scene_{side}.xml")
    )
    m = env.model
    ids = [
        env.actuator_joints.index(source.joint(int(j)).name)
        for j in source.actuator_trnid[:, 0]
    ]
    for field in (
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "actuator_ctrlrange",
    ):
        np.testing.assert_array_equal(getattr(source, field), getattr(m, field)[ids])
    source_geoms = [
        i
        for i in range(source.ngeom)
        if source.body(source.geom_bodyid[i]).name.startswith(side + "_")
    ]
    dest_geoms = [
        i
        for i in range(m.ngeom)
        if m.body(m.geom_bodyid[i]).name.startswith(side + "_")
        and not m.body(m.geom_bodyid[i]).name.startswith(side + "_panda_")
    ]
    assert len(source_geoms) == len(dest_geoms)
    for field in (
        "geom_solref",
        "geom_solimp",
        "geom_priority",
        "geom_friction",
        "geom_margin",
        "geom_contype",
        "geom_conaffinity",
        "geom_pos",
        "geom_quat",
    ):
        np.testing.assert_array_equal(
            getattr(source, field)[source_geoms], getattr(m, field)[dest_geoms]
        )
    for i in range(1, source.nbody):
        name = source.body(i).name
        if name.startswith(side + "_") and name != side + "_mount":
            for field in ("body_mass", "body_inertia"):
                np.testing.assert_array_equal(
                    getattr(source, field)[i], getattr(m, field)[m.body(name).id]
                )
    skins = [
        i for i in skin_geoms(m) if m.body(m.geom_bodyid[i]).name.startswith(side + "_")
    ]
    assert len(skins) == 10
    np.testing.assert_array_equal(m.geom_contype[skins], 1)
    np.testing.assert_array_equal(m.geom_conaffinity[skins], 1)


def test_panda_parameters_preserved_on_both_sides(env):
    original = single.OrcaPandaCubeStackingMujoco()
    try:
        for side in bim.SIDES:
            indices = env.robots[side].actuator_indices[:7]
            for field in (
                "actuator_gainprm",
                "actuator_biasprm",
                "actuator_forcerange",
                "actuator_ctrlrange",
            ):
                np.testing.assert_array_equal(
                    getattr(original.model, field)[:7],
                    getattr(env.model, field)[indices],
                )
    finally:
        original.close()


@pytest.mark.parametrize("side", bim.SIDES)
def test_side_ik_is_accurate_and_never_changes_live_state(env, side):
    qpos, qvel, ctrl = env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy()
    robot = env.robots[side]
    target = env.data.site(robot.hand_site).xpos + [0, 0.035, 0.025]
    rotation = env.data.site(robot.hand_site).xmat.reshape(3, 3).copy()
    q = env.solve_ik(side, target)
    np.testing.assert_array_equal(env.data.qpos, qpos)
    np.testing.assert_array_equal(env.data.qvel, qvel)
    np.testing.assert_array_equal(env.data.ctrl, ctrl)
    scratch = mujoco.MjData(env.model)
    scratch.qpos[:] = qpos
    scratch.qpos[robot.arm_qadr] = q
    mujoco.mj_forward(env.model, scratch)
    np.testing.assert_allclose(scratch.site(robot.hand_site).xpos, target, atol=1e-5)
    np.testing.assert_allclose(
        scratch.site(robot.hand_site).xmat.reshape(3, 3), rotation, atol=1e-4
    )


def test_independent_commands_and_reset_export(env):
    original_qpos = env.data.qpos.copy()
    original_ctrl = env.data.ctrl.copy()
    for side, other in (("right", "left"), ("left", "right")):
        target = env.data.ctrl[env.robots[side].actuator_indices].copy()
        target[0] += 0.01
        before = env.data.ctrl[env.robots[other].actuator_indices].copy()
        action = env.side_action(side, target)
        env.step(action)
        np.testing.assert_array_equal(
            env.data.ctrl[env.robots[other].actuator_indices], before
        )
        np.testing.assert_array_equal(
            env.data.ctrl[env.robots[side].actuator_indices], target
        )
    assert env.data.time == pytest.approx(0.01)
    assert not np.any(env.data.xfrc_applied) and not np.any(env.data.qfrc_applied)
    env.reset()
    np.testing.assert_array_equal(env.data.qpos, original_qpos)
    np.testing.assert_array_equal(env.data.ctrl, original_ctrl)
    m = mujoco.MjModel.from_xml_string(env.xml)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("orcapanda_home").id)
    np.testing.assert_array_equal(d.qpos, original_qpos)
    np.testing.assert_array_equal(d.ctrl, original_ctrl)


def test_inter_robot_collisions_are_real_not_filtered():
    bad = bim.start_configuration()
    bad["left"]["base_pos"][0] = 0.85
    m = mujoco.MjModel.from_xml_string(bim.scene_xml(start_pose=bad))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("orcapanda_home").id)
    mujoco.mj_forward(m, d)
    assert any(
        c.dist < -0.001
        and {m.body(m.geom_bodyid[g]).name.split("_")[0] for g in (c.geom1, c.geom2)}
        == {"left", "right"}
        for c in d.contact
    )
    with pytest.raises(ValueError, match="intersect"):
        bim.OrcaBimanualPandaCubeStackingMujoco(start_pose=bad)


@pytest.mark.parametrize("side", bim.SIDES)
@pytest.mark.parametrize(
    "field,value",
    [
        ("arm_qpos", [0] * 6),
        ("base_yaw", float("nan")),
        ("base_pos", [0, 0, 0]),
        ("hand_qpos", {}),
    ],
)
def test_invalid_homes_rejected(side, field, value):
    home = bim.start_configuration()
    home[side][field] = value
    with pytest.raises(ValueError):
        bim.OrcaBimanualPandaCubeStackingMujoco(start_pose=home)


@pytest.mark.parametrize("side", bim.SIDES)
def test_native_joint_limits_checked(side):
    home = bim.start_configuration()
    home[side]["arm_qpos"][0] = 100
    with pytest.raises(ValueError, match="joint limits"):
        bim.OrcaBimanualPandaCubeStackingMujoco(start_pose=home)


def test_invalid_actions_and_ik_do_not_mutate_state(env):
    before = env.data.qpos.copy()
    for action in (np.zeros(24), np.full(48, np.nan)):
        with pytest.raises(ValueError):
            env.step(action)
    with pytest.raises(ValueError):
        env.side_action("left", np.zeros(23))
    with pytest.raises(ValueError):
        env.solve_ik("left", [0, 0, 0], rotation=np.zeros((3, 3)))
    with pytest.raises(ValueError):
        env.solve_ik("middle", [0, 0, 0])
    np.testing.assert_array_equal(env.data.qpos, before)


def test_objects_can_use_wider_table_without_changing_single_scene():
    original = single.tabletop_configuration()
    layout = copy.deepcopy(original)
    layout["blue_cube"][0] = 1.15
    layout["stack_target"][0] = 1.15
    env = bim.OrcaBimanualPandaCubeStackingMujoco(object_layout=layout)
    try:
        assert env.data.body("blue_cube").xpos[0] == pytest.approx(1.15)
        assert single.tabletop_configuration() == original
    finally:
        env.close()


def test_gym_registration():
    register_envs()
    env = gym.make("OrcaBimanualPandaCubeStackingMujoco-v0")
    try:
        obs, _ = env.reset()
        assert obs.shape == (122,) and env.action_space.shape == (48,)
        env.step(env.unwrapped.data.ctrl.copy())
    finally:
        env.close()


def test_closing_live_home_viewer_counts_the_completed_step(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from orca_sim.demos.bimanual import run

    def fake_render(env):
        env._viewer = SimpleNamespace(is_running=lambda: False, close=lambda: None)

    monkeypatch.setattr(bim.OrcaBimanualPandaCubeStackingMujoco, "render", fake_render)
    report = run(output_dir=tmp_path, viewer=True, hold=True)
    assert report["success"]
    assert report["control_steps"] == 1
    assert report["duration_s"] == pytest.approx(0.005)


@pytest.mark.slow
def test_real_two_arm_motion_rollout(tmp_path):
    from orca_sim.demos.bimanual import run

    report = run(output_dir=tmp_path)
    assert report["success"]
    assert report["control_steps"] == 2400
    assert report["max_inter_robot_depth_m"] == 0
    assert report["max_robot_table_depth_m"] == 0
    assert report["max_joint_limit_violation_rad"] == 0
    assert all(v > 0.09 for v in report["max_hand_displacement_m"].values())
    assert report["external_forces_zero"] and report["equalities"] == 0
