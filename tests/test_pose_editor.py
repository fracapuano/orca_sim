from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest

from orca_sim import mujoco_panda
from orca_sim.mujoco_panda import OrcaPandaCubeStackingMujoco, start_configuration
from orca_sim.pose_editor import PoseEditor


@pytest.fixture(autouse=True)
def fixed_test_home(tmp_path, monkeypatch):
    # UI users can save a new base at any time; use a stable test-only fixture.
    config = start_configuration()
    config["base_pos"] = [-0.25, 0.2, 0.325]
    config["base_yaw"] = 0.0
    path = tmp_path / "fixture_home.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(mujoco_panda, "HOME_POSE", path)


@pytest.fixture
def editor(tmp_path):
    editor = PoseEditor(tmp_path / "start_pose.json", mode="hand")
    yield editor
    editor.close()


def test_hand_slider_changes_real_kinematics_without_moving_arm(editor):
    m, d = editor.env.model, editor.env.data
    before = d.qpos.copy()
    wrist_body = m.jnt_bodyid[m.joint("right_wrist").id]
    wrist_rotation = d.xmat[wrist_body].copy()
    config = start_configuration()
    config["hand_qpos"]["right_wrist"] = 0.1
    editor.update(config)
    assert np.linalg.norm(d.xmat[wrist_body] - wrist_rotation) > 0.01
    np.testing.assert_array_equal(
        d.qpos[editor.env.arm_qadr], before[editor.env.arm_qadr]
    )
    assert editor.env.data.time == 0
    np.testing.assert_allclose(editor.env.data.ctrl[:7], config["arm_qpos"])
    assert not np.any(editor.env.data.xfrc_applied)


def test_arm_and_base_changes_are_rejected(editor):
    for key, index in (("arm_qpos", 0), ("base_pos", 1)):
        config = start_configuration()
        config[key][index] += 0.01
        with pytest.raises(ValueError, match="locked"):
            editor.update(config)


def test_hand_edit_does_not_move_tables_or_cubes(editor):
    m, d = editor.env.model, editor.env.data
    original_cube = d.body("red_cube").xpos.copy()
    original_task_top = d.geom("tabletop").xpos.copy()
    config = start_configuration()
    config["hand_qpos"]["right_i-mcp"] = 0.2
    editor.update(config)
    np.testing.assert_allclose(d.body("link0").xpos, config["base_pos"])
    np.testing.assert_array_equal(d.body("red_cube").xpos, original_cube)
    np.testing.assert_array_equal(d.geom("tabletop").xpos, original_task_top)
    table = m.body("table").id
    for g in range(
        m.body_geomadr[table], m.body_geomadr[table] + m.body_geomnum[table]
    ):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            assert d.geom_xpos[g, 2] - m.geom_size[g, 1] == pytest.approx(0)


def test_save_load_and_reset_roundtrip_with_backup(editor):
    config = start_configuration()
    config["hand_qpos"]["right_i-mcp"] = 0.1
    config["hand_qpos"]["right_t-pip"] = 0.15
    saved = editor.save(config)
    assert saved["saved"] == str(editor.save_path)
    assert json.loads(editor.save_path.read_text()) == config
    e = OrcaPandaCubeStackingMujoco(start_pose=editor.save_path)
    try:
        expected = e.data.qpos.copy()
        np.testing.assert_allclose(e.data.qpos[e.arm_qadr], config["arm_qpos"])
        np.testing.assert_allclose(e.data.body("link0").xpos, config["base_pos"])
        for name, angle in config["hand_qpos"].items():
            assert e.data.qpos[e.model.jnt_qposadr[e.model.joint(name).id]] == angle
            assert e.data.ctrl[e.actuator_joints.index(name)] == angle
        e.step(e.data.ctrl)
        e.reset()
        np.testing.assert_array_equal(e.data.qpos, expected)
    finally:
        e.close()
    editor.save(start_configuration())
    backups = list(editor.save_path.parent.glob("*.bak"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text()) == config


@pytest.mark.parametrize(
    "patch",
    [
        {"version": 3},
        {"arm_qpos": [0] * 6},
        {"arm_qpos": [float("nan")] * 7},
        {"arm_qpos": [10] * 7},
        {"base_pos": [0, 0, 0.325]},
        {"base_pos": [-0.3, 0, -0.1]},
        {"camera": [0, 91, 1]},
        {"hand_qpos": {}},
        {"focus": "invalid"},
    ],
)
def test_invalid_slider_values_are_rejected(editor, patch):
    before = editor.env.data.qpos.copy()
    config = start_configuration()
    config.update(patch)
    with pytest.raises(ValueError):
        editor.update(config)
    np.testing.assert_array_equal(editor.env.data.qpos, before)


@pytest.mark.parametrize("angle", [float("nan"), float("inf"), 100.0])
def test_invalid_hand_angle_is_rejected(editor, angle):
    config = start_configuration()
    config["hand_qpos"]["right_wrist"] = angle
    with pytest.raises(ValueError):
        editor.update(config)


def test_named_hand_keyframe_order_matches_actual_joints(editor):
    # Every hand joint differs: a permutation bug cannot hide behind all zeros.
    config = start_configuration()
    for i, name in enumerate(editor.env.actuator_joints[7:]):
        config["hand_qpos"][name] = 0.005 * (i + 1)
    e = OrcaPandaCubeStackingMujoco(start_pose=config)
    try:
        for name, angle in config["hand_qpos"].items():
            assert e.data.qpos[e.model.jnt_qposadr[e.model.joint(name).id]] == angle
            assert e.data.ctrl[e.actuator_joints.index(name)] == angle
        xml_model = mujoco.MjModel.from_xml_string(e.xml)
        data = mujoco.MjData(xml_model)
        mujoco.mj_resetDataKeyframe(xml_model, data, xml_model.key("orcapanda_home").id)
        np.testing.assert_array_equal(data.qpos, e.data.qpos)
    finally:
        e.close()


def test_version_one_pose_loads_open_hand():
    config = start_configuration()
    config["version"] = 1
    del config["hand_qpos"]
    restored = start_configuration(config)
    assert restored["arm_qpos"] == config["arm_qpos"]
    assert set(restored["hand_qpos"].values()) == {0}


def test_saved_hand_becomes_default_without_explicit_start_pose(editor, monkeypatch):
    config = start_configuration()
    config["hand_qpos"]["right_i-mcp"] = 0.12
    editor.save(config)
    monkeypatch.setattr(mujoco_panda, "HOME_POSE", editor.save_path)
    e = OrcaPandaCubeStackingMujoco()
    try:
        assert e.start_pose == config
        assert e.data.ctrl[e.actuator_joints.index("right_i-mcp")] == 0.12
        np.testing.assert_array_equal(e.data.qpos[e.arm_qadr], editor.locked_arm)
    finally:
        e.close()


def test_intersection_prevents_save(editor, monkeypatch):
    original_state = editor.state

    def intersecting_state():
        return {
            **original_state(),
            "collisions": [{"bodies": ["link1", "table"], "depth_mm": 5}],
        }

    monkeypatch.setattr(editor, "state", intersecting_state)
    with pytest.raises(ValueError, match="intersecting"):
        editor.save(start_configuration())
    assert not editor.save_path.exists()


@pytest.fixture
def layout_editor(tmp_path):
    editor = PoseEditor(tmp_path / "layout.json")
    yield editor
    editor.close()


def test_layout_translation_preserves_joints_and_scene(layout_editor):
    e = layout_editor.env
    qpos, ctrl = e.data.qpos.copy(), e.data.ctrl.copy()
    hand = e.data.site("hand_frame").xpos.copy()
    cubes = [e.data.body(name).xpos.copy() for name in ("red_cube", "blue_cube")]
    table = e.data.geom("tabletop").xpos.copy()
    config = start_configuration()
    shift = np.array([0.03, -0.04, 0.0])
    config["base_pos"] = (np.array(config["base_pos"]) + shift).tolist()
    state = layout_editor.update(config)
    assert state["editor_mode"] == "layout"
    assert state["arm_home_locked"] and state["hand_home_locked"]
    np.testing.assert_allclose(e.data.site("hand_frame").xpos, hand + shift)
    np.testing.assert_array_equal(e.data.qpos, qpos)
    np.testing.assert_array_equal(e.data.ctrl, ctrl)
    np.testing.assert_array_equal(e.data.geom("tabletop").xpos, table)
    for name, pos in zip(("red_cube", "blue_cube"), cubes):
        np.testing.assert_array_equal(e.data.body(name).xpos, pos)
    assert e.data.time == 0
    assert not np.any(e.data.xfrc_applied)


@pytest.mark.parametrize("yaw", [-np.pi, -np.pi / 2, np.pi / 2, np.pi])
def test_layout_yaw_rotates_whole_robot_about_base(layout_editor, yaw):
    e = layout_editor.env
    before = e.data.xpos.copy()
    original_base = e.data.body("link0").xpos.copy()
    config = start_configuration()
    config["base_yaw"] = yaw
    layout_editor.update(config)
    rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    base_id = e.model.body("link0").id
    # The Panda and hand are the last subtree in the composed model.
    for body in range(base_id, e.model.nbody):
        expected = original_base + rotation @ (before[body] - original_base)
        np.testing.assert_allclose(e.data.xpos[body], expected, atol=1e-12)
    np.testing.assert_array_equal(e.data.xpos[:base_id], before[:base_id])


def test_saved_layout_matches_reloaded_physics_and_reset(layout_editor):
    config = start_configuration()
    config["base_pos"][0] += 0.015
    config["base_yaw"] = 0.2
    saved = layout_editor.save(config)
    assert saved["saved_base_pos"] == config["base_pos"]
    assert saved["saved_base_yaw"] == config["base_yaw"]
    assert json.loads(layout_editor.save_path.read_text()) == config
    live = layout_editor.env
    fresh = OrcaPandaCubeStackingMujoco(start_pose=layout_editor.save_path)
    try:
        for field in ("xpos", "xmat", "geom_xpos", "qpos", "ctrl"):
            np.testing.assert_allclose(
                getattr(live.data, field), getattr(fresh.data, field)
            )
        assert live.data.ncon == fresh.data.ncon
        for _ in range(10):
            live.step(live.data.ctrl.copy())
            fresh.step(fresh.data.ctrl.copy())
        np.testing.assert_allclose(live.data.qpos, fresh.data.qpos, atol=1e-12)
        fresh.reset()
        np.testing.assert_allclose(fresh.data.body("link0").xpos, config["base_pos"])
        np.testing.assert_allclose(
            fresh.data.body("link0").xquat, [np.cos(0.1), 0, 0, np.sin(0.1)]
        )
        np.testing.assert_array_equal(fresh.data.ctrl[:7], config["arm_qpos"])
        for name, angle in config["hand_qpos"].items():
            assert fresh.data.ctrl[fresh.actuator_joints.index(name)] == angle
    finally:
        fresh.close()
    layout_editor.save(config)
    assert len(list(layout_editor.save_path.parent.glob("*.bak"))) == 1


@pytest.mark.parametrize("yaw", [float("nan"), float("inf"), -3.2, 3.2, [0], None])
def test_invalid_yaw_is_rejected_without_mutation(layout_editor, yaw):
    e = layout_editor.env
    before = e.data.xpos.copy()
    config = start_configuration()
    config["base_yaw"] = yaw
    with pytest.raises(ValueError, match="yaw"):
        layout_editor.update(config)
    np.testing.assert_array_equal(e.data.xpos, before)


def test_layout_locks_arm_hand_and_mounting_height(layout_editor):
    before = layout_editor.env.data.xpos.copy()
    for key in ("arm_qpos", "hand_qpos", "base_pos"):
        config = start_configuration()
        if key == "hand_qpos":
            config[key]["right_wrist"] += 0.01
        else:
            config[key][2] += 0.01
        with pytest.raises(ValueError):
            layout_editor.update(config)
        np.testing.assert_array_equal(layout_editor.env.data.xpos, before)


def test_hand_editor_locks_yaw(editor):
    config = start_configuration()
    config["base_yaw"] = 0.1
    with pytest.raises(ValueError, match="locked"):
        editor.update(config)


def test_old_files_without_yaw_keep_zero_orientation(tmp_path, monkeypatch):
    old = start_configuration()
    old.pop("base_yaw")
    default_file = tmp_path / "default.json"
    default_file.write_text(json.dumps({**old, "base_yaw": 1.0}))
    monkeypatch.setattr(mujoco_panda, "HOME_POSE", default_file)
    assert start_configuration()["base_yaw"] == 1.0
    for version in (1, 2):
        assert start_configuration({**old, "version": version})["base_yaw"] == 0.0


@pytest.mark.parametrize(
    "base",
    [[-0.281, 0, 0.325], [0.581, 0, 0.325], [0, -0.281, 0.325], [0, 0.281, 0.325]],
)
def test_off_table_mounting_is_rejected(layout_editor, base):
    config = start_configuration()
    config["base_pos"] = base
    with pytest.raises(ValueError, match="tabletop"):
        layout_editor.update(config)


def test_actual_base_cube_collision_blocks_layout_save(layout_editor):
    config = start_configuration()
    cube = layout_editor.env.data.body("red_cube").xpos
    config["base_pos"] = [float(cube[0]), min(float(cube[1]), 0.27), 0.325]
    # Deliberately mount the base on the red cube. This checks real collision
    # structures after moving a static body, not just its rendered position.
    state = layout_editor.update(config)
    assert any(
        set(hit["bodies"]) == {"link0", "red_cube"} for hit in state["collisions"]
    )
    with pytest.raises(ValueError, match="intersecting"):
        layout_editor.save(config)
    assert not layout_editor.save_path.exists()
