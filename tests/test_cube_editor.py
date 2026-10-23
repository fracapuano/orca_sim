import copy
import json

import mujoco
import numpy as np
import pytest

from orca_sim import mujoco_panda
from orca_sim.mujoco_panda import OrcaPandaCubeStackingMujoco, start_configuration
from orca_sim.pose_editor import PoseEditor


@pytest.fixture
def cube_editor(tmp_path, monkeypatch):
    home = start_configuration()
    home.update(base_pos=[0.0, -0.224, 0.325], base_yaw=np.pi / 2)
    home_file = tmp_path / "home.json"
    home_file.write_text(json.dumps(home))
    layout = {
        "red_cube": [0.16, 0.3, 0.353],
        "blue_cube": [0, 0.3, 0.353],
        "stack_target": [0, 0.3, 0.326],
    }
    layout_file = tmp_path / "source_layout.json"
    layout_file.write_text(json.dumps(layout))
    monkeypatch.setattr(mujoco_panda, "HOME_POSE", home_file)
    monkeypatch.setattr(mujoco_panda, "TABLETOP_LAYOUT", layout_file)
    editor = PoseEditor(tmp_path / "edited_layout.json", mode="objects")
    yield editor
    editor.close()


def test_cube_controls_are_robot_relative(cube_editor):
    editor = cube_editor
    e = editor.env
    before_qpos, before_ctrl = e.data.qpos.copy(), e.data.ctrl.copy()
    table = e.data.geom("tabletop").xpos.copy()
    robot = e.data.body("link0").xpos.copy()
    before_red = e.data.body("red_cube").xpos.copy()
    before_blue = e.data.body("blue_cube").xpos.copy()
    payload = editor.state()
    np.testing.assert_allclose(payload["objects_robot"]["red_cube"], [0.524, -0.16])
    payload["objects_robot"]["red_cube"][0] -= 0.1
    payload["objects_robot"]["red_cube"][1] -= 0.04
    editor.update(payload)
    np.testing.assert_allclose(
        e.data.body("red_cube").xpos, before_red + [0.04, -0.1, 0]
    )
    np.testing.assert_allclose(e.data.body("blue_cube").xpos, before_blue, atol=1e-15)
    np.testing.assert_array_equal(e.data.qpos[e.qadr], before_qpos[e.qadr])
    np.testing.assert_array_equal(e.data.ctrl, before_ctrl)
    np.testing.assert_array_equal(e.data.body("link0").xpos, robot)
    np.testing.assert_array_equal(e.data.geom("tabletop").xpos, table)
    assert e.data.time == 0 and not np.any(e.data.qvel)
    assert not np.any(e.data.xfrc_applied) and not np.any(e.data.qfrc_applied)


def test_blue_slider_moves_target_but_not_red(cube_editor):
    payload = cube_editor.state()
    red = cube_editor.env.data.body("red_cube").xpos.copy()
    payload["objects_robot"]["blue_cube"] = [0.42, 0.08]
    cube_editor.update(payload)
    d = cube_editor.env.data
    np.testing.assert_allclose(d.body("blue_cube").xpos, [-0.08, 0.196, 0.353])
    np.testing.assert_allclose(d.body("stack_target").xpos, [-0.08, 0.196, 0.326])
    np.testing.assert_allclose(d.body("red_cube").xpos, red, atol=1e-15)


def test_save_cube_layout_roundtrip_and_backup_without_home_edit(
    cube_editor, monkeypatch
):
    home_before = mujoco_panda.HOME_POSE.read_bytes()
    payload = cube_editor.state()
    payload["objects_robot"]["red_cube"] = [0.42, -0.16]
    payload["objects_robot"]["blue_cube"] = [0.46, 0.02]
    saved = cube_editor.save(payload)
    assert saved["saved"] == str(cube_editor.save_path)
    content = json.loads(cube_editor.save_path.read_text())
    assert set(content) == {"red_cube", "blue_cube", "stack_target"}
    assert mujoco_panda.HOME_POSE.read_bytes() == home_before
    monkeypatch.setattr(mujoco_panda, "TABLETOP_LAYOUT", cube_editor.save_path)
    env = OrcaPandaCubeStackingMujoco()
    try:
        for name, position in content.items():
            np.testing.assert_allclose(env.data.body(name).xpos, position)
        env.step(env.data.ctrl)
        env.reset()
        for name, position in content.items():
            np.testing.assert_allclose(env.data.body(name).xpos, position)
        model = mujoco.MjModel.from_xml_string(env.xml)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("orcapanda_home").id)
        mujoco.mj_forward(model, data)
        for name, position in content.items():
            np.testing.assert_allclose(data.body(name).xpos, position)
        assert env.start_pose == cube_editor.config
    finally:
        env.close()
    restored = PoseEditor(cube_editor.save_path, mode="objects")
    try:
        assert restored.object_layout == content
    finally:
        restored.close()
    cube_editor.save(payload)
    backups = list(cube_editor.save_path.parent.glob("edited_layout.*.bak"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text()) == content


@pytest.mark.parametrize(
    "coords", [[0.4], [0.4, 0, 0], [float("nan"), 0], [0, float("inf")], [10, 0]]
)
def test_invalid_cube_position_does_not_mutate_preview(cube_editor, coords):
    original = cube_editor.env.data.qpos.copy()
    payload = cube_editor.state()
    payload["objects_robot"]["red_cube"] = coords
    with pytest.raises(ValueError):
        cube_editor.update(payload)
    np.testing.assert_array_equal(cube_editor.env.data.qpos, original)


def test_robot_pose_locked_in_cube_mode(cube_editor):
    initial = cube_editor.state()
    for key in ("arm_qpos", "hand_qpos", "base_pos", "base_yaw"):
        payload = copy.deepcopy(initial)
        if key == "hand_qpos":
            payload[key]["right_wrist"] += 0.01
        elif key == "base_yaw":
            payload[key] += 0.01
        else:
            payload[key][0] += 0.01
        with pytest.raises(ValueError, match="locked"):
            cube_editor.update(payload)


def test_cube_overlap_is_previewed_and_blocks_save(cube_editor):
    payload = cube_editor.state()
    payload["objects_robot"]["red_cube"] = payload["objects_robot"]["blue_cube"].copy()
    result = cube_editor.update(payload)
    assert any(
        set(hit["bodies"]) == {"red_cube", "blue_cube"} for hit in result["collisions"]
    )
    with pytest.raises(ValueError, match="intersecting"):
        cube_editor.save(payload)
    assert not cube_editor.save_path.exists()


def test_cube_robot_collision_blocks_save(cube_editor):
    payload = cube_editor.state()
    payload["objects_robot"]["red_cube"] = [0, 0]
    result = cube_editor.update(payload)
    assert result["collisions"]
    with pytest.raises(ValueError, match="intersecting"):
        cube_editor.save(payload)


def test_robot_relative_coordinates_with_arbitrary_yaw(
    cube_editor, tmp_path, monkeypatch
):
    home = start_configuration()
    home.update(base_pos=[0, 0, 0.325], base_yaw=0.37)
    mujoco_panda.HOME_POSE.write_text(json.dumps(home))  # Test-only home file.
    editor = PoseEditor(tmp_path / "rotated_objects.json", mode="objects")
    try:
        payload = editor.state()
        payload["objects_robot"]["red_cube"] = [0.32, -0.1]
        editor.update(payload)
        c, s = np.cos(0.37), np.sin(0.37)
        expected = np.array([[c, -s], [s, c]]) @ [0.32, -0.1]
        np.testing.assert_allclose(editor.env.data.body("red_cube").xpos[:2], expected)
        np.testing.assert_allclose(
            editor.state()["objects_robot"]["red_cube"], [0.32, -0.1]
        )
    finally:
        editor.close()


def test_configuration_rejects_changed_object_height(cube_editor):
    bad = copy.deepcopy(cube_editor.object_layout)
    bad["red_cube"][2] += 0.1
    with pytest.raises(ValueError, match="tabletop"):
        OrcaPandaCubeStackingMujoco(object_layout=bad)
