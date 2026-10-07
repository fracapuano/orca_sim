import numpy as np
import pytest
from types import SimpleNamespace

from orca_sim.bimanual_stacking import STACK_LAYOUT, StackContactAudit, level_rotation
from orca_sim.mujoco_bimanual import (
    OrcaBimanualPandaCubeStackingMujoco,
    tabletop_configuration,
)
from test_bimanual_grasp import contact_fixture


def test_left_thumb_opposition_is_delayed_without_changing_final_targets():
    from orca_sim.bimanual_stacking import BimanualStack

    controller = BimanualStack.__new__(BimanualStack)
    controller.env = SimpleNamespace(
        data=SimpleNamespace(time=13),
        robots={"left": SimpleNamespace(actuator_indices=np.arange(24, 48))},
    )
    controller.phase, controller.start_time = "LEFT PICK BLUE", 0
    controller.action = np.full(48, 0.2)
    controller.controller = SimpleNamespace(
        palm=SimpleNamespace(close_time=11),
        joint_index={"t-abd": 20, "t-cmc": 21},
        command=lambda _: (np.full(48, 0.4), "WRAP FINGERS"),
    )
    action, _ = controller.command()
    np.testing.assert_array_equal(action[:24], np.full(24, 0.2))
    assert action[44] > 0.8 and action[45] < -0.4
    assert action[43] == action[46] == 0.4
    controller.env.data.time = 20
    action, _ = controller.command()
    assert action[44] == 0 and action[45] == 0.55


def test_planning_does_not_mutate_live_state_and_unlifted_placement_is_rejected():
    from orca_sim.bimanual_stacking import BimanualStack, PlaceCube

    e = OrcaBimanualPandaCubeStackingMujoco(
        object_layout=STACK_LAYOUT, target_marker="x"
    )
    try:
        before = [v.copy() for v in (e.data.qpos, e.data.qvel, e.data.ctrl)]
        controller = BimanualStack(e)
        controller.command()
        for expected, actual in zip(before, (e.data.qpos, e.data.qvel, e.data.ctrl)):
            np.testing.assert_array_equal(expected, actual)
        with pytest.raises(RuntimeError, match="not lifted"):
            PlaceCube(e, "left", "blue_cube", [0.5, 0.201])
    finally:
        e.close()


@pytest.mark.parametrize("phase", ["RIGHT PICK RED", "LEFT PICK BLUE"])
def test_failed_pick_cannot_advance_to_placement(phase):
    from orca_sim.bimanual_stacking import BimanualStack

    c = BimanualStack.__new__(BimanualStack)
    c.env = SimpleNamespace(
        data=SimpleNamespace(
            time=23, body=lambda _: SimpleNamespace(xpos=np.array([0.3, 0.2, 0.353]))
        ),
        robots={
            s: SimpleNamespace(actuator_indices=np.arange(i, i + 24))
            for s, i in (("right", 0), ("left", 24))
        },
    )
    c.phase, c.start_time, c.action = phase, 0, np.zeros(48)
    c.controller = SimpleNamespace(
        command=lambda _: (np.zeros(48), "HOLD"),
        palm=SimpleNamespace(close_time=None),
        audit=SimpleNamespace(inspect=lambda: dict(hand_contacts=0)),
    )
    with pytest.raises(RuntimeError, match="grasp/lift not established"):
        c.command()


def test_demo_layout_and_visual_x_do_not_overwrite_saved_defaults():
    saved = tabletop_configuration()
    e = OrcaBimanualPandaCubeStackingMujoco(
        object_layout=STACK_LAYOUT, target_marker="x"
    )
    try:
        np.testing.assert_allclose(
            e.data.body("stack_target").xpos, [0.5, 0.201, 0.326]
        )
        np.testing.assert_allclose(e.data.body("red_cube").xpos, [0.3, 0.201, 0.353])
        np.testing.assert_allclose(e.data.body("blue_cube").xpos, [0.7, 0.201, 0.353])
        assert e.model.ncam == 3 and e.model.nu == 48 and e.model.neq == 0
        for name in ("stack_target_x_a", "stack_target_x_b"):
            geom = e.model.geom(name).id
            assert e.model.geom_contype[geom] == e.model.geom_conaffinity[geom] == 0
        np.testing.assert_array_equal(e.data.qpos, e.model.key("orcapanda_home").qpos)
    finally:
        e.close()
    assert tabletop_configuration() == saved


@pytest.mark.parametrize("angle", [-0.3, 0.0, 0.2, 1.4])
def test_face_levelling_preserves_yaw_and_proper_rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    tilt = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    c, s = np.cos(0.6), np.sin(0.6)
    yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rotation = level_rotation(yaw @ tilt)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1)
    assert max(np.abs(rotation[2])) == pytest.approx(1)
    if abs(angle) < 0.5:
        np.testing.assert_allclose(rotation, yaw, atol=1e-12)


@pytest.mark.parametrize(
    "first,second,forbidden",
    [
        ("right_finger", "red_cube", False),
        ("left_finger", "blue_cube", False),
        ("red_cube", "blue_cube", False),
        ("blue_cube", "table", False),
        ("right_finger", "blue_cube", True),
        ("left_finger", "red_cube", True),
        ("right_finger", "left_finger", True),
        ("left_finger", "table", True),
        ("left_panda_link6", "blue_cube", True),
        ("right_thumb", "right_finger", True),
    ],
)
def test_stacking_audit_allows_only_assigned_grasps_and_object_support(
    first, second, forbidden
):
    env = contact_fixture(first, second)
    assert bool(StackContactAudit(env).inspect()["forbidden"]) == forbidden


@pytest.mark.slow
def test_continuous_bimanual_pick_place_stack(tmp_path):
    from orca_sim.demos.bimanual_stack import run

    report = run(tmp_path)
    assert report["success"], report
    assert report["forbidden_contact_count"] == 0
    assert report["stable_stack_samples"] >= 499
    assert report["validation_settle_seconds"] >= 5
    assert report["base_target_error_m"] < 0.012
    assert report["max_lift_m"]["red_cube"] > 0.15
    assert report["max_lift_m"]["blue_cube"] > 0.15
    assert report["external_forces_zero"] and report["equalities"] == 0
