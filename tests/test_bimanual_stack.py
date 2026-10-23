import numpy as np
import pytest
from types import SimpleNamespace

from orca_sim.bimanual_stacking import STACK_LAYOUT, level_rotation, robot_cube_contacts
from orca_sim.mujoco_bimanual import (
    OrcaBimanualPandaCubeStackingMujoco,
    tabletop_configuration,
)
from test_bimanual_grasp import contact_fixture


@pytest.mark.parametrize("side,offset", [("right", 0), ("left", 24)])
def test_pickup_action_uses_unmodified_native_grasp_targets(side, offset):
    from orca_sim.bimanual_stacking import BimanualStack

    controller = BimanualStack.__new__(BimanualStack)
    controller.env = SimpleNamespace(
        robots={side: SimpleNamespace(actuator_indices=np.arange(offset, offset + 24))},
    )
    picker = SimpleNamespace(
        command=lambda _: (np.arange(48, dtype=float), "WRAP FINGERS"),
    )
    controller.pickers = {side: picker}
    action, phase = controller.pickup_action(side, 13)
    np.testing.assert_array_equal(action, np.arange(offset, offset + 24))
    assert phase == "WRAP FINGERS"


def test_parallel_pickups_share_one_clock_and_merge_only_their_own_targets():
    from orca_sim.bimanual_stacking import BimanualStack

    c = BimanualStack.__new__(BimanualStack)
    c.env = SimpleNamespace(
        data=SimpleNamespace(time=2.5),
        robots={
            s: SimpleNamespace(actuator_indices=np.arange(i, i + 24))
            for s, i in (("right", 0), ("left", 24))
        },
    )
    c.phase, c.start_time, c.action = "BOTH PICK CUBES", 0.0, np.zeros(48)
    calls = []

    def command(side, age):
        calls.append((side, age))
        # The unused side deliberately contains invalid targets: those must
        # never overwrite the other hand's independent command.
        return (
            np.r_[np.full(24, 0.1), np.full(24, 999)]
            if side == "right"
            else np.r_[np.full(24, -999), np.full(24, 0.2)]
        ), "APPROACH"

    c.pickers = {
        side: SimpleNamespace(
            palm=SimpleNamespace(close_time=None),
            command=lambda age, side=side: command(side, age),
        )
        for side in ("right", "left")
    }
    action, phase = c.command()
    assert calls == [("right", 2.5), ("left", 2.5)]
    np.testing.assert_array_equal(action[:24], np.full(24, 0.1))
    np.testing.assert_array_equal(action[24:], np.full(24, 0.2))
    assert "RIGHT: APPROACH" in phase and "LEFT: APPROACH" in phase


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


@pytest.mark.parametrize("failed_side", ["right", "left"])
def test_either_failed_pick_blocks_both_arms_from_placement(failed_side):
    from orca_sim.bimanual_stacking import BimanualStack

    c = BimanualStack.__new__(BimanualStack)
    c.env = SimpleNamespace(
        data=SimpleNamespace(
            time=23, body=lambda _: SimpleNamespace(xpos=np.array([0.3, 0.2, 0.52]))
        ),
        robots={
            s: SimpleNamespace(actuator_indices=np.arange(i, i + 24))
            for s, i in (("right", 0), ("left", 24))
        },
    )
    c.phase, c.start_time, c.action = "BOTH PICK CUBES", 0, np.zeros(48)
    c.pickers = {
        side: SimpleNamespace(
            command=lambda _: (np.zeros(48), "HOLD"),
            palm=SimpleNamespace(close_time=None),
            audit=SimpleNamespace(
                grasp_feedback=lambda side=side: dict(
                    hand_contacts=0 if side == failed_side else 2
                )
            ),
        )
        for side in ("right", "left")
    }
    with pytest.raises(
        RuntimeError, match=f"{failed_side}: grasp/lift not established"
    ):
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
    "first,second,held",
    [
        ("right_finger", "red_cube", True),
        ("left_finger", "blue_cube", True),
        ("red_cube", "blue_cube", False),
        ("blue_cube", "table", False),
        ("right_finger", "blue_cube", True),
        ("left_finger", "red_cube", True),
        ("right_finger", "left_finger", False),
        ("left_finger", "table", False),
        ("left_panda_link6", "blue_cube", True),
        ("right_thumb", "right_finger", False),
    ],
)
def test_released_stack_check_only_detects_robot_cube_support(first, second, held):
    env = contact_fixture(first, second)
    env.geom_robots = tuple(
        next(
            (
                side
                for side in ("right", "left")
                if env.model.body(body).name.startswith(side + "_")
            ),
            None,
        )
        for body in env.model.geom_bodyid
    )
    assert bool(robot_cube_contacts(env)) == held


def test_unloaded_robot_cube_contact_is_not_support(monkeypatch):
    import mujoco

    env = contact_fixture("right_finger", "red_cube")
    monkeypatch.setattr(mujoco, "mj_contactForce", lambda m, d, i, force: force.fill(0))
    env.geom_robots = ("right", None, None)
    assert robot_cube_contacts(env) == 0


@pytest.mark.slow
def test_continuous_bimanual_pick_place_stack(tmp_path, monkeypatch):
    from orca_sim import contact_diagnostics
    from orca_sim.demos import bimanual_grasp
    from orca_sim.demos.bimanual_stack import run

    def diagnostic_called(*args, **kwargs):
        raise AssertionError("Stacking must not run the removed diagnostics")

    monkeypatch.setattr(bimanual_grasp.ContactAudit, "inspect", diagnostic_called)
    for module in (contact_diagnostics, bimanual_grasp):
        monkeypatch.setattr(module, "skin_geoms", diagnostic_called)
        monkeypatch.setattr(module, "skin_vertex_depth", diagnostic_called)
    report = run(tmp_path)
    assert report["success"], report
    assert report["validation_scope"] == "task_outcome_only"
    assert not {
        "collision_checks",
        "forbidden_contact_count",
        "forbidden_contacts",
        "max_contact_depth_m",
        "max_skin_vertex_depth_m",
        "max_joint_limit_violation_rad",
        "max_hand_torque_Nm",
        "external_forces_zero",
        "equalities",
        "skin_vertex_probe_interval_s",
    }.intersection(report)
    assert not list(tmp_path.glob("*.npz"))
    assert report["stable_stack_samples"] >= 499
    assert report["validation_settle_seconds"] >= 5
    assert report["base_target_error_m"] < 0.012
    assert report["max_lift_m"]["red_cube"] > 0.15
    assert report["max_lift_m"]["blue_cube"] > 0.15
    assert report["duration_s"] < 71
    assert report["events"][0]["phase"] == "BOTH PICK CUBES"
    assert (
        abs(
            report["pickup_lift_start_s"]["red_cube"]
            - report["pickup_lift_start_s"]["blue_cube"]
        )
        < 1
    )
