from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from orca_sim.demos.bimanual_grasp import ContactAudit, PalmLift, run
from orca_sim.mujoco_bimanual import (
    OrcaBimanualPandaCubeStackingMujoco,
    tabletop_configuration,
)

REACH_LAYOUT = Path(__file__).parent / "fixtures/bimanual_reach_layout.json"


def contact_fixture(first, second, separation=0.04, margin=0):
    bodies = []
    for i, name in enumerate((first, second)):
        bodies.append(
            f'<body name="{name}" pos="{i * separation} 0 1">'
            f"{'<freejoint/>' if i == 0 else ''}"
            f'<geom name="{name}_geom" type="box" size=".028 .028 .028" '
            f'margin="{margin}"/></body>'
        )
    for i, name in enumerate(("red_cube", "blue_cube")):
        if name not in (first, second):
            bodies.append(
                f'<body name="{name}" pos="{5 + i} 0 1">'
                f'<geom name="{name}_geom" type="box" size=".028 .028 .028"/></body>'
            )
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><option gravity="0 0 0"/><worldbody>'
        + "".join(bodies)
        + "</worldbody></mujoco>"
    )
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert d.ncon > 0
    return SimpleNamespace(model=m, data=d)


@pytest.mark.parametrize(
    "first,second,forbidden",
    [
        ("right_finger", "red_cube", False),
        ("red_cube", "table", False),
        ("blue_cube", "table", False),
        ("right_finger", "table", True),
        ("right_finger", "blue_cube", True),
        ("left_finger", "red_cube", True),
        ("right_panda_link6", "red_cube", True),
        ("right_finger", "left_finger", True),
        ("right_finger", "right_thumb", True),
        ("right_panda_link6", "right_finger", True),
        ("right_finger", "floor", True),
    ],
)
def test_audit_only_allows_target_hand_contact_and_cube_support(
    first, second, forbidden
):
    env = contact_fixture(first, second)
    metrics = ContactAudit(env, "right", "red_cube").inspect()
    assert bool(metrics["forbidden"]) == forbidden


def test_positive_contact_margin_with_force_is_still_a_collision():
    env = contact_fixture("right_finger", "table", separation=0.0561, margin=0.001)
    assert all(c.dist > 0 for c in env.data.contact)
    metrics = ContactAudit(env, "right", "red_cube").inspect()
    assert metrics["forbidden"]
    assert metrics["forbidden"][0]["depth_m"] == 0
    assert metrics["forbidden"][0]["force_N"] > 0


@pytest.mark.parametrize("side", ("right", "left"))
def test_controller_planning_is_scratch_only_and_keeps_other_commands(side):
    layout = tabletop_configuration(REACH_LAYOUT)
    env = OrcaBimanualPandaCubeStackingMujoco(object_layout=layout)
    try:
        qpos, qvel, ctrl = (
            env.data.qpos.copy(),
            env.data.qvel.copy(),
            env.data.ctrl.copy(),
        )
        cube = "red_cube" if side == "right" else "blue_cube"
        controller = PalmLift(env, side, cube)
        np.testing.assert_array_equal(env.data.qpos, qpos)
        np.testing.assert_array_equal(env.data.qvel, qvel)
        np.testing.assert_array_equal(env.data.ctrl, ctrl)
        other = env.robots["left" if side == "right" else "right"].actuator_indices
        for time in (0, 1, 3, 5, 7):
            action, _ = controller.command(time)
            np.testing.assert_array_equal(action[other], ctrl[other])
            assert np.all(action >= env.action_space.low)
            assert np.all(action <= env.action_space.high)
        # Closure coordinates the fingers instead of relying on table/self contact.
        controller.palm.close_time = 11
        for time in (11.5, 12, 13, 14, 20):
            action, _ = controller.command(time)
            np.testing.assert_array_equal(action[other], ctrl[other])
            assert np.all(action >= env.action_space.low)
            assert np.all(action <= env.action_space.high)
    finally:
        env.close()


@pytest.mark.parametrize("side", ("right", "left"))
@pytest.mark.slow
def test_independent_real_grasp_with_explicit_reachable_layout(tmp_path, side):
    original = tabletop_configuration()
    layout = tabletop_configuration(REACH_LAYOUT)
    report = run(side=side, output_dir=tmp_path / side, object_layout=layout)
    assert report["success"], report
    assert report["collision_checks"] == 28000
    assert report["forbidden_contact_count"] == 0
    assert report["max_hand_cube_depth_m"] < 0.0001
    assert report["max_skin_vertex_depth_m"] == 0
    assert report["minimum_hold_lift_m"] > 0.17
    assert report["minimum_hold_contacts"] >= 2
    assert "t" in report["continuously_loaded_hold_digits"]
    assert len(report["continuously_loaded_hold_digits"]) >= 2
    assert report["max_inactive_command_change"] == 0
    assert report["max_other_cube_displacement_m"] < 1e-5
    assert report["external_forces_zero"] and report["equalities"] == 0
    assert tabletop_configuration() == original


@pytest.mark.parametrize("duration", (0, 10, 27, float("nan"), float("inf")))
def test_short_or_invalid_validation_rejected(tmp_path, duration):
    with pytest.raises(ValueError, match="at least 28"):
        run(side="right", output_dir=tmp_path, duration=duration)
