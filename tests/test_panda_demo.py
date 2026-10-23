"""Controller and full-physics regressions for the installable Panda demo."""

import json

import mujoco
import numpy as np
import pytest

from orca_sim.mujoco_panda import OrcaPandaCubeStackingMujoco


@pytest.fixture
def env():
    e = OrcaPandaCubeStackingMujoco()
    yield e
    e.close()


def test_sideways_approach_keeps_home_orientation_and_joint_branch(env):
    from orca_sim.demos.panda_grasp import translation_approach

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
    from orca_sim.demos.panda_grasp import run

    with pytest.raises(ValueError, match="saved home orientation"):
        run(output_dir=tmp_path, sideways=True, **options)


@pytest.mark.slow
def test_approach_grasp_lift(tmp_path):
    from orca_sim.demos.panda_grasp import run

    result = run(output_dir=tmp_path)
    assert result["success"]
    assert result["minimum_hold_lift_m"] > 0.1
    assert result["max_hand_torque_Nm"] <= 1.0 + 1e-10
    assert result["equalities"] == 0 and result["external_forces_zero"]
    assert not result["warnings"]


@pytest.mark.slow
def test_stack_release_and_retreat(tmp_path):
    from orca_sim.demos.panda_grasp import run

    result = run(output_dir=tmp_path, stack=True)
    assert result["success"] and result["stack_success"]
    assert result["final_stack"]["stack_contacts"] >= 1
    assert result["final_stack"]["stack_height_m"] == pytest.approx(0.056, abs=0.005)
    assert result["equalities"] == 0 and result["external_forces_zero"]


@pytest.mark.slow
def test_sideways_home_orientation_grasp_and_stack(tmp_path):
    from orca_sim.demos.panda_grasp import run

    result = run(output_dir=tmp_path, sideways=True, stack=True)
    assert result["success"] and result["grasp_success"] and result["stack_success"]
    assert result["max_approach_hand_frame_rotation_deg"] < 1.0
    assert result["placement_orientation_mode"] == "keep_grasp_orientation"
    assert result["minimum_hold_lift_m"] > 0.1
    assert result["equalities"] == 0 and result["external_forces_zero"]
    assert not result["warnings"]


@pytest.mark.slow
def test_palm_contact_precedes_wrap_lift_and_stack(tmp_path):
    from orca_sim.demos.panda_grasp import run

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
