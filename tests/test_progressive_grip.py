"""Model-only comparison: increasing native targets, no protective controller."""

import mujoco
import numpy as np
import pytest

from scripts.progressive_grip_comparison import verify_mechanics
from scripts.standalone_hand_grasp import scene_xml, targets
from scripts import standalone_squeeze_stress as stress


@pytest.mark.parametrize("side", ["right", "left"])
def test_progressive_schedule_monotonic_and_reaches_native_limits(side):
    model = mujoco.MjModel.from_xml_string(scene_xml(side))
    base = targets(model, side, 0.6)
    commands = np.array(
        [stress.progressive_target(model, side, t) for t in np.linspace(5, 40, 701)]
    )
    np.testing.assert_array_equal(commands[0], base)
    assert np.all(np.diff(commands, axis=0) >= -1e-14)
    for i, j in enumerate(model.actuator_trnid[:, 0]):
        if model.joint(int(j)).name.endswith(("-mcp", "-pip")):
            assert commands[-1, i] == pytest.approx(model.actuator_ctrlrange[i, 1])
        else:
            np.testing.assert_array_equal(commands[:, i], base[i])
    assert np.all(commands >= model.actuator_ctrlrange[:, 0])
    assert np.all(commands <= model.actuator_ctrlrange[:, 1] + 1e-14)
    np.testing.assert_allclose(
        stress.progressive_target(model, side, 15),
        stress.squeeze_target(model, side, "flexion", 1),
    )
    np.testing.assert_array_equal(
        stress.progressive_target(model, side, 25),
        stress.progressive_target(model, side, 40),
    )
    np.testing.assert_array_equal(model.actuator_forcerange, np.tile([-1, 1], (17, 1)))
    np.testing.assert_array_equal(model.actuator_gainprm[:, 0], 2)


@pytest.mark.parametrize("side", ["right", "left"])
def test_ab_has_identical_mechanics_and_target_commands(side):
    assert "actuator_forcerange" in verify_mechanics(side)
    native = mujoco.MjModel.from_xml_string(scene_xml(side))
    previous = mujoco.MjModel.from_xml_string(scene_xml(side, contact_model="legacy"))
    for t in (0, 1, 5, 8, 15, 20, 25, 40):
        np.testing.assert_array_equal(
            stress.progressive_target(native, side, t),
            stress.progressive_target(previous, side, t),
        )


@pytest.mark.parametrize("t", [-1, np.nan, np.inf])
def test_progressive_schedule_rejects_invalid_time(t):
    model = mujoco.MjModel.from_xml_string(scene_xml())
    with pytest.raises(ValueError):
        stress.progressive_target(model, "right", t)


def test_progressive_needs_time_to_finish_and_hold():
    with pytest.raises(ValueError, match="30 s"):
        stress.run(profile="progressive")


def test_default_simulation_does_not_use_protective_control():
    import inspect

    assert inspect.signature(stress.run).parameters["profile"].default == "flexion"


@pytest.mark.parametrize("side", ["right", "left"])
def test_thumb_and_palm_are_real_collision_surfaces(side):
    model = mujoco.MjModel.from_xml_string(scene_xml(side))
    regions = stress.hand_contact_regions(model, side)
    assert {"fingers", "thumb", "palm"} <= set(regions.values())
    cube = model.geom("cube_geom").id
    for geom, region in regions.items():
        if region not in ("palm", "thumb"):
            continue
        assert (
            model.geom_contype[geom] & model.geom_conaffinity[cube]
            or model.geom_conaffinity[geom] & model.geom_contype[cube]
        )
        assert model.geom_priority[geom] == 1
        np.testing.assert_allclose(model.geom_solref[geom], [0.004, 1])


@pytest.mark.slow
@pytest.mark.parametrize("side", ["right", "left"])
def test_progressive_native_grip_without_preload_adapter(side, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("This comparison must not call GraspPreload")

    monkeypatch.setattr(stress, "GraspPreload", forbidden)
    previous = stress.run(
        side=side, profile="progressive", duration=40, contact_model="legacy"
    )
    corrected = stress.run(side=side, profile="progressive", duration=40)
    assert previous["command_trace_sha256"] == corrected["command_trace_sha256"]
    assert not previous["retained_at_end"]
    assert previous["max_skin_vertex_depth_m"] > 0.002
    assert corrected["retained_at_end"]
    assert corrected["passes_contact_check"]
    assert not previous["passes_contact_check"]
    for region in ("fingers", "thumb", "palm"):
        assert corrected["peak_normal_force_by_region_N"][region] > 1, region
    assert corrected["max_skin_vertex_depth_m"] < 0.0001
    assert corrected["max_raw_contact_depth_every_physics_step_m"] < 0.0001
    assert corrected["max_abs_motor_torque_every_physics_step_Nm"] <= 1
    assert corrected["peak_total_normal_N"] > 200
    assert corrected["applied_preload_multiplier"] is None
    assert corrected["squeeze_ramp_seconds"] == 20
    assert corrected["finite_state_every_physics_step"]
    assert (
        corrected["external_force_arrays_zero"]
        and corrected["equality_constraints"] == 0
    )
    assert not any(corrected["warnings"])
