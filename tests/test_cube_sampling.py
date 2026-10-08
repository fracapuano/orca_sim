import copy
import json

import mujoco
import numpy as np
import pytest

from orca_sim.cube_sampling import CUBES, separated, spawn_configuration
from orca_sim.mujoco_bimanual import (
    CUBE_SPAWN_REGIONS,
    OBJECT_XY_LIMITS,
    STACK_LAYOUT,
    OrcaBimanualPandaCubeStackingMujoco,
)


def config():
    return json.loads(CUBE_SPAWN_REGIONS.read_text())


def make_env(**kwargs):
    return OrcaBimanualPandaCubeStackingMujoco(
        object_layout=STACK_LAYOUT,
        cube_spawn_regions=kwargs.pop("regions", config()),
        **kwargs,
    )


def test_configuration_roundtrip_and_detached_copy(tmp_path):
    original = config()
    parsed = spawn_configuration(original, OBJECT_XY_LIMITS)
    original["red_cube"]["x"][0] = -10
    assert parsed == config()
    path = tmp_path / "regions.json"
    path.write_text(json.dumps(parsed))
    assert spawn_configuration(path, OBJECT_XY_LIMITS) == parsed


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.update(version=2),
        lambda c: c.update(version=True),
        lambda c: c.update(min_gap_m=-1),
        lambda c: c.update(min_gap_m=float("nan")),
        lambda c: c.update(max_attempts=True),
        lambda c: c.update(max_attempts=0),
        lambda c: c.update(max_attempts=10001),
        lambda c: c.update(extra=1),
        lambda c: c.pop("blue_cube"),
        lambda c: c["red_cube"].update(z=[0, 1]),
        lambda c: c["red_cube"].update(x=[0.4, 0.2]),
        lambda c: c["red_cube"].update(x=[float("nan"), 0.2]),
        lambda c: c["red_cube"].update(x=[-0.5, 0.2]),
        lambda c: c["red_cube"].update(x=[0.2]),
    ],
)
def test_invalid_configuration_is_rejected(change):
    value = config()
    change(value)
    with pytest.raises(ValueError):
        spawn_configuration(value, OBJECT_XY_LIMITS)


def test_surface_gap_not_just_centre_distance():
    sizes = {name: np.array([0.028] * 3) for name in CUBES}
    assert not separated(
        dict(red_cube=np.zeros(2), blue_cube=np.array([0.03, 0.03])), sizes, 0
    )
    assert not separated(
        dict(red_cube=np.zeros(2), blue_cube=np.array([0.07, 0])), sizes, 0.02
    )
    assert separated(
        dict(red_cube=np.zeros(2), blue_cube=np.array([0.08, 0])), sizes, 0.02
    )


def test_seeded_reset_reproducible_across_envs_and_advances_without_seed():
    a, b = make_env(), make_env()
    try:
        first, info = a.reset(seed=123)
        second, other = b.reset(seed=123)
        np.testing.assert_array_equal(first, second)
        assert info == other
        assert info["spawn_attempts"] == 1
        for _ in range(5):
            obs, metadata = a.reset()
            obs_b, metadata_b = b.reset()
            np.testing.assert_array_equal(obs, obs_b)
            assert metadata == metadata_b
            assert not np.array_equal(obs, first)
        np.testing.assert_array_equal(a.reset(seed=123)[0], first)
        # Returned metadata and public configuration are detached snapshots.
        info["object_layout"]["red_cube"][0] = 100
        info["cube_spawn_regions"]["red_cube"]["x"][0] = 100
        public = a.cube_spawn_regions
        public["red_cube"]["x"][0] = 100
        assert a.cube_spawn_regions == config()
        assert a.object_layout["red_cube"][0] < 1
    finally:
        a.close()
        b.close()


def test_samples_stay_inside_regions_without_modifying_homes_or_model():
    env = make_env()
    try:
        m, d = env.model, env.data
        original_key = m.key("orcapanda_home").qpos.copy()
        original_ctrl = m.key("orcapanda_home").ctrl.copy()
        model_id = id(m)
        all_xy = []
        for seed in range(100):
            # Reset must also clear all previous episode dynamics/forces.
            d.qvel[:] = 1
            d.xfrc_applied[:] = 1
            _, info = env.reset(seed=seed)
            assert id(env.model) == model_id
            np.testing.assert_array_equal(d.qpos[env.qadr], original_key[env.qadr])
            np.testing.assert_array_equal(d.ctrl, original_ctrl)
            np.testing.assert_array_equal(m.key("orcapanda_home").qpos, original_key)
            assert not np.any(d.qvel) and not np.any(d.xfrc_applied)
            assert d.time == 0
            np.testing.assert_allclose(d.body("stack_target").xpos, [0.5, 0.201, 0.326])
            for name in CUBES:
                position = d.body(name).xpos
                np.testing.assert_array_equal(position, info["object_layout"][name])
                for i, axis in enumerate(("x", "y")):
                    low, high = config()[name][axis]
                    assert low <= position[i] <= high
                assert position[2] == pytest.approx(0.353)
                np.testing.assert_array_equal(d.body(name).xquat, [1, 0, 0, 0])
            all_xy.append(
                np.r_[d.body("red_cube").xpos[:2], d.body("blue_cube").xpos[:2]]
            )
        assert np.all(np.ptp(all_xy, axis=0) > 0.03)
    finally:
        env.close()


def test_fixed_layout_remains_default():
    env = make_env(regions=None)
    try:
        a, info = env.reset(seed=1)
        b, _ = env.reset(seed=2)
        np.testing.assert_array_equal(a, b)
        assert info["cube_spawn_regions"] is None
        assert info["spawn_attempts"] == 0
        assert info["object_layout"] == json.loads(STACK_LAYOUT.read_text())
    finally:
        env.close()


def test_overlapping_regions_use_seeded_joint_rejection():
    regions = config()
    regions["red_cube"] = regions["blue_cube"] = dict(x=[0.4, 0.6], y=[0.16, 0.24])
    env = make_env(regions=regions)
    try:
        attempts = []
        for seed in range(30):
            _, info = env.reset(seed=seed)
            xy = {name: env.data.body(name).xpos[:2] for name in CUBES}
            sizes = {name: env.model.geom(name + "_geom").size for name in CUBES}
            assert separated(xy, sizes, regions["min_gap_m"])
            attempts.append(info["spawn_attempts"])
        assert max(attempts) > 1
        _, a = env.reset(seed=7)
        _, b = env.reset(seed=7)
        assert a == b
    finally:
        env.close()


@pytest.mark.parametrize("robot_overlap", [False, True])
def test_impossible_regions_fail_explicitly(robot_overlap):
    regions = config()
    regions["max_attempts"] = 3
    if robot_overlap:
        # Centre of the right Panda's base, at cube resting height.
        regions["red_cube"] = dict(x=[0, 0], y=[-0.224, -0.224])
    else:
        regions["red_cube"] = regions["blue_cube"] = dict(x=[0.5, 0.5], y=[0.2, 0.2])
    with pytest.raises(ValueError, match="No valid cube placement after 3 proposals"):
        make_env(regions=regions)


def test_demo_and_policy_receive_identical_seeded_reset(monkeypatch, tmp_path):
    from orca_sim.demos import bimanual_stack as demo

    captured = {}

    def stop_after_reset(env):
        captured["qpos"] = env.data.qpos.copy()
        captured["layout"] = copy.deepcopy(env.object_layout)
        raise RuntimeError("Only checking the shared reset, not running the controller")

    monkeypatch.setattr(demo, "BimanualStack", stop_after_reset)
    report = demo.run(tmp_path, cube_spawn_regions=CUBE_SPAWN_REGIONS, seed=17)
    env = make_env(target_marker="x")
    try:
        _, info = env.reset(seed=17)
        np.testing.assert_array_equal(env.data.qpos, captured["qpos"])
        assert info["object_layout"] == captured["layout"] == report["object_layout"]
        reset = json.loads((tmp_path / "reset.json").read_text())
        assert reset == dict(seed=17, **info)
        assert report["cube_spawn_regions"] == config()
        # Exported scene also reproduces the sampled initial positions.
        m = mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("orcapanda_home").id)
        mujoco.mj_forward(m, d)
        for name in CUBES:
            np.testing.assert_array_equal(d.body(name).xpos, env.data.body(name).xpos)
    finally:
        env.close()
