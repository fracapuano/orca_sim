import mujoco
import numpy as np

from orca_sim.demos.spawn_regions import add_region_overlays
from orca_sim.mujoco_bimanual import (
    CUBE_SPAWN_REGIONS,
    STACK_LAYOUT,
    OrcaBimanualPandaCubeStackingMujoco,
)


def test_region_plot_changes_render_scene_only_and_keeps_x_fixed():
    env = OrcaBimanualPandaCubeStackingMujoco(
        object_layout=STACK_LAYOUT,
        target_marker="x",
        cube_spawn_regions=CUBE_SPAWN_REGIONS,
    )
    try:
        env.reset(seed=3)
        model, data = env.model, env.data
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, data, state, spec)
        sizes, rgba = model.geom_size.copy(), model.geom_rgba.copy()
        camera = mujoco.MjvCamera()
        scene = mujoco.MjvScene(model, maxgeom=2000)
        mujoco.mjv_updateScene(
            model,
            data,
            mujoco.MjvOption(),
            None,
            camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            scene,
        )
        count = scene.ngeom
        add_region_overlays(scene, env)
        assert scene.ngeom == count + 18
        for index, name in enumerate(("red_cube", "blue_cube")):
            fill = scene.geoms[count + index * 9]
            region = env.cube_spawn_regions[name]
            bounds = np.array([region[axis] for axis in ("x", "y")])
            np.testing.assert_allclose(fill.pos[:2], bounds.mean(axis=1))
            np.testing.assert_allclose(
                fill.size[:2], np.diff(bounds, axis=1).ravel() / 2
            )
            assert fill.category == mujoco.mjtCatBit.mjCAT_DECOR
        after = np.empty_like(state)
        mujoco.mj_getState(model, data, after, spec)
        np.testing.assert_array_equal(state, after)
        np.testing.assert_array_equal(model.geom_size, sizes)
        np.testing.assert_array_equal(model.geom_rgba, rgba)
        np.testing.assert_array_equal(
            data.body("stack_target").xpos, [0.5, 0.201, 0.326]
        )
        # Native model and all three policy cameras remain unmodified.
        assert model.ncam == 3
        assert data.time == 0
    finally:
        env.close()
