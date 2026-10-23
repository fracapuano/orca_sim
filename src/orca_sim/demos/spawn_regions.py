"""Plot cube-centre sampling regions on the actual MuJoCo table, without stepping."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from orca_sim.cube_sampling import CUBES
from orca_sim.mujoco_bimanual import (
    CUBE_SPAWN_REGIONS,
    STACK_LAYOUT,
    OrcaBimanualPandaCubeStackingMujoco,
)
from orca_sim.mujoco_panda import TABLE_HEIGHT

COLORS = ((0.95, 0.08, 0.05, 0.65), (0.04, 0.25, 0.95, 0.65))


def add_region_overlays(scene, env):
    """Edit only a render scene: never model geometry, physics or observations.

    Filled rectangles are the support of each cube-centre distribution. A
    light outline shows one physical cube footprint centred in each region.
    Actual cubes are hidden in this render scene so they don't obscure the
    smaller sampling rectangles. The original X and robots are untouched.
    """
    config = env.cube_spawn_regions
    if config is None:
        raise ValueError("Configure cube_spawn_regions before plotting")
    if scene.ngeom + 18 > scene.maxgeom:
        raise ValueError("Render scene needs room for 18 region overlay geoms")
    cube_ids = {env.model.geom(name + "_geom").id for name in CUBES}
    for geom in scene.geoms[: scene.ngeom]:
        if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid in cube_ids:
            geom.rgba[3] = 0
            geom.size[:] = 0  # Alpha alone still leaves a misleading cube shadow.

    def box(pos, size, color):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_BOX,
            np.asarray(size, dtype=float),
            np.asarray(pos, dtype=float),
            np.eye(3).ravel(),
            np.asarray(color, dtype=np.float32),
        )
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1

    def border(center, half_size, color, z):
        x, y = center
        dx, dy = half_size
        width = 0.00055
        for sign in (-1, 1):
            box([x + sign * dx, y, z], [width, max(dy, width), 0.00015], color)
            box([x, y + sign * dy, z], [max(dx, width), width, 0.00015], color)

    for name, color in zip(CUBES, COLORS):
        bounds = np.array([config[name][axis] for axis in ("x", "y")])
        center, half_size = bounds.mean(axis=1), np.diff(bounds, axis=1).ravel() / 2
        box(
            [*center, TABLE_HEIGHT + 0.0015],
            [*np.maximum(half_size, 0.0002), 0.0001],
            color,
        )
        border(center, half_size, (*color[:3], 1), TABLE_HEIGHT + 0.0018)
        footprint = env.model.geom(name + "_geom").size[:2]
        border(center, footprint, (0.32, 0.32, 0.32, 1), TABLE_HEIGHT + 0.002)


def run(output_dir, *, regions=CUBE_SPAWN_REGIONS):
    from PIL import Image, ImageDraw, ImageFont

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = OrcaBimanualPandaCubeStackingMujoco(
        object_layout=STACK_LAYOUT,
        target_marker="x",
        cube_spawn_regions=regions,
    )
    renderer = None
    try:
        env.reset(seed=0)
        before_qpos, before_x = (
            env.data.qpos.copy(),
            env.data.body("stack_target").xpos.copy(),
        )
        renderer = mujoco.Renderer(env.model, width=1280, height=720)
        images = []
        for camera in ("front", "detail"):
            view = camera
            if camera == "detail":
                view = mujoco.MjvCamera()
                view.lookat[:] = before_x
                view.distance, view.azimuth, view.elevation = 0.68, 90, -90
            renderer.update_scene(env.data, camera=view)
            add_region_overlays(renderer.scene, env)
            if camera == "detail":
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            pixels = Image.fromarray(renderer.render().copy())
            pixels.save(output_dir / f"{camera}.png")
            images.append(pixels)
        # Report-size two-view figure. Nothing about these labels enters a
        # camera observation during training or ordinary simulation rendering.
        figure = Image.new("RGB", (1280, 1255), (244, 246, 249))
        draw = ImageDraw.Draw(figure)
        title = ImageFont.load_default(size=28)
        label = ImageFont.load_default(size=21)
        small = ImageFont.load_default(size=18)
        dark = (26, 34, 47)
        draw.text(
            (28, 18), "Cube starting-position distributions", font=title, fill=dark
        )
        draw.text(
            (28, 55),
            "Filled regions: uniform cube centres  |  Grey outlines: 5.6 cm cube footprint",
            font=label,
            fill=dark,
        )
        figure.paste(images[0].resize((1152, 648)), (64, 94))
        draw.text(
            (28, 758), "Top-down detail - same scale in X and Y", font=label, fill=dark
        )
        # Tight crop of the rendered tabletop, not a distorted rescale.
        figure.paste(images[1].crop((0, 220, 1280, 500)), (0, 796))
        config = env.cube_spawn_regions
        for name, x, color, arm in zip(
            CUBES, (95, 805), ((185, 28, 30), (25, 75, 190)), ("right", "left")
        ):
            region = config[name]
            width, depth = (np.diff(region[axis])[0] * 100 for axis in ("x", "y"))
            draw.text(
                (x, 1094),
                f"{name.split('_')[0].upper()} / {arm} arm: {width:g} x {depth:g} cm",
                font=label,
                fill=color,
            )
            draw.text(
                (x, 1126),
                f"X {region['x'][0]:.3f} - {region['x'][1]:.3f} m",
                font=small,
                fill=dark,
            )
            draw.text(
                (x, 1150),
                f"Y {region['y'][0]:.3f} - {region['y'][1]:.3f} m",
                font=small,
                fill=dark,
            )
        draw.text((490, 1094), "X stays fixed", font=label, fill=dark)
        draw.text(
            (485, 1126),
            f"({before_x[0]:.3f}, {before_x[1]:.3f}) m",
            font=small,
            fill=dark,
        )
        draw.text(
            (28, 1212),
            "Cubes hidden for visibility. Robot home, target and physics are unchanged. This is not a success-probability map.",
            font=small,
            fill=dark,
        )
        figure.save(output_dir / "distributions.png")
        np.testing.assert_array_equal(env.data.qpos, before_qpos)
        np.testing.assert_array_equal(env.data.body("stack_target").xpos, before_x)
        assert env.data.time == 0
    finally:
        if renderer is not None:
            renderer.close()
        env.close()
    print(output_dir / "distributions.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-spawn-regions", type=Path, default=CUBE_SPAWN_REGIONS)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("renders/cube_spawn_regions")
    )
    args = parser.parse_args()
    run(args.output_dir, regions=args.cube_spawn_regions)


if __name__ == "__main__":
    main()
