#!/usr/bin/env python3
"""Inspect a recorded native-hand experiment with synchronized camera orbits.

Replays the saved scene and commands to one recorded sample, verifies its cube
pose and skin-vertex overlap, then pauses physics. Only the camera moves in the
output video. No geometry, collision settings, or physical state is edited.
The optional diagnostic view changes cube opacity and adds visual markers only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from standalone_hand_grasp import skin_geoms, skin_vertex_depth, targets


def replay_sample(directory: Path, side: str, variant: str, sample_time: float):
    record = json.loads((directory / f"{side}_{variant}.json").read_text())
    result = record["result"]
    source = Path(result["source_body"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != result["source_body_sha256"]:
        raise ValueError(f"Source MJCF has changed since recording: {source}")
    sample = min(record["samples"], key=lambda s: abs(s["t"] - sample_time))
    model = mujoco.MjModel.from_xml_path(
        str((directory / f"{side}_{variant}_scene.xml").resolve())
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    support = model.body("support").id
    mocap = int(model.body_mocapid[support])
    start = model.body_pos[support].copy()
    target = targets(model, side, result["closure_fraction"])
    for step in range(round(sample["t"] / model.opt.timestep)):
        t = step * model.opt.timestep
        data.ctrl[:] = target * np.clip((t - 0.5) / 2, 0, 1)
        data.mocap_pos[mocap] = start - [0, 0, 0.25 * np.clip((t - 3) / 0.5, 0, 1)]
        mujoco.mj_step(model, data)
        if step % 10 == 0:
            mujoco.mj_forward(model, data)
    mujoco.mj_forward(model, data)
    cube = model.geom("cube_geom").id
    overlap = skin_vertex_depth(model, data, cube, skin_geoms(model))
    np.testing.assert_allclose(
        data.xpos[model.body("cube").id], sample["cube_pos"], atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        overlap, sample["skin_vertex_depth_m"], atol=1e-10, rtol=0
    )
    assert not any(w.number for w in data.warning)
    print(
        f"Verified {variant}: t={data.time:.3f}, skin overlap={overlap * 1000:.3f} mm",
        flush=True,
    )
    return model, data, overlap


def penetrating_skin_points(model, data):
    """World-space skin vertices more than 0.1 mm inside the exact cube."""
    cube = model.geom("cube_geom").id
    rotation = data.geom_xmat[cube].reshape(3, 3)
    points = []
    for geom in skin_geoms(model):
        mesh = int(model.geom_dataid[geom])
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        world = (
            model.mesh_vert[start : start + count]
            @ data.geom_xmat[geom].reshape(3, 3).T
            + data.geom_xpos[geom]
        )
        local = (world - data.geom_xpos[cube]) @ rotation
        depth = np.min(model.geom_size[cube] - np.abs(local), axis=1)
        selected = world[depth > 0.0001]
        # Bound diagnostic-marker count without changing the overlap metric.
        points.extend(selected[:: max(1, len(selected) // 80)])
    return points


def render_pair(states, renderers, azimuth, elevation, distance, diagnostic=False):
    frames = []
    font = ImageFont.load_default(size=21)
    small = ImageFont.load_default(size=18)
    for (model, data, overlap), renderer, title in zip(
        states, renderers, ("ORIGINAL SKIN MASKS", "CORRECTED MJCF")
    ):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [-0.01, 0.17, 0.395]
        camera.distance = distance
        camera.azimuth, camera.elevation = azimuth, elevation
        renderer.update_scene(data, camera=camera)
        if diagnostic:
            # Opacity belongs to the rendering scene, not to the physics model.
            cube = model.geom("cube_geom").id
            for geom in renderer.scene.geoms[: renderer.scene.ngeom]:
                if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid == cube:
                    geom.rgba[3] = 0.18
                    geom.transparent = 1
            for point in penetrating_skin_points(model, data):
                if renderer.scene.ngeom >= renderer.scene.maxgeom:
                    raise RuntimeError("Too many diagnostic markers")
                mujoco.mjv_initGeom(
                    renderer.scene.geoms[renderer.scene.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.00065] * 3),
                    pos=point,
                    mat=np.eye(3).reshape(-1),
                    rgba=np.array([1, 0.72, 0.05, 1]),
                )
                renderer.scene.ngeom += 1
        frame = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, 960, 91), fill=(20, 24, 31))
        draw.text((15, 9), title, fill="white", font=font)
        draw.text(
            (15, 38),
            f"Physics paused at {data.time:.3f}s | camera orbit only",
            fill="white",
            font=small,
        )
        draw.text(
            (15, 65),
            f"Sampled skin overlap: {overlap * 1000:.2f} mm",
            fill="white",
            font=small,
        )
        draw.rectangle((0, 688, 960, 720), fill=(20, 24, 31))
        draw.text(
            (15, 694),
            "DIAGNOSTIC: transparent cube | amber dots = skin vertices inside (>0.1 mm)"
            if diagnostic
            else f"Azimuth {azimuth % 360:.0f} deg | elevation {elevation:.0f} deg",
            fill="white",
            font=small,
        )
        frames.append(np.asarray(frame))
    return np.concatenate(frames, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path("renders/standalone_hand/validated")
    )
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--time", type=float, default=5)
    parser.add_argument("--elevation", type=float, default=15)
    parser.add_argument("--distance", type=float, default=0.23)
    parser.add_argument("--seconds", type=float, default=16)
    parser.add_argument("--stills-only", action="store_true")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Transparent cube and markers on penetrating skin vertices (rendering only)",
    )
    args = parser.parse_args()
    if args.seconds <= 0 or args.distance <= 0:
        parser.error("Duration and camera distance must be positive")
    states = [
        replay_sample(args.directory, args.side, variant, args.time)
        for variant in ("skin_disabled", "released")
    ]
    renderers = []
    output = args.directory / ("orbit_diagnostic" if args.diagnostic else "orbit")
    output.mkdir(parents=True, exist_ok=True)
    try:
        renderers = [
            mujoco.Renderer(model, height=720, width=960) for model, _, _ in states
        ]
        for angle in (0, 45, 90, 135, 180, 225, 270, 315):
            imageio.imwrite(
                output / f"{args.side}_azimuth_{angle}.png",
                render_pair(
                    states,
                    renderers,
                    angle,
                    args.elevation,
                    args.distance,
                    args.diagnostic,
                ),
            )
        if not args.stills_only:
            with imageio.get_writer(
                output / f"{args.side}_orbit.mp4", fps=30, codec="libx264", quality=8
            ) as writer:
                for frame in range(round(args.seconds * 30)):
                    angle = 90 + 360 * frame / (args.seconds * 30)
                    writer.append_data(
                        render_pair(
                            states,
                            renderers,
                            angle,
                            args.elevation,
                            args.distance,
                            args.diagnostic,
                        )
                    )
        print(output.resolve(), flush=True)
    finally:
        for renderer in renderers:
            renderer.close()


if __name__ == "__main__":
    main()
