"""Preview all three bimanual cameras; optionally record a stepped motion demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from orca_sim.demos.bimanual import motion_action, motion_paths
from orca_sim.mujoco_bimanual import OrcaBimanualPandaCubeStackingMujoco


def mosaic(images, *, caption="HOME"):
    """Label copies of the images; never modify the raw camera observations."""
    frame = Image.fromarray(np.concatenate(list(images.values()), axis=1))
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default(size=18)
    x = 0
    for name, rgb in images.items():
        draw.rectangle((x, 0, x + rgb.shape[1], 32), fill=(20, 24, 31))
        draw.text((x + 10, 6), f"{name} | {caption}", font=font, fill="white")
        x += rgb.shape[1]
    return frame


def run(output_dir, *, video=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = OrcaBimanualPandaCubeStackingMujoco()
    writer = None
    try:
        images = env.render_cameras()
        for name, rgb in images.items():
            Image.fromarray(rgb).save(output_dir / f"{name}.png")
        mosaic(images).save(output_dir / "cameras.png")
        (output_dir / "scene.xml").write_text(env.xml)
        maximum = dict(inter_robot_depth_m=0.0, robot_table_depth_m=0.0)
        if video:
            paths, home = motion_paths(env), env.data.ctrl.copy()
            writer = imageio.get_writer(
                output_dir / "cameras.mp4", fps=10, codec="libx264", quality=8
            )
            writer.append_data(np.asarray(mosaic(images, caption="t=0.0s")))
            next_frame = 0.1
            while env.data.time < 12 - 1e-9:
                action, _ = motion_action(env, home, paths, env.data.time)
                env.step(action)
                metrics = env.contact_metrics()
                for name in maximum:
                    maximum[name] = max(maximum[name], metrics[name])
                if env.data.time >= next_frame - 1e-9:
                    writer.append_data(
                        np.asarray(
                            mosaic(
                                env.render_cameras(), caption=f"t={env.data.time:.1f}s"
                            )
                        )
                    )
                    next_frame += 0.1
        report = dict(
            cameras=list(env.camera_names),
            duration_s=float(env.data.time),
            task="independent_arm_motion" if video else "home_snapshot",
            object_layout=env.object_layout,
            **maximum,
            warnings={
                str(i): int(w.number)
                for i, w in enumerate(env.data.warning)
                if w.number
            },
        )
        (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return report
    finally:
        if writer is not None:
            writer.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("renders/bimanual_cameras")
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Step and record 12 seconds of arm/hand motion",
    )
    args = parser.parse_args()
    run(args.output_dir, video=args.video)


if __name__ == "__main__":
    main()
