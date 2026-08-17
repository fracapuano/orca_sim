#!/usr/bin/env python3
"""Run the deterministic Genesis bimanual ORCA-Panda grasp validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from orca_sim.genesis_env import BimanualOrcaPandaCubeGenesisEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render-mode",
        choices=("none", "human", "rgb_array"),
        default="rgb_array",
        help="Use rgb_array to save ready/open/grasp/lift validation frames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("renders/genesis"),
        help="Directory for rgb_array validation frames.",
    )
    parser.add_argument(
        "--backend", default="cpu", help="Genesis backend (default: cpu)."
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Optional continuous MP4 rollout (requires --render-mode rgb_array).",
    )
    parser.add_argument("--fps", type=int, default=30, help="Recorded video frame rate.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    render_mode = None if args.render_mode == "none" else args.render_mode
    if args.video is not None and render_mode != "rgb_array":
        raise ValueError("--video requires --render-mode rgb_array")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    env = BimanualOrcaPandaCubeGenesisEnv(
        render_mode=render_mode,
        backend=args.backend,
    )

    capture = None
    writer = None
    record = None
    if render_mode == "rgb_array":
        from PIL import Image

        args.output_dir.mkdir(parents=True, exist_ok=True)

        def capture(phase: str, frame: np.ndarray) -> None:
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
                args.output_dir / f"{phase}.png"
            )

        if args.video is not None:
            import imageio.v2 as imageio

            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(
                args.video,
                fps=args.fps,
                codec="libx264",
                quality=8,
                pixelformat="yuv420p",
            )

            def record(frame: np.ndarray) -> None:
                writer.append_data(np.asarray(frame, dtype=np.uint8))

    try:
        result = env.run_scripted_grasp(
            capture=capture,
            record=record,
            record_fps=args.fps,
        )
    finally:
        if writer is not None:
            writer.close()
        env.close()

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    if render_mode == "rgb_array":
        print(f"Saved validation frames to {args.output_dir.resolve()}")
    if args.video is not None:
        print(f"Saved continuous rollout to {args.video.resolve()}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
