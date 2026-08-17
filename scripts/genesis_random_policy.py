#!/usr/bin/env python3
"""Visualize a smooth bounded random policy in the Genesis environment."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from orca_sim.genesis_env import (
    N_ARM_DOFS,
    N_HAND_DOFS,
    BimanualOrcaPandaCubeGenesisEnv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render-mode",
        choices=("human", "rgb_array"),
        default="human",
        help="Open the live viewer or record an MP4.",
    )
    parser.add_argument("--duration", type=float, default=30.0, help="Rollout seconds.")
    parser.add_argument("--fps", type=int, default=30, help="Viewer/video frame rate.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--arm-noise", type=float, default=0.10, help="Arm target σ in rad."
    )
    parser.add_argument(
        "--hand-noise", type=float, default=0.18, help="Hand target σ in rad."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("renders/genesis_random_policy.mp4"),
        help="Output used with --render-mode rgb_array.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration <= 0 or args.fps <= 0:
        raise ValueError("duration and fps must be positive")
    if args.arm_noise < 0 or args.hand_noise < 0:
        raise ValueError("noise scales cannot be negative")

    # One Gym step should correspond closely to one displayed video frame.
    physics_dt = 0.0025
    frame_skip = max(1, round(1.0 / (physics_dt * args.fps)))
    env = BimanualOrcaPandaCubeGenesisEnv(
        render_mode=args.render_mode,
        frame_skip=frame_skip,
        max_episode_steps=int(np.ceil(args.duration * args.fps)) + 1,
    )
    rng = np.random.default_rng(args.seed)
    writer = None

    try:
        obs, _ = env.reset(options={"settle_steps": 100})
        nominal = obs[: env.action_space.shape[0]].copy()
        action = nominal.copy()
        target = nominal.copy()
        sigma = np.concatenate(
            [
                np.full(N_ARM_DOFS, args.arm_noise),
                np.full(N_HAND_DOFS, args.hand_noise),
                np.full(N_ARM_DOFS, args.arm_noise),
                np.full(N_HAND_DOFS, args.hand_noise),
            ]
        )
        target_hold_frames = max(1, round(1.5 * args.fps))

        if args.render_mode == "rgb_array":
            import imageio.v2 as imageio

            args.output.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(
                args.output,
                fps=args.fps,
                codec="libx264",
                quality=8,
                pixelformat="yuv420p",
            )

        final_info = {}
        for step_index in range(round(args.duration * args.fps)):
            frame_started = time.perf_counter()
            if step_index % target_hold_frames == 0:
                target = np.clip(
                    nominal + rng.normal(size=nominal.shape) * sigma,
                    env.action_low,
                    env.action_high,
                )

            # Low-pass filtering prevents a random target draw from becoming an
            # unphysical one-frame joint discontinuity.
            action += 0.08 * (target - action)
            obs, _, terminated, truncated, final_info = env.step(action)

            if writer is not None:
                writer.append_data(np.asarray(env.render(), dtype=np.uint8))
            elif args.render_mode == "human":
                remaining = 1.0 / args.fps - (time.perf_counter() - frame_started)
                if remaining > 0:
                    time.sleep(remaining)

            if terminated or truncated:
                break
    finally:
        if writer is not None:
            writer.close()
        env.close()

    if args.render_mode == "rgb_array":
        print(f"Saved random-policy rollout to {args.output.resolve()}")
    print(
        "final cube position:",
        np.asarray(final_info.get("cube_position", np.full(3, np.nan))).round(5),
    )
    return 0 if final_info.get("finite", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
