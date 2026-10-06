#!/usr/bin/env python3
"""Isolate the right hand's added skin contacts with identical action replays.

This diagnostic targets Genesis 1.3.3. It keeps one built scene (including its
mass/inertia, visuals, collider meshes, solver, and original collision filters)
and switches only pairs involving the active right hand's added skin colliders.
The idle left robot is identical in both trials. One left-hand skin is merged
with other geometry by the loader, so a both-hands toggle would not isolate it.
It is an ablation of their contact contribution, not a raw-original-MJCF test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from orca_sim.genesis_env import BimanualOrcaPandaCubeGenesisEnv, _as_numpy


def without_skin_pairs(pairs: np.ndarray, skin_indices: list[int]) -> np.ndarray:
    """Preserve pair IDs and every non-skin entry; -1 disables a pair."""
    result = pairs.copy()
    result[skin_indices, :] = -1
    result[:, skin_indices] = -1
    return result


class SkinContactSwitch:
    """Diagnostic-only access to the pinned engine's pair-filter field."""

    def __init__(self, env: BimanualOrcaPandaCubeGenesisEnv) -> None:
        if env.gs.__version__ != "1.3.3":
            raise RuntimeError(
                "Revalidate the pair-filter adapter for this Genesis version"
            )
        expected = {
            geom.get("name")
            for geom in ET.parse(env.asset_path).getroot().iter("geom")
            if geom.get("name", "").startswith("genesis_right_")
            and "Skin_collision_" in geom.get("name", "")
        }
        found = {
            geom.mesh.metadata.get("name"): geom.idx
            for geom in env.robot.geoms
            if geom.mesh.metadata.get("name") in expected
        }
        if not expected or found.keys() != expected:
            raise RuntimeError(
                f"Cannot isolate every skin collider: {expected - found.keys()}"
            )
        self.names = sorted(found)
        self.indices = [int(found[name]) for name in self.names]
        self.field = env.scene.rigid_solver.collider._collider_info.collision_pair_idx
        self.on = self.field.to_numpy().copy()
        self.off = without_skin_pairs(self.on, self.indices)
        self.inertias = self.inertial_signature(env)
        self.disabled_pairs = int(np.count_nonzero(self.on != self.off))
        if not self.disabled_pairs:
            raise RuntimeError("The skin switch would disable no collision pairs")

    @staticmethod
    def inertial_signature(env: BimanualOrcaPandaCubeGenesisEnv) -> str:
        digest = hashlib.sha256()
        for link in env.robot.links:
            for value in (
                link.inertial_mass,
                link.inertial_pos,
                link.inertial_quat,
                link.inertial_i,
            ):
                digest.update(np.asarray(value, dtype=np.float64).tobytes())
        return digest.hexdigest()

    def set(self, env: BimanualOrcaPandaCubeGenesisEnv, enabled: bool) -> None:
        # Both SAP and all-vs-all broadphase consult this field. Retain pair IDs
        # and allocation sizes so the two trials also use the same kernels.
        target = self.on if enabled else self.off
        self.field.from_numpy(target)
        np.testing.assert_array_equal(self.field.to_numpy(), target)
        if self.inertial_signature(env) != self.inertias:
            raise RuntimeError("Mass/inertia changed during the contact ablation")


def label_frame(
    frame: np.ndarray, variant: str, sim_time: float, height: float, contacts: int
) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(frame).resize((960, 540))
    canvas = Image.new("RGB", (960, 612), (20, 24, 31))
    canvas.paste(image, (0, 72))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    color = (105, 220, 174) if variant == "on" else (255, 179, 105)
    draw.text(
        (20, 10), f"Right-hand skin contacts: {variant.upper()}", font=font, fill=color
    )
    draw.text(
        (20, 40),
        f"t={sim_time:05.2f}s    cube lift={height * 100:+.1f} cm    contacts={contacts}",
        font=font,
        fill="white",
    )
    return np.asarray(canvas)


def replay(env, switch, actions, variant, output_dir, record):
    import imageio.v2 as imageio

    enabled = variant == "on"
    switch.set(env, enabled)
    initial_obs, _ = env.reset()
    switch.set(env, enabled)
    cube_start = env._cube_position().copy()
    writer = None
    if record:
        writer = imageio.get_writer(
            output_dir / f"skin_{variant}.mp4",
            fps=20,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
    cube_positions, joint_positions, contact_counts, skin_counts = [], [], [], []
    raw_depths, speeds = [], []
    start = time.perf_counter()
    try:
        for index, action in enumerate(actions):
            if env.render_mode == "human" and not env.scene.viewer.is_alive():
                break
            env._command_qpos = action.copy()
            env.robot.control_dofs_position(action)
            env._physics_step()
            pos = env._cube_position()
            cube_positions.append(pos.copy())
            joint_positions.append(_as_numpy(env.robot.get_qpos()))
            contacts = env.cube.get_contacts(with_entity=env.robot)
            a = _as_numpy(contacts["geom_a"]).astype(int)
            b = _as_numpy(contacts["geom_b"]).astype(int)
            count = len(a)
            skin_count = int(
                np.count_nonzero(
                    np.isin(a, switch.indices) | np.isin(b, switch.indices)
                )
            )
            if not enabled and skin_count:
                raise RuntimeError("Disabled skin colliders still generated contact")
            contact_counts.append(count)
            skin_counts.append(skin_count)
            raw_depths.append(
                float(np.max(_as_numpy(contacts["penetration"]), initial=0))
            )
            speeds.append(
                float(np.linalg.norm(_as_numpy(env.cube.get_dofs_velocity())[:3]))
            )
            if writer is not None and (index % 20 == 0 or index == len(actions) - 1):
                writer.append_data(
                    label_frame(
                        env.render(),
                        variant,
                        (index + 1) * 0.0025,
                        pos[2] - cube_start[2],
                        count,
                    )
                )
            if index % 2000 == 0:
                print(
                    f"{variant}: {index}/{len(actions)} steps; cube z={pos[2]:.4f}",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()
    if not cube_positions:
        raise RuntimeError("Viewer closed before replay began")
    if env.render_mode != "human":
        np.savez_compressed(
            output_dir / f"skin_{variant}_trace.npz",
            cube_positions=cube_positions,
            joint_positions=joint_positions,
            contacts=contact_counts,
            skin_contacts=skin_counts,
        )
    result = {
        "completed_steps": len(cube_positions),
        "action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
        "initial_observation": initial_obs.tolist(),
        "inertia_sha256": switch.inertial_signature(env),
        "final_cube_position": cube_positions[-1].tolist(),
        "lift_height_m": float(cube_positions[-1][2] - cube_start[2]),
        "final_contacts": contact_counts[-1],
        "final_skin_contacts": skin_counts[-1],
        "peak_skin_contacts": max(skin_counts),
        "max_collision_depth_m": max(raw_depths),
        "final_cube_speed_m_s": speeds[-1],
        "finite": bool(
            np.isfinite(cube_positions).all() and np.isfinite(joint_positions).all()
        ),
        "elapsed_wall_seconds": time.perf_counter() - start,
    }
    print(
        json.dumps(
            {
                variant: {
                    key: val
                    for key, val in result.items()
                    if key != "initial_observation"
                }
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def combine_videos(output_dir):
    import imageio.v2 as imageio

    left = imageio.get_reader(output_dir / "skin_off.mp4")
    right = imageio.get_reader(output_dir / "skin_on.mp4")
    try:
        with imageio.get_writer(
            output_dir / "comparison.mp4",
            fps=20,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        ) as writer:
            for a, b in zip(left, right, strict=True):
                writer.append_data(np.concatenate([a, b], axis=1))
    finally:
        left.close()
        right.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("renders/skin_collision_ablation")
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--view",
        choices=("off", "on"),
        help="Replay a saved action trace in the interactive viewer, then hold the final target.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    actions_path = args.output_dir / "actions.npz"
    if args.view and not actions_path.is_file():
        parser.error("Run the comparison first to generate actions.npz")
    mode = "human" if args.view else (None if args.no_video else "rgb_array")
    env = BimanualOrcaPandaCubeGenesisEnv(render_mode=mode)
    try:
        switch = SkinContactSwitch(env)
        print(
            f"Identified {len(switch.indices)} added skin geoms; toggle affects {switch.disabled_pairs} pairs.",
            flush=True,
        )
        if env.camera is not None:
            env.camera.set_pose(pos=(1.05, -0.27, 0.79), lookat=(0.49, 0.17, 0.41))
        if args.view:
            with np.load(actions_path) as saved:
                actions = saved["actions"]
            replay(env, switch, actions, args.view, args.output_dir, False)
            print(
                "Replay complete; orbit/zoom to inspect. Close the window to exit.",
                flush=True,
            )
            while env.scene.viewer.is_alive():
                env.scene.step()
            return 0

        # Generate the known successful trajectory once. The two measured
        # trials below replay its exact targets; neither uses adaptive closure.
        actions = []
        original_step = env._physics_step

        def capture_action():
            actions.append(env._command_qpos.copy())
            original_step()

        env._physics_step = capture_action
        print(
            "Capturing the reference command stream with skin contacts ON...",
            flush=True,
        )
        try:
            reference = env.run_scripted_grasp()
        finally:
            env._physics_step = original_step
        if not reference.success:
            raise RuntimeError(f"Reference grasp failed: {reference}")
        actions = np.asarray(actions)
        np.savez_compressed(actions_path, actions=actions, dt=0.0025)
        results = {
            variant: replay(
                env, switch, actions, variant, args.output_dir, not args.no_video
            )
            for variant in ("on", "off")
        }
        np.testing.assert_array_equal(
            results["on"]["initial_observation"], results["off"]["initial_observation"]
        )
        np.testing.assert_allclose(
            results["on"]["final_cube_position"],
            reference.final_cube_position,
            atol=1e-6,
            rtol=0,
        )
        if not all(result["finite"] for result in results.values()):
            raise RuntimeError("Nonfinite state in comparison")
        report = {
            "genesis_version": env.gs.__version__,
            "backend": "cpu",
            "precision": "64",
            "method": "Same built scene; only added right-hand skin collision-pair validity is toggled. Identical recorded position targets, initial state, mass/inertia, geometry, friction, cube contact skin and solver settings. Left hand unchanged.",
            "scope": "Marginal effect of added right-hand skin contacts in the current stack; not a comparison of raw original assets or engines. Recorded targets were generated by the skin-enabled controller.",
            "penetration_caveat": "Solver contact depth is not visual penetration. Removing skin contacts makes the environment's contact-based visual-overlap estimate unsuitable for comparing these trials.",
            "skin_geom_names": switch.names,
            "disabled_pair_count": switch.disabled_pairs,
            "reference": reference.to_dict(),
            "trials": results,
        }
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        switch.set(env, True)
        if not args.no_video:
            combine_videos(args.output_dir)
        print(f"COMPARISON_READY: {args.output_dir.resolve()}", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
