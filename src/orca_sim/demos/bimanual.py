"""Native bimanual scene: independent arm IK and hand motion, or a live home viewer.

This is an assembly/control demonstration, not a coordinated two-hand grasp.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
import mujoco

from orca_sim.mujoco_bimanual import OrcaBimanualPandaCubeStackingMujoco, SIDES


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def motion_paths(env):
    """Short constant-orientation Cartesian paths; all IK uses scratch data."""
    result = {}
    for side, direction in (("right", -1), ("left", 1)):
        robot = env.robots[side]
        start = env.data.site(robot.hand_site).xpos.copy()
        rotation = env.data.site(robot.hand_site).xmat.reshape(3, 3).copy()
        seed = env.data.qpos[robot.arm_qadr].copy()
        path = [seed]
        for fraction in np.linspace(0, 1, 41)[1:]:
            seed = env.solve_ik(
                side,
                start + fraction * np.array([direction * 0.025, 0.09, 0.045]),
                rotation=rotation,
                seed=seed,
            )
            path.append(seed)
        result[side] = np.array(path)
    return result


def motion_action(env, home, paths, t):
    action = home.copy()
    weights = {
        "right": smooth((t - 1) / 2) * (1 - smooth((t - 9) / 2)),
        "left": smooth((t - 3) / 2) * (1 - smooth((t - 9) / 2)),
    }
    closure = smooth((t - 5) / 2) * (1 - smooth((t - 7) / 2))
    for side in SIDES:
        robot = env.robots[side]
        indices = robot.actuator_indices
        path = paths[side]
        action[indices[:7]] = [
            np.interp(weights[side], np.linspace(0, 1, len(path)), path[:, j])
            for j in range(7)
        ]
        for i, name in zip(indices[7:], robot.actuator_joints[7:]):
            if name.endswith(("-mcp", "-pip")):
                action[i] = home[i] * (1 - closure) + 0.7 * closure
    phase = (
        "HOLD HOME"
        if t < 1
        else "RIGHT ARM IK"
        if t < 3
        else "LEFT ARM IK"
        if t < 5
        else "BOTH HANDS CLOSE"
        if t < 7
        else "BOTH HANDS OPEN"
        if t < 9
        else "RETURN HOME"
        if t < 11
        else "HOLD HOME"
    )
    return action, phase


def run(*, output_dir, video=False, viewer=False, hold=False, duration=12.0):
    if video and viewer:
        raise ValueError("Choose --video or --viewer")
    if not np.isfinite(duration) or duration < 12:
        raise ValueError("Motion validation needs at least 12 seconds")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = OrcaBimanualPandaCubeStackingMujoco(
        render_mode="human" if viewer else "rgb_array" if video else None,
        frame_skip=5,
    )
    writer = None
    try:
        m, d = env.model, env.data
        home = d.ctrl.copy()
        hand_start = {s: d.site(env.robots[s].hand_site).xpos.copy() for s in SIDES}
        maximum_motion = dict.fromkeys(SIDES, 0.0)
        paths = None if hold else motion_paths(env)
        (output_dir / "scene.xml").write_text(env.xml)
        if video:
            import imageio.v2 as imageio
            from PIL import Image

            Image.fromarray(env.render()).save(output_dir / "home.png")
            writer = imageio.get_writer(
                output_dir / "rollout.mp4", fps=20, codec="libx264", quality=8
            )
        max_inter = max_table = max_limit = max_hand_torque = 0.0
        forces_zero = True
        next_frame = 0.0
        control_steps = 0
        last_phase = None
        dt = m.opt.timestep * env.frame_skip
        for _ in itertools.count():
            if not (viewer and hold) and d.time >= duration - 1e-9:
                break
            started = time.monotonic()
            action, phase = (
                (home, "HOLD HOME") if hold else motion_action(env, home, paths, d.time)
            )
            if phase != last_phase:
                print(f"{d.time:5.1f}s | {phase}", flush=True)
                last_phase = phase
            env.step(action)
            control_steps += 1
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                raise RuntimeError("Non-finite simulation state")
            contacts = env.contact_metrics()
            max_inter = max(max_inter, contacts["inter_robot_depth_m"])
            max_table = max(max_table, contacts["robot_table_depth_m"])
            for s, robot in env.robots.items():
                maximum_motion[s] = max(
                    maximum_motion[s],
                    float(np.linalg.norm(d.site(robot.hand_site).xpos - hand_start[s])),
                )
                max_hand_torque = max(
                    max_hand_torque,
                    float(np.max(np.abs(d.actuator_force[robot.actuator_indices[7:]]))),
                )
            q = d.qpos[env.qadr]
            max_limit = max(
                max_limit,
                float(np.max(env.action_space.low - q)),
                float(np.max(q - env.action_space.high)),
            )
            forces_zero &= not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied)
            if writer is not None and d.time >= next_frame:
                from PIL import Image, ImageDraw, ImageFont

                frame = Image.fromarray(env.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 1280, 80), fill=(20, 24, 31))
                font = ImageFont.load_default(size=22)
                draw.text(
                    (16, 9),
                    f"NATIVE MUJOCO | 2 PANDA + LEFT/RIGHT ORCA | {phase}",
                    font=font,
                )
                draw.text(
                    (16, 43),
                    f"t={d.time:.1f}s | 48 joint targets | Inter-robot contact depth {max_inter * 1000:.3f} mm",
                    font=font,
                )
                writer.append_data(np.asarray(frame))
                next_frame += 1 / 20
            if viewer:
                env.render()
                if not env._viewer.is_running():
                    break
                time.sleep(max(0.0, dt - (time.monotonic() - started)))
        warnings = {str(i): int(w.number) for i, w in enumerate(d.warning) if w.number}
        result = dict(
            engine="MuJoCo",
            version=mujoco.__version__,
            duration_s=float(d.time),
            control_steps=control_steps,
            physics_timestep_s=float(m.opt.timestep),
            actuators=int(m.nu),
            home=env.start_pose,
            object_layout=env.object_layout,
            max_hand_displacement_m=maximum_motion,
            max_inter_robot_depth_m=max_inter,
            max_robot_table_depth_m=max_table,
            max_joint_limit_violation_rad=max_limit,
            max_hand_torque_Nm=max_hand_torque,
            external_forces_zero=bool(forces_zero),
            equalities=int(m.neq),
            warnings=warnings,
            task="home_hold" if hold else "independent_bimanual_motion",
            success=bool(
                (hold or d.time >= duration - 1e-8)
                and max_inter < 0.0001
                and max_table < 0.0001
                and max_limit < 0.03
                and max_hand_torque <= 1.0 + 1e-10
                and forces_zero
                and m.neq == 0
                and not warnings
                and (hold or all(v > 0.07 for v in maximum_motion.values()))
            ),
        )
        (output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        return result
    finally:
        if writer is not None:
            writer.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("renders/mujoco_bimanual")
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--hold",
        action="store_true",
        help="Hold home; with --viewer, run until the window is closed",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = run(
        output_dir=args.output_dir, video=args.video, viewer=args.viewer, hold=args.hold
    )
    if args.check and not result["success"]:
        raise SystemExit("Bimanual validation failed; see report.json")


if __name__ == "__main__":
    main()
