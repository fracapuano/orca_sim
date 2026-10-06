#!/usr/bin/env python3
"""Approach, grasp and lift the stacking scene's unmodified red cube in MuJoCo."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from orca_sim.mujoco_panda import (
    CUBE_HALF_SIZE,
    PACKAGE,
    TOWER_ROTATION,
    OrcaPandaCubeStackingMujoco,
    inverse_kinematics,
)
from standalone_hand_grasp import skin_geoms, skin_vertex_depth
from orca_sim.panda_stacking import StackController, stack_metrics
from orca_sim.palm_grasp import PalmFirstGrasp


def smooth(t):
    x = np.clip(t, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def finger_targets(env, fraction):
    result = np.zeros(env.model.nu)
    for i, name in enumerate(env.actuator_joints):
        if name.endswith(
            ("i-mcp", "i-pip", "m-mcp", "m-pip", "r-mcp", "r-pip", "p-mcp", "p-pip")
        ):
            result[i] = 1.45 * fraction
        for joint, value in {
            "t-cmc": 0.55,
            "t-abd": 0.0,
            "t-mcp": 0.3,
            "t-pip": 0.8,
        }.items():
            if name == "right_" + joint:
                result[i] = value * fraction
    return np.clip(result, env.action_space.low, env.action_space.high)


def translation_approach(env, destination, rotation):
    """Continue the saved home IK branch along a constant-orientation path."""
    start = env.data.site("hand_frame").xpos.copy()
    seed = env.data.qpos[env.arm_qadr].copy()
    path = [seed.copy()]
    for amount in np.linspace(0, 1, 81)[1:]:
        seed = inverse_kinematics(
            env, start * (1 - amount) + destination * amount, rotation, seed=seed
        )
        path.append(seed.copy())
    return np.array(path)


def cube_contacts(env, cube_name="red_cube"):
    m, d = env.model, env.data
    cube = m.geom(cube_name + "_geom").id
    hand, table, depth, force = 0, 0, 0.0, 0.0
    palm_top, palm_load = 0, 0.0
    digits = set()
    wrench = np.zeros(6)
    for contact_id, c in enumerate(d.contact):
        if cube not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == cube else c.geom1
        body = int(m.geom_bodyid[other])
        if m.body(body).name.startswith("right_"):
            depth = max(depth, -float(c.dist))
            mujoco.mj_contactForce(m, d, contact_id, wrench)
            if wrench[0] > 1e-4:
                hand += 1
                force += wrench[0]
                mesh = int(m.geom_dataid[other])
                if mesh >= 0 and m.mesh(mesh).name == "right_R-Carpals":
                    local_point = (c.pos - d.geom_xpos[cube]) @ d.geom_xmat[
                        cube
                    ].reshape(3, 3)
                    normal = c.frame[:3] @ d.geom_xmat[cube].reshape(3, 3)
                    if (
                        abs(local_point[2] - m.geom_size[cube, 2]) < 0.002
                        and abs(normal[2]) > 0.8
                    ):
                        palm_top += 1
                        palm_load += float(wrench[0])
                while body:
                    for j in range(
                        m.body_jntadr[body], m.body_jntadr[body] + m.body_jntnum[body]
                    ):
                        name = m.joint(j).name
                        if name.startswith("right_") and "-" in name:
                            digits.add(name.split("_")[-1].split("-")[0])
                    body = int(m.body_parentid[body])
        elif other == m.geom("tabletop").id:
            table += 1
    return dict(
        hand_contacts=hand,
        table_contacts=table,
        contact_depth_m=depth,
        normal_load_N=float(force),
        digits=sorted(digits),
        palm_top_contacts=palm_top,
        palm_top_load_N=palm_load,
    )


def run(
    *,
    output_dir,
    video=False,
    viewer=False,
    offset=None,
    grip=None,
    preshape=0.0,
    duration=None,
    horizontal=True,
    yaw_degrees=None,
    start_pose=None,
    wrist=None,
    stack=False,
    sideways=False,
    palm_first=False,
):
    sideways = sideways or palm_first
    if duration is None:
        duration = (
            (50.0 if stack else 28.0) if palm_first else (40.0 if stack else 20.0)
        )
    grip = (1.0 if palm_first else 0.65) if grip is None else grip
    wrist = (0.06 if palm_first else 0.3) if wrist is None else wrist
    if stack and duration < 39:
        raise ValueError("Stacking requires at least 39 seconds")
    if palm_first and duration < (49 if stack else 27):
        raise ValueError(
            "Palm-first runs need at least 49 s for stacking or 27 s for grasping"
        )
    if video and viewer:
        raise ValueError("Use --video or --viewer separately")
    if sideways and (not horizontal or yaw_degrees is not None):
        raise ValueError(
            "Sideways mode uses the saved home orientation, not yaw/vertical overrides"
        )
    if offset is None:
        offset = (
            (0.01, 0.15, -0.02 - CUBE_HALF_SIZE)
            if palm_first
            else ((-0.03, 0.1762, -0.10) if sideways else (-0.04, 0.1762, -0.09))
        )
    if palm_first and preshape != 0:
        raise ValueError("Palm-first contact requires initially open fingers")
    if not (0 <= preshape <= 1 and 0 <= grip <= 1):
        raise ValueError("Grip and preshape fractions must be between zero and one")
    if not np.isfinite(wrist) or not -1.1344640137963142 <= wrist <= 0.6108652381980153:
        raise ValueError("Wrist target must lie in the native joint range")
    if np.asarray(offset).shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("offset must contain three finite coordinates")
    if (
        (yaw_degrees is not None and not np.isfinite(yaw_degrees))
        or not np.isfinite(duration)
        or duration < 18
    ):
        raise ValueError("Use a finite yaw and a duration of at least 18 seconds")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = OrcaPandaCubeStackingMujoco(
        render_mode="human" if viewer else "rgb_array" if video else None,
        frame_skip=1,
        start_pose=start_pose,
    )
    m, d = env.model, env.data
    (output_dir / "scene.xml").write_text(env.xml)
    cube_start = d.body("red_cube").xpos.copy()
    home_rotation = d.site("hand_frame").xmat.reshape(3, 3).copy()
    if yaw_degrees is None:
        yaw_degrees = float(np.rad2deg(env.start_pose["base_yaw"]) - 90.0)
    yaw = np.deg2rad(yaw_degrees)
    rotation = (
        np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        if horizontal
        else TOWER_ROTATION
    )
    if sideways:
        rotation = home_rotation.copy()
        yaw_degrees = float(np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0])))
    grasp = cube_start - rotation @ np.array(offset)
    pre = grasp + [0, 0, 0.16]
    home = d.qpos[env.arm_qadr].copy()
    home_hand = d.ctrl[7:].copy()
    # Select the elbow branch that leaves room for the later lateral transfer.
    # This is only a scratch-data IK seed, never the robot's initial qpos.
    approach = translation_approach(env, pre, rotation) if sideways else None
    approach_amounts = np.linspace(0, 1, len(approach)) if sideways else None
    approach_seed = np.array([1.4, -0.9, -1.7, -2.5, 2.7, 2.3, 0.9])
    pre_q = (
        approach[-1]
        if sideways
        else inverse_kinematics(env, pre, rotation=rotation, seed=approach_seed)
    )
    # Precompute a Cartesian path. IK only updates scratch data, never live qpos.
    zs = np.linspace(-0.03, 0.26, 146) if palm_first else np.linspace(0.0, 0.18, 91)
    solutions = []
    seed = pre_q
    for z in reversed(zs):
        seed = inverse_kinematics(env, grasp + [0, 0, z], seed=seed, rotation=rotation)
        solutions.append(seed)
    solutions = np.array(solutions[::-1])

    def arm_at(z):
        return np.array([np.interp(z, zs, solutions[:, j]) for j in range(7)])

    writer = None
    if video:
        import imageio.v2 as imageio
        from PIL import Image

        Image.fromarray(env.render()).save(output_dir / "home.png")
        writer = imageio.get_writer(
            output_dir / "rollout.mp4", fps=30, codec="libx264", quality=8
        )
    trace = []
    settled_checks = []
    trace_interval = max(1, round(0.02 / m.opt.timestep))
    skins = skin_geoms(m)
    cube_geom = m.geom("red_cube_geom").id
    blue_geom = m.geom("blue_cube_geom").id
    max_depth = max_skin = 0.0
    max_table_depth = max_arm_table_depth = max_joint_violation = 0.0
    forces_zero = True
    table_geoms = {m.geom("tabletop").id}
    max_arm_torque, max_hand_torque = 0.0, 0.0
    next_frame = 0.0
    placement = None
    palm = PalmFirstGrasp(m.opt.timestep, grip) if palm_first else None
    command_hash = hashlib.sha256()
    last_phase = None
    try:
        for step in range(round(duration / m.opt.timestep)):
            started = time.monotonic()
            t = d.time
            if t < 1:
                phase, arm, fraction = "SETTLE", home, 0.0
            elif t < 5:
                phase = "APPROACH"
                s = smooth((t - 1) / 4)
                arm, fraction = home * (1 - s) + pre_q * s, preshape * s
                if sideways:
                    arm = np.array(
                        [
                            np.interp(s, approach_amounts, approach[:, j])
                            for j in range(7)
                        ]
                    )
            elif t < 9:
                phase, fraction = "DESCEND", preshape
                arm = arm_at(0.16 * (1 - smooth((t - 5) / 4)))
            elif t < 12:
                phase, arm = "CLOSE", arm_at(0)
                fraction = preshape + (grip - preshape) * smooth((t - 9) / 3)
            elif t < 16:
                phase, fraction = "LIFT", grip
                arm = arm_at(0.18 * smooth((t - 12) / 4))
            else:
                phase, fraction, arm = "HOLD", grip, arm_at(0.18)
            if palm_first and t >= 5:
                phase, z, fraction = palm.command(t, cube_contacts(env))
                arm = arm_at(z)
            action = finger_targets(env, fraction)
            # Abduct the thumb during descent so it passes beside the cube.
            # Closing it only after reaching grasp height avoids pushing the
            # free cube sideways with an extended thumb before finger closure.
            thumb_abd = env.actuator_joints.index("right_t-abd")
            action[thumb_abd] = 0.9 * (1 - smooth((t - 9) / 3))
            if palm_first:
                action[thumb_abd] = (
                    0.9 * (1 - smooth((t - palm.close_time) / 3))
                    if palm.close_time is not None
                    else 0.9
                )
            if t < 1:
                action[7:] = home_hand
            elif t < 5:
                action[7:] = home_hand * (1 - s) + finger_targets(env, preshape)[7:] * s
                action[thumb_abd] = home_hand[thumb_abd - 7] * (1 - s) + 0.9 * s
            if t >= 1:
                w = smooth((t - 1) / 4)
                action[env.actuator_joints.index("right_wrist")] = (
                    home_hand[0] * (1 - w) + wrist * w
                )
                if palm_first:
                    thumb_cmc = env.actuator_joints.index("right_t-cmc")
                    closure = (
                        smooth((t - palm.close_time) / 3)
                        if palm.close_time is not None
                        else 0.0
                    )
                    thumb_target = -0.5 * (1 - closure) + 0.55 * grip * closure
                    action[thumb_cmc] = (
                        home_hand[thumb_cmc - 7] * (1 - w) + thumb_target * w
                    )
                    if palm.close_time is not None:
                        spread = smooth((t - palm.close_time) / 0.6)
                        action[env.actuator_joints.index("right_p-abd")] = -0.4 * spread
                        action[env.actuator_joints.index("right_r-abd")] = (
                            -0.15 * spread
                        )
            action[:7] = arm
            if stack and t >= (palm.stack_time if palm_first else 20):
                if placement is None:
                    placement = StackController(env, keep_grasp_orientation=sideways)
                action, phase = placement.action()
            if phase != last_phase:
                print(f"{t:5.1f}s | {phase}", flush=True)
                last_phase = phase
            command_hash.update(action.astype("<f8").tobytes())
            env.step(action)
            metrics = cube_contacts(env)
            if stack and d.time >= duration - 3.0:
                settled_checks.append({**metrics, **stack_metrics(env)})
            max_depth = max(
                max_depth,
                metrics["contact_depth_m"],
                cube_contacts(env, "blue_cube")["contact_depth_m"],
            )
            max_arm_torque = max(
                max_arm_torque, float(np.max(np.abs(d.actuator_force[:7])))
            )
            max_hand_torque = max(
                max_hand_torque, float(np.max(np.abs(d.actuator_force[7:])))
            )
            robot_q = d.qpos[env.qadr]
            joint_ranges = m.jnt_range[m.actuator_trnid[:, 0]]
            max_joint_violation = max(
                max_joint_violation,
                float(np.max(joint_ranges[:, 0] - robot_q)),
                float(np.max(robot_q - joint_ranges[:, 1])),
            )
            forces_zero &= not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied)
            for c in d.contact:
                if c.geom1 in table_geoms or c.geom2 in table_geoms:
                    other = c.geom2 if c.geom1 in table_geoms else c.geom1
                    name = m.body(m.geom_bodyid[other]).name
                    if name.startswith("right_"):
                        max_table_depth = max(max_table_depth, -float(c.dist))
                    elif name.startswith("link"):
                        max_arm_table_depth = max(max_arm_table_depth, -float(c.dist))
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                raise RuntimeError("Non-finite simulation state")
            if step % trace_interval == 0:
                skin = max(
                    skin_vertex_depth(m, d, cube_geom, skins),
                    skin_vertex_depth(m, d, blue_geom, skins),
                )
                max_skin = max(max_skin, skin)
                trace.append(
                    dict(
                        t=float(d.time),
                        phase=phase,
                        cube_pos=d.body("red_cube").xpos.tolist(),
                        hand_pos=d.site("hand_frame").xpos.tolist(),
                        hand_frame_rotation_deg=float(
                            np.rad2deg(
                                np.arccos(
                                    np.clip(
                                        (
                                            np.trace(
                                                d.site("hand_frame").xmat.reshape(3, 3)
                                                @ home_rotation.T
                                            )
                                            - 1
                                        )
                                        / 2,
                                        -1,
                                        1,
                                    )
                                )
                            )
                        ),
                        ctrl=d.ctrl.tolist(),
                        qpos=d.qpos.tolist(),
                        qvel=d.qvel.tolist(),
                        skin_depth_m=skin,
                        **metrics,
                        **stack_metrics(env),
                    )
                )
            if writer is not None and d.time >= next_frame:
                from PIL import Image, ImageDraw, ImageFont

                frame = Image.fromarray(env.render())
                # A second real render, not a different simulation or replay.
                # Keep the whole robot visible and show the contact region too.
                if d.time >= 5:
                    saved_camera = (
                        env.camera.lookat.copy(),
                        env.camera.distance,
                        env.camera.azimuth,
                        env.camera.elevation,
                    )
                    env.camera.lookat[:] = d.body("red_cube").xpos + [
                        0.0,
                        0.0,
                        m.geom("red_cube_geom").size[2],
                    ]
                    env.camera.distance, env.camera.azimuth, env.camera.elevation = (
                        0.48,
                        155 + yaw_degrees
                        if sideways
                        else 65 + np.rad2deg(env.start_pose["base_yaw"]),
                        -12,
                    )
                    detail = Image.fromarray(env.render()).resize((480, 270))
                    frame.paste(detail, (784, 434))
                    env.camera.lookat[:] = saved_camera[0]
                    env.camera.distance, env.camera.azimuth, env.camera.elevation = (
                        saved_camera[1:]
                    )
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 1280, 86), fill=(20, 24, 31))
                font = ImageFont.load_default(size=23)
                draw.text(
                    (16, 10),
                    f"NATIVE MUJOCO | Panda + fixed ORCA MJCF | {phase} | t={d.time:.1f}s",
                    font=font,
                )
                height = d.body("red_cube").xpos[2] - cube_start[2]
                draw.text(
                    (16, 46),
                    (
                        f"Palm-top contacts {metrics['palm_top_contacts']} | palm load {metrics['palm_top_load_N']:.1f} N | cube lift {height * 100:.1f} cm"
                        if palm_first
                        else f"Cube lift {height * 100:.1f} cm | hand contacts {metrics['hand_contacts']} | skin overlap {max_skin * 1000:.2f} mm"
                    ),
                    font=font,
                )
                writer.append_data(np.asarray(frame))
                next_frame += 1 / 30
            if viewer:
                if step % 8 == 0:
                    env.render()
                    if not env._viewer.is_running():
                        break
                time.sleep(max(0, m.opt.timestep - (time.monotonic() - started)))
        hold = [
            s
            for s in trace
            if (palm.hold_time + 1 if palm_first else 17)
            <= s["t"]
            <= (palm.stack_time if palm_first else 20)
        ]
        min_lift = min((s["cube_pos"][2] - cube_start[2] for s in hold), default=0.0)
        result = dict(
            engine="MuJoCo",
            start_pose=env.start_pose,
            version=mujoco.__version__,
            timestep_s=float(m.opt.timestep),
            initial_objects={
                "red_cube": cube_start.tolist(),
                "blue_cube": m.body("blue_cube").pos.tolist(),
                "stack_target": m.body("stack_target").pos.tolist(),
            },
            duration_s=float(d.time),
            offset=list(offset),
            grip=grip,
            preshape=preshape,
            wrist_target_rad=wrist,
            task="stack" if stack else "grasp_lift",
            horizontal=horizontal,
            sideways=sideways,
            palm_first=palm_first,
            grasp_rotation=rotation.tolist(),
            yaw_degrees=yaw_degrees,
            max_approach_hand_frame_rotation_deg=max(
                s["hand_frame_rotation_deg"] for s in trace if s["t"] <= 9
            ),
            cube_size_m=(2 * m.geom_size[cube_geom]).tolist(),
            cube_mass_kg=float(m.body_mass[m.body("red_cube").id]),
            final_lift_m=float(d.body("red_cube").xpos[2] - cube_start[2]),
            minimum_hold_lift_m=float(min_lift),
            max_hand_cube_contact_depth_m=max_depth,
            max_skin_vertex_depth_m=max_skin,
            max_hand_table_depth_m=max_table_depth,
            max_arm_table_depth_m=max_arm_table_depth,
            max_joint_limit_violation_rad=max_joint_violation,
            max_arm_torque_Nm=max_arm_torque,
            max_hand_torque_Nm=max_hand_torque,
            equalities=int(m.neq),
            external_forces_zero=bool(forces_zero),
            command_sha256=command_hash.hexdigest(),
            scene_sha256=hashlib.sha256(env.xml.encode()).hexdigest(),
            code_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    Path(__file__).resolve(),
                    PACKAGE / "mujoco_panda.py",
                    PACKAGE / "panda_stacking.py",
                    PACKAGE / "palm_grasp.py",
                )
            },
            warnings={
                str(i): int(w.number) for i, w in enumerate(d.warning) if w.number
            },
            source_sha256={
                str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    PACKAGE / "models/v2/mjcf/orcahand_right.mjcf",
                    PACKAGE / "models/v2/mjcf/orcahand_right_body.xml",
                    PACKAGE / "scenes/cube_stacking.xml",
                    PACKAGE / "assets/mujoco/tabletop_layout.json",
                ]
            },
            success=bool(
                min_lift > 0.10
                and max_depth < 0.002
                and max_skin < 0.001
                and max_table_depth < 0.001
                and max_arm_table_depth < 0.001
                and max_joint_violation < 0.03
                and forces_zero
                and not any(w.number for w in d.warning)
                and hold
                and all(
                    s["hand_contacts"] >= 2
                    and s["table_contacts"] == 0
                    and "t" in s["digits"]
                    and len(s["digits"]) >= 2
                    for s in hold
                )
            ),
        )
        if palm_first:
            result["palm_first_contact_time_s"] = palm.first_contact_time
            result["finger_closure_start_time_s"] = palm.close_time
            result["palm_rest_duration_s"] = palm.rest_duration
            result["peak_palm_top_load_N"] = palm.peak_palm_load
            result["palm_rest_force_range_N"] = [
                palm.rest_min_force,
                palm.rest_max_force,
            ]
            result["palm_before_closure_verified"] = bool(
                palm.close_time is not None
                and palm.first_contact_time is not None
                and palm.close_time >= palm.first_contact_time + 0.6 - 1e-8
                and palm.rest_duration >= 0.6
            )
            result["success"] = (
                result["success"] and result["palm_before_closure_verified"]
            )
        if stack:
            settled = settled_checks
            result["grasp_success"] = result["success"]
            result["stack_success"] = bool(
                settled
                and all(
                    s["stack_contacts"] >= 1
                    and s["blue_table_contacts"] >= 1
                    and s["hand_contacts"] == 0
                    and s["table_contacts"] == 0
                    and s["stack_xy_error_m"] < 0.015
                    and abs(
                        s["stack_height_m"]
                        - m.geom("red_cube_geom").size[2]
                        - m.geom("blue_cube_geom").size[2]
                    )
                    < 0.005
                    and s["cube_linear_speed_m_s"] < 0.01
                    and s["cube_angular_speed_rad_s"] < 0.1
                    and s["stack_contact_depth_m"] < 0.001
                    for s in settled
                )
            )
            result["success"] = result["grasp_success"] and result["stack_success"]
            result["final_stack"] = stack_metrics(env)
            result["stack_face_rotation"] = placement.target_rotation.tolist()
            result["placement_orientation_mode"] = (
                "keep_grasp_orientation" if sideways else "level_cube_face"
            )
            result["placement_height_m"] = placement.placement_height(
                placement.target_rotation
            )
            result["retreat_vector_m"] = placement.retreat_vector.tolist()
            result["post_release_validation_seconds"] = 3.0
            result["post_release_validation_samples"] = len(settled)
            result["post_release_max_linear_speed_m_s"] = max(
                s["cube_linear_speed_m_s"] for s in settled
            )
            result["post_release_max_angular_speed_rad_s"] = max(
                s["cube_angular_speed_rad_s"] for s in settled
            )
        (output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        (output_dir / "trace.json").write_text(json.dumps(trace) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        return result
    except Exception as exc:
        (output_dir / "trace.json").write_text(json.dumps(trace) + "\n")
        (output_dir / "failure.json").write_text(
            json.dumps(
                {
                    "error": str(exc),
                    "time_s": float(d.time),
                    "qpos": d.qpos.tolist(),
                    "ctrl": d.ctrl.tolist(),
                    "stack": stack_metrics(env),
                },
                indent=2,
            )
            + "\n"
        )
        raise
    finally:
        if writer is not None:
            writer.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("renders/mujoco_panda_grasp")
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--stack", action="store_true", help="Place red on blue, release and retreat"
    )
    parser.add_argument(
        "--start-pose", type=Path, help="JSON saved by the interactive pose editor"
    )
    parser.add_argument("--grip", type=float)
    parser.add_argument("--preshape", type=float, default=0.0)
    parser.add_argument("--wrist", type=float)
    parser.add_argument(
        "--palm-first",
        action="store_true",
        help="Rest the open palm on the cube before wrapping fingers; implies --sideways",
    )
    parser.add_argument(
        "--sideways",
        action="store_true",
        help="Keep the saved home hand-frame orientation and approach by translation",
    )
    parser.add_argument(
        "--vertical", action="store_false", dest="horizontal", default=True
    )
    parser.add_argument(
        "--yaw",
        type=float,
        help="World hand-frame yaw in degrees; default follows base yaw minus 90 degrees",
    )
    parser.add_argument(
        "--offset",
        nargs=3,
        type=float,
        help="Cube offset in the hand frame; default depends on grasp mode",
    )
    args = parser.parse_args()
    result = run(
        output_dir=args.output_dir,
        video=args.video,
        viewer=args.viewer,
        offset=args.offset,
        grip=args.grip,
        preshape=args.preshape,
        horizontal=args.horizontal,
        yaw_degrees=args.yaw,
        start_pose=args.start_pose,
        wrist=args.wrist,
        stack=args.stack,
        sideways=args.sideways,
        palm_first=args.palm_first,
    )
    if args.check and not result["success"]:
        raise SystemExit("Grasp validation failed; see report.json")


if __name__ == "__main__":
    main()
