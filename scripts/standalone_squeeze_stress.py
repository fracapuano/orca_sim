#!/usr/bin/env python3
"""Increase closure after grasp acquisition and measure cube slip in the palm.

The native MJCF supplies contact settings, force limits, friction and gains.
``--contact-model legacy`` reproduces the former soft-contact settings in memory.
Baseline acquisition is identical to standalone_hand_grasp.py. At 5 seconds,
targets ramp from scale 0.6 to the requested scale over two seconds. ``flexion``
changes only MCP/PIP targets; ``full`` also changes thumb opposition. Scale 1.0
is the existing closing-pose target, not 100% motor force or joint-limit travel.
``progressive`` reaches that strong pose at 15 s, all MCP/PIP upper target
limits at 25 s, then holds; use ``--duration 40``. It uses no preload adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

if __package__:
    from .standalone_hand_grasp import (
        default_cube_position,
        scene_xml,
        skin_geoms,
        skin_vertex_depth,
        targets,
    )
else:
    from standalone_hand_grasp import (
        default_cube_position,
        scene_xml,
        skin_geoms,
        skin_vertex_depth,
        targets,
    )

from orca_sim.versions import PACKAGE_ROOT
from orca_sim.mujoco_grasp import GraspPreload

START = 5.0
RAMP_SECONDS = 2.0
DURATION = 18.5
TRANSLATION_TOLERANCE = 0.002
ROTATION_TOLERANCE_DEG = 5.0
PROGRESSIVE_STRONG_TIME = 15.0
PROGRESSIVE_LIMIT_TIME = 25.0
PROGRESSIVE_DURATION = 40.0


def squeeze_target(model, side, profile, fraction):
    base = targets(model, side, 0.6)
    if profile in ("baseline", "preload", "progressive"):
        return base
    strong = targets(model, side, fraction)
    if profile == "flexion":
        for actuator in range(model.nu):
            joint = model.joint(int(model.actuator_trnid[actuator, 0])).name
            if not joint.endswith(("-mcp", "-pip")):
                strong[actuator] = base[actuator]
    elif profile != "full":
        raise ValueError(f"Unknown profile: {profile}")
    return strong


def progressive_target(model, side, t):
    """State-independent closure: nominal -> strong -> native joint limits.

    Only MCP/PIP targets increase. Wrist, abduction and thumb opposition stay
    fixed. No grasp feedback, preload scaling or extra torque cap is used.
    Larger targets are a demand, not a claim of increasing measured force.
    """
    if not np.isfinite(t) or t < 0:
        raise ValueError("Time must be finite and nonnegative")
    base = targets(model, side, 0.6)
    strong = squeeze_target(model, side, "flexion", 1.0)
    limit = strong.copy()
    for i, joint in enumerate(model.actuator_trnid[:, 0]):
        if model.joint(int(joint)).name.endswith(("-mcp", "-pip")):
            limit[i] = model.actuator_ctrlrange[i, 1]
    first = np.clip((t - START) / (PROGRESSIVE_STRONG_TIME - START), 0, 1)
    second = np.clip(
        (t - PROGRESSIVE_STRONG_TIME)
        / (PROGRESSIVE_LIMIT_TIME - PROGRESSIVE_STRONG_TIME),
        0,
        1,
    )
    return (base + first * (strong - base) + second * (limit - strong)) * np.clip(
        (t - 0.5) / 2, 0, 1
    )


def relative_pose(data, cube, palm):
    palm_rotation = data.xmat[palm].reshape(3, 3)
    position = palm_rotation.T @ (data.xpos[cube] - data.xpos[palm])
    rotation = palm_rotation.T @ data.xmat[cube].reshape(3, 3)
    return position, rotation


def pose_delta(position, rotation, reference):
    distance = float(np.linalg.norm(position - reference[0]))
    cosine = (np.trace(reference[1].T @ rotation) - 1) / 2
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    return distance, angle


def contact_metrics(model, data, side, cube_geom, support_geom):
    count, support_count, depth, total_normal = 0, 0, 0.0, 0.0
    digits = set()
    force = np.zeros(6)
    for index, contact in enumerate(data.contact):
        if cube_geom not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom1 if contact.geom2 == cube_geom else contact.geom2
        if other == support_geom:
            support_count += 1
            continue
        body = int(model.geom_bodyid[other])
        if not model.body(body).name.startswith(side + "_"):
            continue
        mujoco.mj_contactForce(model, data, index, force)
        total_normal += max(0.0, float(force[0]))
        depth = max(depth, -float(contact.dist))
        if force[0] <= 1e-6:
            continue
        count += 1
        while body:
            for joint in range(
                model.body_jntadr[body],
                model.body_jntadr[body] + model.body_jntnum[body],
            ):
                name = model.joint(joint).name
                if name.startswith(side + "_") and "-" in name:
                    digits.add(name.split("_")[-1].split("-")[0])
            body = int(model.body_parentid[body])
    return count, sorted(digits), support_count, depth, total_normal


def hand_contact_regions(model, side):
    """Classify collision geometry anatomically, without changing the model."""
    palm = int(model.jnt_bodyid[model.joint(f"{side}_wrist").id])
    regions = {}
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        if not model.body(body).name.startswith(side + "_"):
            continue
        region = "palm" if body == palm else "other_hand"
        while body and body != palm:
            for joint in range(
                model.body_jntadr[body],
                model.body_jntadr[body] + model.body_jntnum[body],
            ):
                name = model.joint(joint).name
                if name.startswith(side + "_t-"):
                    region = "thumb"
                elif any(name.startswith(side + "_" + digit + "-") for digit in "imrp"):
                    region = "fingers"
            body = int(model.body_parentid[body])
        regions[geom] = region
    return regions


def contact_region_loads(model, data, cube_geom, regions):
    """Positive normal force per hand region; only a measurement, not feedback."""
    loads = dict.fromkeys(("fingers", "thumb", "palm", "other_hand"), 0.0)
    force = np.zeros(6)
    for i, contact in enumerate(data.contact):
        if cube_geom not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == cube_geom else contact.geom1
        if other in regions:
            mujoco.mj_contactForce(model, data, i, force)
            loads[regions[other]] += max(0.0, float(force[0]))
    return loads


def run(
    side="right",
    variant="released",
    profile="flexion",
    fraction=1.0,
    output_dir=None,
    video=False,
    duration=DURATION,
    contact_model="native",
    multiplier=4.0,
    cube_pos=None,
    cube_mass=0.08,
):
    if not np.isfinite(fraction) or not 0.6 <= fraction <= 1.0:
        raise ValueError("Target scale must be between 0.6 and 1.0")
    if not np.isfinite(duration) or duration < START + RAMP_SECONDS + 1:
        raise ValueError("Record at least one second after the squeeze ramp")
    if variant not in ("released", "skin_disabled"):
        raise ValueError("Use current MJCF or original skin masks")
    if not np.isfinite(multiplier) or multiplier < 1:
        raise ValueError("Preload multiplier must be finite and at least one")
    if profile == "progressive" and duration < PROGRESSIVE_LIMIT_TIME + 5:
        raise ValueError(
            "Progressive closure needs at least 30 s to reach limits and hold"
        )
    source = PACKAGE_ROOT / "models" / "v2" / "mjcf" / f"orcahand_{side}_body.xml"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    xml = scene_xml(
        side=side,
        variant=variant,
        contact_model=contact_model,
        cube_pos=cube_pos,
        cube_mass=cube_mass,
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cube, cube_geom = model.body("cube").id, model.geom("cube_geom").id
    support, support_geom = model.body("support").id, model.geom("support_geom").id
    mocap = int(model.body_mocapid[support])
    support_start = model.body_pos[support].copy()
    palm = int(model.jnt_bodyid[model.joint(f"{side}_wrist").id])
    skins = skin_geoms(model)
    regions = hand_contact_regions(model, side)
    hand_geoms = {
        i
        for i in range(model.ngeom)
        if model.body(int(model.geom_bodyid[i])).name.startswith(side + "_")
    }
    max_step_penetration = 0.0
    max_step_motor_torque = 0.0
    command_hash = hashlib.sha256()
    finite_state = True
    initial_depth = skin_vertex_depth(model, data, cube_geom, skins)
    initial_contacts = contact_metrics(model, data, side, cube_geom, support_geom)
    if max(initial_depth, initial_contacts[3]) > 1e-5:
        raise ValueError("Initial fixture intersects hand")
    base = targets(model, side, 0.6)
    strong = squeeze_target(model, side, profile, fraction)
    name = f"{side}_{variant}_{profile}_{fraction:g}"
    preload = None
    renderer = writer = None
    samples, reference, next_frame = [], None, 0.0
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    if video:
        if output_dir is None:
            raise ValueError("Video needs an output directory")
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw, ImageFont

        renderer = mujoco.Renderer(model, width=960, height=720)
        writer = imageio.get_writer(
            output_dir / f"{name}.mp4", fps=30, codec="libx264", quality=8
        )
        font = ImageFont.load_default(size=22)
        small = ImageFont.load_default(size=18)
    try:
        for step in range(round(duration / model.opt.timestep)):
            t = step * model.opt.timestep
            mix = float(np.clip((t - START) / RAMP_SECONDS, 0, 1))
            data.ctrl[:] = (base * (1 - mix) + strong * mix) * np.clip(
                (t - 0.5) / 2, 0, 1
            )
            if profile == "progressive":
                data.ctrl[:] = progressive_target(model, side, t)
            if profile == "preload" and t >= START:
                if preload is None:
                    contacts = contact_metrics(
                        model, data, side, cube_geom, support_geom
                    )
                    if (
                        contacts[0] < 2
                        or "t" not in contacts[1]
                        or len(contacts[1]) < 2
                        or contacts[2]
                    ):
                        raise ValueError(
                            "Acquire an unsupported opposing-contact grasp before squeezing"
                        )
                    preload = GraspPreload(model, data, side)
                data.ctrl[:] = preload.command(1 + mix * (multiplier - 1))
            command_hash.update(np.asarray(data.ctrl, dtype="<f8").tobytes())
            data.mocap_pos[mocap] = support_start - [
                0,
                0,
                0.25 * np.clip((t - 3) / 0.5, 0, 1),
            ]
            mujoco.mj_step(model, data)
            # Check raw overlap/torque at every physics step, not only video or
            # 20 ms mesh-probe samples. mj_step contacts describe its start state.
            for contact in data.contact:
                if (contact.geom1 == cube_geom and contact.geom2 in hand_geoms) or (
                    contact.geom2 == cube_geom and contact.geom1 in hand_geoms
                ):
                    max_step_penetration = max(
                        max_step_penetration, -float(contact.dist)
                    )
            max_step_motor_torque = max(
                max_step_motor_torque, float(np.max(np.abs(data.actuator_force)))
            )
            finite_state &= bool(
                np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
            )
            if step % 10:
                continue
            mujoco.mj_forward(model, data)
            pos, rot = relative_pose(data, cube, palm)
            if reference is None and t >= START:
                reference = (pos.copy(), rot.copy())
            movement, rotation = (
                pose_delta(pos, rot, reference) if reference else (0.0, 0.0)
            )
            count, digits, support_count, penetration, normal = contact_metrics(
                model, data, side, cube_geom, support_geom
            )
            skin_depth = skin_vertex_depth(model, data, cube_geom, skins)
            sample = {
                "t": float(data.time),
                "cube_pos": data.xpos[cube].tolist(),
                "palm_relative_cube_pos": pos.tolist(),
                "palm_relative_cube_rotation": rot.tolist(),
                "translation_from_pre_squeeze_m": movement,
                "rotation_from_pre_squeeze_deg": rotation,
                "force_bearing_hand_contacts": count,
                "digits": digits,
                "support_contacts": support_count,
                "sum_hand_normal_force_N": normal,
                "normal_force_by_region_N": contact_region_loads(
                    model, data, cube_geom, regions
                ),
                "raw_contact_depth_m": penetration,
                "skin_vertex_depth_m": skin_depth,
                "ctrl": data.ctrl.tolist(),
                "actuator_force": data.actuator_force.tolist(),
                "qpos": data.qpos.tolist(),
                "qvel": data.qvel.tolist(),
                "saturated_motor_count": int(
                    np.count_nonzero(np.abs(data.actuator_force) >= 0.999)
                ),
                "applied_preload_multiplier": preload.applied_multiplier
                if preload
                else 1.0,
            }
            samples.append(sample)
            if writer and data.time >= next_frame:
                camera = mujoco.MjvCamera()
                # Identical camera schedule across trials; widen to include the
                # floor so a dropped cube does not just disappear out of view.
                zoom = float(np.clip((t - 8) / 4, 0, 1))
                camera.lookat[:] = [-0.01, 0.17, 0.395 - 0.115 * zoom]
                camera.distance, camera.azimuth, camera.elevation = (
                    0.29 + 0.56 * zoom,
                    0,
                    15 - 35 * zoom,
                )
                renderer.update_scene(data, camera=camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 960, 130), fill=(20, 24, 31))
                title = (
                    "PROGRESSIVE CLOSURE (NO ADAPTER)"
                    if profile == "progressive"
                    else "BALANCED PRELOAD SQUEEZE"
                    if profile == "preload"
                    else "UNCHANGED SQUEEZE"
                    if profile == "baseline"
                    else f"STRONG SQUEEZE ({profile})"
                )
                draw.text(
                    (14, 9),
                    f"{contact_model.upper()}{' / SKIN OFF' if variant == 'skin_disabled' else ''}: {title}",
                    fill="white",
                    font=font,
                )
                phase = (
                    "acquiring grasp"
                    if t < START
                    else "increasing closure"
                    if t < START + RAMP_SECONDS and profile != "baseline"
                    else "holding targets"
                )
                if profile == "progressive":
                    phase = (
                        "acquiring grasp"
                        if t < START
                        else "ramping to strong pose"
                        if t < PROGRESSIVE_STRONG_TIME
                        else "ramping to joint limits"
                        if t < PROGRESSIVE_LIMIT_TIME
                        else "holding maximum targets"
                    )
                draw.text(
                    (14, 39),
                    f"t={data.time:.2f}s | {phase} | +/-1 Nm"
                    + (
                        f" | preload {preload.applied_multiplier:.2f}x (request {multiplier:g}x)"
                        if preload
                        else f" | saturated motors {sample['saturated_motor_count']}"
                        if profile == "progressive"
                        else ""
                    ),
                    fill="white",
                    font=small,
                )
                draw.text(
                    (14, 66),
                    f"Motion relative to palm: {movement * 1000:.1f} mm / {rotation:.1f} deg (since 5s)",
                    fill="white",
                    font=small,
                )
                draw.text(
                    (14, 93),
                    f"Skin overlap {skin_depth * 1000:.2f} mm | total contact normal force {normal:.1f} N",
                    fill="white",
                    font=small,
                )
                writer.append_data(np.asarray(frame))
                next_frame += 1 / 30
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
    after = [s for s in samples if s["t"] >= START]
    before = [s for s in samples if 4.5 <= s["t"] < START]
    ramp_end = (
        PROGRESSIVE_LIMIT_TIME if profile == "progressive" else START + RAMP_SECONDS
    )
    end_ramp = min(samples, key=lambda s: abs(s["t"] - ramp_end))
    tail = [s for s in samples if s["t"] >= duration - 0.2]
    max_translation = max(s["translation_from_pre_squeeze_m"] for s in after)
    max_rotation = max(s["rotation_from_pre_squeeze_deg"] for s in after)
    warnings = [int(w.number) for w in data.warning]
    retained = bool(
        all(
            s["force_bearing_hand_contacts"] >= 2
            and "t" in s["digits"]
            and len(s["digits"]) >= 2
            and not s["support_contacts"]
            and s["cube_pos"][2] > default_cube_position(side)[2] - 0.05
            for s in tail
        )
    )
    result = {
        "side": side,
        "variant": variant,
        "profile": profile,
        "command_trace_sha256": command_hash.hexdigest(),
        "progressive_strong_time_s": PROGRESSIVE_STRONG_TIME
        if profile == "progressive"
        else None,
        "progressive_joint_limit_time_s": PROGRESSIVE_LIMIT_TIME
        if profile == "progressive"
        else None,
        "contact_model": contact_model,
        "cube_mass_kg": cube_mass,
        "initial_cube_position": list(
            cube_pos if cube_pos is not None else default_cube_position(side)
        ),
        "requested_preload_multiplier": multiplier if profile == "preload" else None,
        "applied_preload_multiplier": preload.applied_multiplier if preload else None,
        "initial_target_scale": 0.6,
        "final_target_scale": None
        if profile in ("preload", "progressive")
        else 0.6
        if profile == "baseline"
        else fraction,
        "initial_ctrl": base.tolist(),
        "final_ctrl": data.ctrl.tolist(),
        "actuator_joints": [
            model.joint(int(j)).name for j in model.actuator_trnid[:, 0]
        ],
        "source_body": str(source),
        "source_body_sha256": source_hash,
        "source_definition_sha256": hashlib.sha256(
            source.with_name(f"orcahand_{side}.mjcf").read_bytes()
        ).hexdigest(),
        "solver_cone": int(model.opt.cone),
        "solver_impratio": float(model.opt.impratio),
        "mujoco_version": mujoco.__version__,
        "timestep": model.opt.timestep,
        "actuator_force_limits": model.actuator_forcerange.tolist(),
        "actuator_kp": model.actuator_gainprm[:, 0].tolist(),
        "palm_body": model.body(palm).name,
        "duration_s": duration,
        "squeeze_start_s": START,
        "squeeze_ramp_seconds": ramp_end - START,
        "motion_reference_time_s": after[0]["t"],
        "max_translation_from_pre_squeeze_m": max_translation,
        "max_rotation_from_pre_squeeze_deg": max_rotation,
        "end_ramp_translation_m": end_ramp["translation_from_pre_squeeze_m"],
        "end_ramp_rotation_deg": end_ramp["rotation_from_pre_squeeze_deg"],
        "pre_squeeze_mean_total_normal_N": float(
            np.mean([s["sum_hand_normal_force_N"] for s in before])
        ),
        "end_ramp_total_normal_N": end_ramp["sum_hand_normal_force_N"],
        "peak_total_normal_N": max(s["sum_hand_normal_force_N"] for s in after),
        "peak_normal_force_by_region_N": {
            region: max(s["normal_force_by_region_N"][region] for s in after)
            for region in ("fingers", "thumb", "palm", "other_hand")
        },
        "max_skin_vertex_depth_m": max(s["skin_vertex_depth_m"] for s in after),
        "max_raw_contact_depth_m": max(s["raw_contact_depth_m"] for s in after),
        "max_skin_vertex_depth_entire_rollout_m": max(
            s["skin_vertex_depth_m"] for s in samples
        ),
        "max_raw_contact_depth_entire_rollout_m": max(
            s["raw_contact_depth_m"] for s in samples
        ),
        "max_raw_contact_depth_every_physics_step_m": max_step_penetration,
        "max_abs_motor_torque_every_physics_step_Nm": max_step_motor_torque,
        "finite_state_every_physics_step": finite_state,
        "equality_constraints": model.neq,
        "external_force_arrays_zero": bool(
            not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
        ),
        "retained_at_end": retained,
        "final_cube_position": samples[-1]["cube_pos"],
        "warnings": warnings,
        "passes_motion_check": bool(
            retained
            and not any(warnings)
            and finite_state
            and max_translation <= TRANSLATION_TOLERANCE
            and max_rotation <= ROTATION_TOLERANCE_DEG
            and max(s["skin_vertex_depth_m"] for s in samples) <= 0.001
            and max(s["raw_contact_depth_m"] for s in samples) <= 0.002
            and max_step_penetration <= 0.002
        ),
        "motion_tolerances": {
            "translation_m": TRANSLATION_TOLERANCE,
            "rotation_deg": ROTATION_TOLERANCE_DEG,
        },
        "measurement_notes": "Peak excursion, not endpoint drift; poses relative to the palm from 5.002s, sampled every 20ms. Skin-vertex penetration is a lower bound. Contact normal force is the sum of positive hand-cube normal forces, not a single-finger force or net external force. Legacy contact settings are an explicit ablation; native uses source MJCF unchanged.",
    }
    # Contact correctness is independent of an optional immobility/controller
    # criterion. Teleoperation is allowed to move an object, not through solids.
    result["passes_contact_check"] = bool(
        retained
        and finite_state
        and not any(warnings)
        and result["max_skin_vertex_depth_entire_rollout_m"] <= 0.001
        and max_step_penetration <= 0.002
        and result["external_force_arrays_zero"]
        and model.neq == 0
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    if output_dir:
        (output_dir / f"{name}.json").write_text(
            json.dumps({"result": result, "samples": samples}, indent=2) + "\n"
        )
        (output_dir / f"{name}_scene.xml").write_text(xml)
    return result


def validation_suite(output_dir, video=False, multiplier=4.0):
    """Both hands, load/pose perturbations, long hold and causal ablations."""
    from itertools import product
    from orca_sim.versions import PACKAGE_ROOT

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    cases = []
    for side in ("right", "left"):
        for mass in (0.04, 0.08, 0.12):
            cases.append(
                (f"{side}_mass_{mass:g}", dict(side=side, cube_mass=mass), "stable")
            )
        for axis, sign in ((0, 1), (1, 1), (2, -1), (2, 1)):
            pos = np.array(default_cube_position(side))
            pos[axis] += 0.003 * sign * (-1 if axis == 0 and side == "left" else 1)
            cases.append(
                (
                    f"{side}_offset_{axis}_{sign}",
                    dict(side=side, cube_pos=pos),
                    "stable",
                )
            )
        cases.append((f"{side}_long_hold", dict(side=side, duration=60), "stable"))
        for profile in ("baseline", "flexion", "full"):
            cases.append(
                (
                    f"{side}_native_{profile}",
                    dict(side=side, profile=profile),
                    "stable" if profile == "baseline" else "contact",
                )
            )
        cases.append(
            (
                f"{side}_legacy_flexion",
                dict(side=side, profile="flexion", contact_model="legacy"),
                "failure",
            )
        )
        cases.append(
            (
                f"{side}_legacy_preload",
                dict(side=side, contact_model="legacy"),
                "motion_failure",
            )
        )
    for name, kwargs, expected in cases:
        # This historical suite explicitly includes the optional controller;
        # it is not the model-only teleoperation/contact acceptance suite.
        kwargs.setdefault("profile", "preload")
        record_video = video and name in (
            "right_mass_0.08",
            "left_mass_0.08",
            "right_native_flexion",
            "right_legacy_flexion",
        )
        result = run(
            **kwargs,
            multiplier=multiplier,
            output_dir=output_dir / name,
            video=record_video,
        )
        if expected == "stable":
            passed = result["passes_motion_check"]
            if result["profile"] == "preload":
                passed &= (
                    result["end_ramp_total_normal_N"]
                    > result["pre_squeeze_mean_total_normal_N"] * 1.1
                )
        elif expected == "contact":
            passed = (
                result["retained_at_end"]
                and not any(result["warnings"])
                and result["max_raw_contact_depth_entire_rollout_m"] <= 0.002
                and result["max_raw_contact_depth_every_physics_step_m"] <= 0.002
                and result["max_skin_vertex_depth_entire_rollout_m"] <= 0.001
            )
        elif expected == "failure":
            passed = (
                not result["retained_at_end"]
                and result["max_skin_vertex_depth_m"] > 0.002
            )
        else:
            passed = not result["passes_motion_check"]
        results.append(
            dict(
                case=name,
                expected=expected,
                passes_expected_check=bool(passed),
                **result,
            )
        )
        print(
            f"{name}: expected_check={passed}, retained={result['retained_at_end']}, peak={result['max_translation_from_pre_squeeze_m'] * 1000:.3f}mm/{result['max_rotation_from_pre_squeeze_deg']:.3f}deg, skin={result['max_skin_vertex_depth_m'] * 1000:.3f}mm",
            flush=True,
        )
    sources = [
        PACKAGE_ROOT / "models" / "v2" / "mjcf" / f"orcahand_{s}{suffix}"
        for s, suffix in product(("left", "right"), ("_body.xml", ".mjcf"))
    ]
    sources += [
        PACKAGE_ROOT / "models" / "v2" / "assets" / "options.xml",
        PACKAGE_ROOT / "mujoco_grasp.py",
        Path(__file__),
        Path(__file__).with_name("standalone_hand_grasp.py"),
    ]
    report = {
        "all_expected_checks_pass": all(r["passes_expected_check"] for r in results),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "results": results,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if video:
        import imageio.v2 as imageio

        for name, left_case, left_profile in (
            ("before_vs_fixed", "right_legacy_flexion", "flexion"),
            ("fixed_contacts_raw_vs_preload", "right_native_flexion", "flexion"),
        ):
            with (
                imageio.get_reader(
                    output_dir / left_case / f"right_released_{left_profile}_1.mp4"
                ) as left,
                imageio.get_reader(
                    output_dir / "right_mass_0.08" / "right_released_preload_1.mp4"
                ) as right,
                imageio.get_writer(
                    output_dir / f"{name}.mp4", fps=30, codec="libx264", quality=8
                ) as writer,
            ):
                for a, b in zip(left, right):
                    writer.append_data(np.concatenate([a, b], axis=1))
    return report


def compose_comparisons(directory, fraction):
    import imageio.v2 as imageio

    comparisons = (
        ("current_baseline_vs_strong", "released_baseline", "released_flexion"),
        ("original_vs_current_strong", "skin_disabled_flexion", "released_flexion"),
    )
    for name, first, second in comparisons:
        with (
            imageio.get_reader(directory / f"right_{first}_{fraction:g}.mp4") as left,
            imageio.get_reader(directory / f"right_{second}_{fraction:g}.mp4") as right,
            imageio.get_writer(
                directory / f"{name}.mp4", fps=30, codec="libx264", quality=8
            ) as writer,
        ):
            for first_frame, second_frame in zip(left, right):
                writer.append_data(np.concatenate([first_frame, second_frame], axis=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--variant", choices=("released", "skin_disabled"), default="released"
    )
    parser.add_argument(
        "--profile",
        choices=("baseline", "flexion", "full", "preload", "progressive"),
        default="progressive",
    )
    parser.add_argument(
        "--contact-model", choices=("native", "legacy"), default="native"
    )
    parser.add_argument("--multiplier", type=float, default=4.0)
    parser.add_argument(
        "--duration",
        type=float,
        help="Defaults to 40 s for progressive closure, otherwise 18.5 s",
    )
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--suite", action="store_true")
    parser.add_argument(
        "--validation-suite",
        action="store_true",
        help="Historical optional-controller comparison, not model-only contact acceptance",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check retention, penetration, finite state and absence of artificial holding forces",
    )
    parser.add_argument(
        "--check-motion",
        action="store_true",
        help="Additionally require peak movement below 2 mm / 5 degrees (not a contact-correctness criterion)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("renders/standalone_hand/squeeze_fixed"),
    )
    args = parser.parse_args()
    if args.duration is None:
        args.duration = (
            PROGRESSIVE_DURATION
            if args.profile == "progressive" and not args.suite
            else DURATION
        )
    if args.validation_suite:
        report = validation_suite(args.output_dir, args.video, args.multiplier)
        if not report["all_expected_checks_pass"]:
            raise SystemExit(1)
        return
    if args.suite:
        results = []
        for side in ("right", "left"):
            for variant in ("released", "skin_disabled"):
                for profile in ("baseline", "flexion", "full"):
                    result = run(
                        side,
                        variant,
                        profile,
                        args.fraction,
                        args.output_dir,
                        args.video
                        and side == "right"
                        and (variant == "released" or profile == "flexion"),
                        duration=args.duration,
                        contact_model=args.contact_model,
                    )
                    results.append(result)
                    print(
                        f"{side}/{variant}/{profile}: retained={result['retained_at_end']}, peak={result['max_translation_from_pre_squeeze_m'] * 1000:.1f}mm, {result['max_rotation_from_pre_squeeze_deg']:.1f}deg",
                        flush=True,
                    )
        report = {
            "all_pass_motion_check": all(r["passes_motion_check"] for r in results),
            "all_pass_contact_check": all(r["passes_contact_check"] for r in results),
            "results": results,
        }
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        if args.video:
            compose_comparisons(args.output_dir, args.fraction)
        if (args.check and not report["all_pass_contact_check"]) or (
            args.check_motion and not report["all_pass_motion_check"]
        ):
            raise SystemExit(1)
    else:
        result = run(
            args.side,
            args.variant,
            args.profile,
            args.fraction,
            args.output_dir,
            args.video,
            duration=args.duration,
            contact_model=args.contact_model,
            multiplier=args.multiplier,
        )
        print(json.dumps(result, indent=2))
        if (args.check and not result["passes_contact_check"]) or (
            args.check_motion and not result["passes_motion_check"]
        ):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
