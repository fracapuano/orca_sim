#!/usr/bin/env python3
"""Step the native v2 hand MJCF with a free cube and retracting support.

No arms, Genesis, IK, grasp welds, external cube forces, or gain overrides.
The default uses the current source files directly. ``skin_disabled`` reproduces
the pre-fix collision masks in memory, without changing any other parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape

import mujoco
import numpy as np

from orca_sim.versions import PACKAGE_ROOT

VARIANTS = ("released", "skin_disabled", "open_control")
MAX_SKIN_VERTEX_DEPTH = 0.001
MAX_CONTACT_DEPTH = 0.002
# Small, collision-free perturbations; x is mirrored for the left hand.
FIXTURE_OFFSETS = ((0.003, 0, 0), (0, 0.003, 0), (0, 0, -0.003), (0, 0, 0.003))


def xml_include(path):
    return f'<include file="{escape(str(path), {chr(34): "&quot;"})}"/>'


def default_cube_position(side):
    return (-0.0143 if side == "right" else -0.0057, 0.1762, 0.4049)


def scene_xml(
    side="right",
    cube_pos=None,
    cube_mass=0.08,
    variant="released",
    body_file=None,
    contact_model="native",
):
    if side not in ("left", "right") or variant not in VARIANTS:
        raise ValueError("Unknown side or experiment variant")
    if not np.isfinite(cube_mass) or cube_mass <= 0:
        raise ValueError("Cube mass must be positive and finite")
    if contact_model not in ("native", "legacy"):
        raise ValueError("Unknown contact model")
    root = PACKAGE_ROOT / "models" / "v2"
    x, y, z = cube_pos if cube_pos is not None else default_cube_position(side)
    if not np.isfinite([x, y, z]).all():
        raise ValueError("Cube position must be finite")
    body_path = body_file or root / "mjcf" / f"orcahand_{side}_body.xml"
    body_content = xml_include(body_path)
    if variant == "skin_disabled" or contact_model == "legacy":
        body = ET.parse(body_path).getroot().find("body")
        if contact_model == "legacy":
            body.attrib.pop("childclass", None)
        if variant == "skin_disabled":
            for geom in body.iter("geom"):
                if "Skin" in geom.get("mesh", ""):
                    geom.set("contype", "0")
                    geom.set("conaffinity", "0")
        body_content = ET.tostring(body, encoding="unicode")
    legacy_option = (
        '<option cone="pyramidal" impratio="1"/>' if contact_model == "legacy" else ""
    )
    return f'''<mujoco model="released_{side}_hand_grasp_test">
  {xml_include(root / "assets" / "options.xml")}
  {xml_include(root / "mjcf" / f"orcahand_{side}.mjcf")}
  {legacy_option}
  <visual><global offwidth="1280" offheight="720"/><headlight ambient="0.3 0.3 0.3" diffuse="0.6 0.6 0.6"/></visual>
  <worldbody>
    <light pos="-0.3 0.4 1.2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="0.2 0.5 0.15" dir="0 -1 0.5" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="1 1 0.01" rgba="0.15 0.18 0.22 1"/>
    <body name="hand_mount" pos="0 0 0.45">
      {body_content}
    </body>
    <body name="cube" pos="{x} {y} {z}">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="0.025 0.025 0.025" mass="{cube_mass}" contype="1" conaffinity="3" rgba="0.85 0.07 0.04 1"/>
    </body>
    <body name="support" mocap="true" pos="{x} {y} {z - 0.033}">
      <geom name="support_geom" type="box" size="0.04 0.04 0.008" contype="2" conaffinity="0" rgba="0.18 0.4 0.55 1"/>
    </body>
  </worldbody>
</mujoco>'''


def skin_geoms(model):
    return [
        i
        for i in range(model.ngeom)
        if model.geom_dataid[i] >= 0
        and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
        and "Skin" in model.mesh(int(model.geom_dataid[i])).name
    ]


def skin_vertex_depth(model, data, cube_id, skins):
    """Lower bound: deepest rendered skin mesh vertex inside the actual cube."""
    maximum = 0.0
    cube_rotation = data.geom_xmat[cube_id].reshape(3, 3)
    for geom in skins:
        mesh = int(model.geom_dataid[geom])
        adr, num = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        verts = model.mesh_vert[adr : adr + num]
        world = verts @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        local = (world - data.geom_xpos[cube_id]) @ cube_rotation
        depth = np.min(model.geom_size[cube_id] - np.abs(local), axis=1)
        maximum = max(maximum, float(np.max(depth, initial=0)))
    return maximum


def targets(model, side, fraction):
    target = np.zeros(model.nu)
    for i in range(model.nu):
        name = model.joint(int(model.actuator_trnid[i, 0])).name
        if name.endswith(
            ("i-mcp", "i-pip", "m-mcp", "m-pip", "r-mcp", "r-pip", "p-mcp", "p-pip")
        ):
            target[i] = 1.45 * fraction
        for joint, value in {
            "t-cmc": 0.55,
            "t-abd": 0,
            "t-mcp": 0.3,
            "t-pip": 0.8,
        }.items():
            if name == f"{side}_{joint}":
                target[i] = value * fraction
    return np.clip(
        target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    )


def run(
    side="right",
    variant="released",
    cube_pos=None,
    fraction=0.6,
    video=None,
    output_dir=None,
    hold_seconds=5.0,
    cube_mass=0.08,
    body_file=None,
    label=None,
    contact_model="native",
):
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Closure fraction must be between zero and one")
    if not np.isfinite(hold_seconds) or hold_seconds < 1:
        raise ValueError("Hold duration must be at least one second")
    cube_pos = tuple(cube_pos if cube_pos is not None else default_cube_position(side))
    source = (
        body_file
        or PACKAGE_ROOT / "models" / "v2" / "mjcf" / f"orcahand_{side}_body.xml"
    )
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    xml = scene_xml(side, cube_pos, cube_mass, variant, body_file, contact_model)
    model = mujoco.MjModel.from_xml_string(xml)
    skins = skin_geoms(model)
    data = mujoco.MjData(model)
    cube = model.body("cube").id
    cube_geom = model.geom("cube_geom").id
    support_geom = model.geom("support_geom").id
    support_mocap = int(model.body_mocapid[model.body("support").id])
    support_start = model.body_pos[model.body("support").id].copy()
    mujoco.mj_forward(model, data)
    initial_skin_depth = skin_vertex_depth(model, data, cube_geom, skins)
    initial_collision_depth = max(
        [0.0]
        + [
            -float(c.dist)
            for c in data.contact
            if cube_geom in (c.geom1, c.geom2)
            and support_geom not in (c.geom1, c.geom2)
        ]
    )
    if max(initial_skin_depth, initial_collision_depth) > 1e-5:
        raise ValueError(
            "The cube starts intersecting the hand; choose a collision-free fixture position"
        )
    close_target = targets(model, side, fraction if variant != "open_control" else 0)
    renderer = writer = None
    if video:
        import imageio.v2 as imageio

        video.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, height=720, width=1280)
        writer = imageio.get_writer(video, fps=30, codec="libx264", quality=8)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-0.01, 0.17, 0.39]
    camera.distance, camera.azimuth, camera.elevation = 0.32, 90, 15
    duration = 3.5 + hold_seconds
    samples = []
    next_frame = 0.0
    start = time.perf_counter()
    try:
        for step in range(round(duration / model.opt.timestep)):
            t = step * model.opt.timestep
            data.ctrl[:] = close_target * np.clip((t - 0.5) / 2.0, 0, 1)
            data.mocap_pos[support_mocap] = support_start - [
                0,
                0,
                0.25 * np.clip((t - 3.0) / 0.5, 0, 1),
            ]
            mujoco.mj_step(model, data)
            if step % 10:
                continue
            mujoco.mj_forward(model, data)
            digits, hand_contacts, support_contacts, max_depth = set(), 0, 0, 0.0
            for contact in data.contact:
                if cube_geom not in (contact.geom1, contact.geom2):
                    continue
                other = contact.geom2 if contact.geom1 == cube_geom else contact.geom1
                if other == support_geom:
                    support_contacts += 1
                elif model.body(int(model.geom_bodyid[other])).name.startswith(
                    side + "_"
                ):
                    hand_contacts += 1
                    max_depth = max(max_depth, -float(contact.dist))
                    body = int(model.geom_bodyid[other])
                    while body:
                        for j in range(
                            model.body_jntadr[body],
                            model.body_jntadr[body] + model.body_jntnum[body],
                        ):
                            name = model.joint(j).name
                            if name.startswith(side + "_") and "-" in name:
                                digits.add(name.split("_")[-1].split("-")[0])
                        body = int(model.body_parentid[body])
            pos = data.xpos[cube].copy()
            depth = skin_vertex_depth(model, data, cube_geom, skins)
            samples.append(
                {
                    "t": float(data.time),
                    "cube_pos": pos.tolist(),
                    "contacts": hand_contacts,
                    "digits": sorted(digits),
                    "support_contacts": support_contacts,
                    "collision_depth_m": max_depth,
                    "skin_vertex_depth_m": depth,
                }
            )
            if writer and data.time >= next_frame:
                from PIL import Image, ImageDraw, ImageFont

                renderer.update_scene(data, camera=camera)
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 1280, 78), fill=(20, 24, 31))
                font = ImageFont.load_default(size=24)
                phase = (
                    "HOLD: support removed"
                    if t >= 3.5
                    else (
                        "Retracting support"
                        if t >= 3
                        else "Closing"
                        if t >= 0.5
                        else "Settling"
                    )
                )
                title = label or f"{side} / {variant}"
                draw.text(
                    (18, 10),
                    f"{title}   |   {phase}   |   t={data.time:.2f}s",
                    fill="white",
                    font=font,
                )
                draw.text(
                    (18, 44),
                    f"hand contacts={hand_contacts}    skin vertex overlap={depth * 1000:.2f} mm",
                    fill="white",
                    font=font,
                )
                writer.append_data(np.asarray(image))
                next_frame += 1 / 30
        warnings = [int(w.number) for w in data.warning]
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
    mujoco.mj_forward(model, data)
    hold = [s for s in samples if s["t"] >= 4.0]
    drop = float(cube_pos[2] - data.xpos[cube, 2])
    hold_drift = (
        float(np.linalg.norm(np.asarray(hold[-1]["cube_pos"]) - hold[0]["cube_pos"]))
        if hold
        else float("inf")
    )
    held = bool(
        hold
        and all(
            s["contacts"] >= 2
            and "t" in s["digits"]
            and len(s["digits"]) >= 2
            and not s["support_contacts"]
            for s in hold
        )
        and drop < 0.05
        and hold_drift < 0.005
        and np.isfinite(data.qpos).all()
        and not any(warnings)
    )
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("The source asset changed during the experiment")
    result = {
        "side": side,
        "variant": variant,
        "contact_model": contact_model,
        "mujoco_version": mujoco.__version__,
        "cube_mass_kg": cube_mass,
        "cube_initial_position": list(cube_pos),
        "closure_fraction": fraction,
        "hold_seconds": hold_seconds,
        "held_after_support_removal": held,
        "final_cube_position": data.xpos[cube].tolist(),
        "cube_drop_m": drop,
        "hold_drift_m": hold_drift,
        "max_collision_depth_m": max(s["collision_depth_m"] for s in samples),
        "max_skin_vertex_depth_m": max(s["skin_vertex_depth_m"] for s in samples),
        "final_contacts": samples[-1]["contacts"],
        "final_digits": samples[-1]["digits"],
        "actuator_kp": model.actuator_gainprm[:, 0].tolist(),
        "actuator_force_range": model.actuator_forcerange.tolist(),
        "hand_mass_kg": float(
            sum(
                model.body_mass[i]
                for i in range(model.nbody)
                if model.body(i).name.startswith(side + "_")
            )
        ),
        "timestep": model.opt.timestep,
        "source_body": str(source),
        "source_body_sha256": source_hash,
        "initial_skin_vertex_depth_m": initial_skin_depth,
        "initial_collision_depth_m": initial_collision_depth,
        "skin_collision_enabled": bool(np.all(model.geom_contype[skins] != 0)),
        "penetration_measurement": "Deepest rendered skin mesh vertex inside the exact cube, sampled every 20 ms; a lower bound, not a complete triangle-intersection test.",
        "hold_measurement": "Support retracts between 3.0 and 3.5 seconds. Check opposing-digit contact and <5 mm drift from 4.0 seconds onward (0.5-second load-transfer settling allowance).",
        "warnings": warnings,
        "elapsed_seconds": time.perf_counter() - start,
    }
    result["passes_grasp_smoke_test"] = bool(
        held
        and result["max_skin_vertex_depth_m"] <= MAX_SKIN_VERTEX_DEPTH
        and result["max_collision_depth_m"] <= MAX_CONTACT_DEPTH
    )
    result["smoke_test_thresholds"] = {
        "max_skin_vertex_depth_m": MAX_SKIN_VERTEX_DEPTH,
        "max_collision_depth_m": MAX_CONTACT_DEPTH,
        "max_hold_drift_m": 0.005,
    }
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{side}_{variant}.json").write_text(
            json.dumps({"result": result, "samples": samples}, indent=2) + "\n"
        )
        (output_dir / f"{side}_{variant}_scene.xml").write_text(xml)
    return result


def run_suite(output_dir, video=False):
    """Run a reproducible baseline/fix/control study and bounded robustness sweep."""
    results = []
    for side in ("right", "left"):
        for variant, title in (
            ("skin_disabled", "ORIGINAL MASKS + SOFT CONTACTS"),
            ("released", "FIXED MJCF"),
            ("open_control", "OPEN-HAND CONTROL"),
        ):
            print(f"Running {side}: {variant}", flush=True)
            results.append(
                run(
                    side=side,
                    variant=variant,
                    output_dir=output_dir,
                    video=output_dir / f"{side}_{variant}.mp4" if video else None,
                    label=f"{title} ({side})",
                    contact_model="legacy" if variant == "skin_disabled" else "native",
                )
            )
        for mass in (0.04, 0.12):
            results.append(
                run(side=side, cube_mass=mass, output_dir=output_dir / f"mass_{mass}")
            )
        for index, offset in enumerate(FIXTURE_OFFSETS):
            position = np.asarray(default_cube_position(side)) + np.asarray(offset) * [
                -1 if side == "left" else 1,
                1,
                1,
            ]
            results.append(
                run(
                    side=side,
                    cube_pos=position,
                    output_dir=output_dir / f"offset_{index}",
                )
            )
    # Original-mask trials must reproduce the defect; open controls must drop.
    passed = all(
        (r["held_after_support_removal"] and r["max_skin_vertex_depth_m"] > 0.003)
        if r["variant"] == "skin_disabled"
        else (not r["held_after_support_removal"] and r["cube_drop_m"] > 0.20)
        if r["variant"] == "open_control"
        else r["passes_grasp_smoke_test"]
        for r in results
    )
    report = {
        "experiment_passed": passed,
        "scope": "Native v2 MJCF, normal gravity, fixed forearm, free 5 cm cube; 5 seconds after support withdrawal. No Genesis, arms, gain/inertia/friction overrides, cube inflation, or grasp attachments.",
        "baseline": "The historical baseline uses skin_disabled AND contact_model=legacy: original zero skin masks, soft contacts and pyramidal cones. Use skin_disabled with native contacts for a mask-only ablation. Actions, gains, masses and friction coefficients are identical.",
        "limitations": "A bounded smoke test, not an arbitrary-object benchmark. The visual penetration probe samples skin vertices every 20 ms and is a lower bound, not a complete triangle-intersection test. Four selected collision-free offsets are tested; other offsets can intersect the initial open hand and are rejected.",
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if video:
        import imageio.v2 as imageio

        for side in ("right", "left"):
            with (
                imageio.get_reader(output_dir / f"{side}_skin_disabled.mp4") as before,
                imageio.get_reader(output_dir / f"{side}_released.mp4") as after,
                imageio.get_writer(
                    output_dir / f"{side}_comparison.mp4",
                    fps=30,
                    codec="libx264",
                    quality=8,
                ) as writer,
            ):
                for first, second in zip(before, after):
                    writer.append_data(np.concatenate([first, second], axis=1))
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--variant", choices=VARIANTS, default="released")
    parser.add_argument(
        "--contact-model", choices=("native", "legacy"), default="native"
    )
    parser.add_argument("--cube-pos", type=float, nargs=3)
    parser.add_argument(
        "--cube-mass", type=float, default=0.08, help="Cube mass in kg (default: 0.08)"
    )
    parser.add_argument("--fraction", type=float, default=0.6)
    parser.add_argument("--hold-seconds", type=float, default=5)
    parser.add_argument(
        "--body-file",
        type=Path,
        help="Optional archived original hand body MJCF for reproducing the pre-fix baseline.",
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run both hands: original masks, fixed, open control, mass and position sweeps; other single-trial settings are ignored",
    )
    parser.add_argument("--label", help="Optional video title")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero unless the grasp and penetration smoke checks pass",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("renders/standalone_hand")
    )
    args = parser.parse_args()
    if args.suite:
        passed = run_suite(args.output_dir, args.video)
        print(
            f"Experiment {'PASS' if passed else 'FAIL'}: {args.output_dir / 'report.json'}"
        )
        raise SystemExit(0 if passed else 1)
    video = args.output_dir / f"{args.side}_{args.variant}.mp4" if args.video else None
    result = run(
        args.side,
        args.variant,
        args.cube_pos,
        args.fraction,
        video,
        args.output_dir,
        args.hold_seconds,
        args.cube_mass,
        args.body_file,
        args.label,
        args.contact_model,
    )
    print(json.dumps(result, indent=2))
    if args.check and not result["passes_grasp_smoke_test"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
