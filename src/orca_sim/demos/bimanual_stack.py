"""Both hands pick together, then left places blue and right stacks red on it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from orca_sim.bimanual_stacking import BimanualStack, STACK_LAYOUT, StackContactAudit
from orca_sim.contact_diagnostics import skin_geoms, skin_vertex_depth
from orca_sim.mujoco_bimanual import OrcaBimanualPandaCubeStackingMujoco, scene_xml
from orca_sim.panda_stacking import stack_metrics


def run(output_dir, *, video=False, duration=80.0, cube_spawn_regions=None, seed=None):
    if not np.isfinite(duration) or duration < 71:
        raise ValueError(
            "Allow at least 71 seconds for parallel pickups, placements and settling"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = OrcaBimanualPandaCubeStackingMujoco(
        object_layout=STACK_LAYOUT,
        target_marker="x",
        frame_skip=1,
        render_mode="rgb_array" if video else None,
        cube_spawn_regions=cube_spawn_regions,
    )
    m, d = env.model, env.data
    writer = controller = None
    trace, settled, forbidden = [], [], []
    error = None
    max_depth = max_skin = max_limit = max_torque = 0.0
    max_lift = dict(red_cube=0.0, blue_cube=0.0)
    pickup_lift_start = dict.fromkeys(max_lift)
    force_free = True
    checks = 0
    try:
        _, reset_info = env.reset(seed=seed)
        initial_heights = {name: float(d.body(name).xpos[2]) for name in max_lift}
        # Export the sampled initial layout, not the nominal constructor XML.
        (output_dir / "scene.xml").write_text(
            scene_xml(
                start_pose=env.start_pose,
                object_layout=env.object_layout,
                target_marker="x",
            )
        )
        (output_dir / "reset.json").write_text(
            json.dumps(dict(seed=seed, **reset_info), indent=2) + "\n"
        )
        controller = BimanualStack(env)
        audit = StackContactAudit(env)
        skins = skin_geoms(m)
        cubes = [m.geom(n + "_geom").id for n in max_lift]
        if video:
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw, ImageFont

            writer = imageio.get_writer(
                output_dir / "rollout.mp4", fps=20, codec="libx264", quality=8
            )
            Image.fromarray(env.render_camera("front", width=1280, height=720)).save(
                output_dir / "home.png"
            )
        next_frame, last_phase = 0.0, None
        saved_stages = set()
        for step in range(round(duration / m.opt.timestep)):
            action, phase = controller.command()
            if phase != last_phase:
                print(f"{d.time:5.1f}s | {phase}", flush=True)
                last_phase = phase
            if controller.phase not in saved_stages:
                state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
                state = np.empty(mujoco.mj_stateSize(m, state_spec))
                mujoco.mj_getState(m, d, state, state_spec)
                np.savez(
                    output_dir / (controller.phase.lower().replace(" ", "_") + ".npz"),
                    state=state,
                )
                saved_stages.add(controller.phase)
            env.step(action)
            checks += 1
            contacts = audit.inspect()
            max_depth = max(max_depth, contacts["max_depth_m"])
            if contacts["forbidden"]:
                forbidden.extend(
                    dict(t=float(d.time), **v) for v in contacts["forbidden"]
                )
                raise RuntimeError("Unintended contact: " + str(forbidden[0]))
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                raise RuntimeError("Nonfinite physics state")
            force_free &= not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied)
            q = d.qpos[env.qadr]
            max_limit = max(
                max_limit,
                float(np.max(env.action_space.low - q)),
                float(np.max(q - env.action_space.high)),
            )
            max_torque = max(
                max_torque, float(np.max(np.abs(d.actuator_force[np.r_[7:24, 31:48]])))
            )
            for name in max_lift:
                max_lift[name] = max(
                    max_lift[name], float(d.body(name).xpos[2] - initial_heights[name])
                )
                if (
                    pickup_lift_start[name] is None
                    and d.body(name).xpos[2] > initial_heights[name] + 0.03
                ):
                    pickup_lift_start[name] = float(d.time)
            if step % 10 == 0:
                max_skin = max(
                    max_skin, *(skin_vertex_depth(m, d, g, skins) for g in cubes)
                )
                metrics = dict(
                    t=float(d.time),
                    phase=phase,
                    red=d.body("red_cube").xpos.tolist(),
                    blue=d.body("blue_cube").xpos.tolist(),
                    qpos=d.qpos.tolist(),
                    ctrl=d.ctrl.tolist(),
                    **stack_metrics(env),
                )
                trace.append(metrics)
                if controller.phase == "SETTLE STACK":
                    settled.append(
                        dict(**metrics, hand_contacts=contacts["hand_contacts"])
                    )
            if writer is not None and d.time >= next_frame - 1e-9:
                frame = Image.fromarray(
                    env.render_camera("front", width=1280, height=720)
                )
                # An orbit-camera inset makes contacts readable; it is a debug
                # view, not a fourth named observation camera.
                draw = ImageDraw.Draw(frame)
                views = (
                    (("right", "red_cube", 16), ("left", "blue_cube", 816))
                    if controller.phase == "BOTH PICK CUBES"
                    else (
                        (("left", "blue_cube", 816),)
                        if controller.phase.startswith("LEFT")
                        else (("right", "red_cube", 816),)
                    )
                )
                for side, active, x in views:
                    env.camera.lookat[:] = d.body(active).xpos + [0, 0, 0.02]
                    env.camera.distance = 0.48
                    env.camera.azimuth = 120 if active == "blue_cube" else 60
                    env.camera.elevation = -18
                    frame.paste(
                        Image.fromarray(env.render()).resize((448, 252)), (x, 452)
                    )
                    draw.rectangle((x, 452, x + 448, 477), fill=(20, 24, 31))
                    draw.text(
                        (x + 8, 456),
                        f"{side.upper()} CONTACT DETAIL (orbit view)",
                        font=ImageFont.load_default(size=16),
                        fill="white",
                    )
                draw.rectangle((0, 0, 1280, 65), fill=(20, 24, 31))
                draw.text(
                    (16, 10),
                    f"{d.time:.1f}s | {phase}",
                    font=ImageFont.load_default(size=21),
                    fill="white",
                )
                draw.text(
                    (16, 37),
                    "Native MuJoCo | right: red | left: blue | X: base placement | no object attachments",
                    font=ImageFont.load_default(size=17),
                    fill="white",
                )
                writer.append_data(np.asarray(frame))
                next_frame += 0.05
            if controller.done:
                break
        if not controller.done:
            raise RuntimeError("Demo did not complete before time limit")
    except RuntimeError as exc:
        error = str(exc)
        print(error, flush=True)
    finally:
        final = stack_metrics(env)
        xy = float(
            np.linalg.norm(
                d.body("blue_cube").xpos[:2] - d.body("stack_target").xpos[:2]
            )
        )
        warnings = {str(i): int(w.number) for i, w in enumerate(d.warning) if w.number}
        stable = len(settled) >= 499 and all(
            s["stack_contacts"] > 0
            and s["blue_table_contacts"] > 0
            and s["hand_contacts"] == 0
            and s["stack_xy_error_m"] < 0.012
            and abs(s["stack_height_m"] - 0.056) < 0.004
            and s["cube_linear_speed_m_s"] < 0.02
            and s["cube_angular_speed_rad_s"] < 0.2
            for s in settled
        )
        report = dict(
            engine="MuJoCo",
            version=mujoco.__version__,
            duration_s=float(d.time),
            object_layout=env.object_layout,
            cube_spawn_regions=env.cube_spawn_regions,
            seed=seed,
            events=controller.events if controller else [],
            task="bimanual_pick_place_stack",
            max_lift_m=max_lift,
            pickup_lift_start_s=pickup_lift_start,
            base_target_error_m=xy,
            **final,
            collision_checks=checks,
            forbidden_contact_count=len(forbidden),
            forbidden_contacts=forbidden[:20],
            max_contact_depth_m=max_depth,
            max_skin_vertex_depth_m=max_skin,
            max_joint_limit_violation_rad=max_limit,
            max_hand_torque_Nm=max_torque,
            external_forces_zero=bool(force_free),
            equalities=int(m.neq),
            warnings=warnings,
            stable_stack_samples=len(settled),
            validation_settle_seconds=float(d.time - controller.start_time)
            if controller and controller.phase == "SETTLE STACK"
            else 0.0,
            skin_vertex_probe_interval_s=0.01,
            error=error,
            success=bool(
                error is None
                and stable
                and xy < 0.012
                and not forbidden
                and max_skin < 0.0001
                and max_depth < 0.001
                and max_limit < 0.03
                and max_torque <= 1 + 1e-10
                and force_free
                and m.neq == 0
                and not warnings
            ),
        )
        (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (output_dir / "trace.json").write_text(json.dumps(trace) + "\n")
        if writer is not None:
            Image.fromarray(env.render_camera("front", width=1280, height=720)).save(
                output_dir / "final.png"
            )
            Image.fromarray(env.render()).save(output_dir / "final_detail.png")
            writer.close()
        env.close()
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("renders/bimanual_stack")
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--cube-spawn-regions",
        type=Path,
        help="Environment reset distribution JSON (omit to retain the fixed layout)",
    )
    parser.add_argument("--seed", type=int, help="Reproducible environment reset seed")
    args = parser.parse_args()
    result = run(
        args.output_dir,
        video=args.video,
        cube_spawn_regions=args.cube_spawn_regions,
        seed=args.seed,
    )
    if args.check and not result["success"]:
        raise SystemExit("Bimanual stack validation failed; see report.json")


if __name__ == "__main__":
    main()
