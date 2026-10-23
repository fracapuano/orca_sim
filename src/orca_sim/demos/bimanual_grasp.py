"""Independently grasp/lift red with the right ARM and blue with the left ARM.

Both robots remain dynamic. The inactive robot receives its unchanged home
targets. All collision masks/exclusions and native actuator limits are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from orca_sim.contact_diagnostics import skin_geoms, skin_vertex_depth
from orca_sim.mujoco_bimanual import OrcaBimanualPandaCubeStackingMujoco, SIDES
from orca_sim.palm_grasp import PalmFirstGrasp, smooth
from orca_sim.panda_stacking import blend_rotation


class ContactAudit:
    """Whitelist intended contacts; inspect every 1 ms physics step."""

    def __init__(self, env, side, cube):
        self.env, self.side, self.cube = env, side, cube
        self.hand_side = env.robots[side].hand_side if hasattr(env, "robots") else side
        m = env.model
        self.cube_id = m.geom(cube + "_geom").id
        self.roles, self.digits = [], []
        for g in range(m.ngeom):
            body = int(m.geom_bodyid[g])
            name = m.body(body).name
            role = "environment"
            if name in ("table", "red_cube", "blue_cube"):
                role = name
            for s in SIDES:
                if name.startswith(s + "_"):
                    role = s + ("_arm" if name.startswith(s + "_panda_") else "_hand")
            if hasattr(env, "geom_robots") and env.geom_robots[g] is not None:
                owner = env.geom_robots[g]
                role = owner + (
                    "_arm" if name.startswith(owner + "_panda_") else "_hand"
                )
            self.roles.append(role)
            digits = set()
            while body:
                for j in range(
                    m.body_jntadr[body], m.body_jntadr[body] + m.body_jntnum[body]
                ):
                    joint = m.joint(j).name
                    if joint.startswith(self.hand_side + "_") and "-" in joint:
                        digits.add(joint.split("_")[-1].split("-")[0])
                body = int(m.body_parentid[body])
            self.digits.append(digits)
        self.palm_mesh = (
            f"{self.hand_side}_{'R' if self.hand_side == 'right' else 'L'}-Carpals"
        )

    def grasp_feedback(self):
        """Only the hand/cube signals used by control; no collision auditing."""
        m, d = self.env.model, self.env.data
        result = dict(hand_contacts=0, palm_top_contacts=0, palm_top_load_N=0.0)
        wrench = np.empty(6)
        rotation = d.geom_xmat[self.cube_id].reshape(3, 3)
        for i, contact in enumerate(d.contact):
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if self.cube_id not in (g1, g2):
                continue
            other = g2 if g1 == self.cube_id else g1
            if self.roles[other] != self.side + "_hand":
                continue
            mujoco.mj_contactForce(m, d, i, wrench)
            if wrench[0] <= 1e-4:
                continue
            result["hand_contacts"] += 1
            mesh = int(m.geom_dataid[other])
            if mesh >= 0 and m.mesh(mesh).name == self.palm_mesh:
                point = (contact.pos - d.geom_xpos[self.cube_id]) @ rotation
                normal = contact.frame[:3] @ rotation
                if (
                    abs(point[2] - m.geom_size[self.cube_id, 2]) < 0.002
                    and abs(normal[2]) > 0.8
                ):
                    result["palm_top_contacts"] += 1
                    result["palm_top_load_N"] += float(wrench[0])
        return result

    def inspect(self):
        m, d = self.env.model, self.env.data
        result = dict(
            hand_contacts=0,
            table_contacts=0,
            contact_depth_m=0.0,
            palm_top_contacts=0,
            palm_top_load_N=0.0,
            digits=set(),
            forbidden=[],
        )
        wrench = np.zeros(6)
        for i, c in enumerate(d.contact):
            g1, g2 = int(c.geom1), int(c.geom2)
            roles = {self.roles[g1], self.roles[g2]}
            mujoco.mj_contactForce(m, d, i, wrench)
            force = float(wrench[0])
            physical = c.dist < -1e-7 or force > 1e-5
            intended = roles == {self.side + "_hand", self.cube}
            support = "table" in roles and bool(roles & {"red_cube", "blue_cube"})
            if physical and not (intended or support):
                result["forbidden"].append(
                    dict(
                        roles=sorted(roles),
                        bodies=[m.body(m.geom_bodyid[g]).name for g in (g1, g2)],
                        depth_m=max(0.0, -float(c.dist)),
                        force_N=force,
                    )
                )
            if self.cube_id not in (g1, g2):
                continue
            other = g2 if g1 == self.cube_id else g1
            if self.roles[other] == "table" and physical:
                result["table_contacts"] += 1
            if not intended:
                continue
            result["contact_depth_m"] = max(result["contact_depth_m"], -float(c.dist))
            if force <= 1e-4:
                continue
            result["hand_contacts"] += 1
            result["digits"].update(self.digits[other])
            mesh = int(m.geom_dataid[other])
            if mesh >= 0 and m.mesh(mesh).name == self.palm_mesh:
                rotation = d.geom_xmat[self.cube_id].reshape(3, 3)
                point = (c.pos - d.geom_xpos[self.cube_id]) @ rotation
                normal = c.frame[:3] @ rotation
                if (
                    abs(point[2] - m.geom_size[self.cube_id, 2]) < 0.002
                    and abs(normal[2]) > 0.8
                ):
                    result["palm_top_contacts"] += 1
                    result["palm_top_load_N"] += force
        result["digits"] = sorted(result["digits"])
        return result


class PalmLift:
    """Side-specific version of the validated palm-first controller."""

    def __init__(self, env, side, cube, *, yaw_degrees=0, offset=None, waypoints=()):
        self.env, self.side, self.cube = env, side, cube
        self.robot = env.robots[side]
        self.hand_side = self.robot.hand_side
        self.home = env.data.ctrl.copy()
        self.indices = self.robot.actuator_indices
        self.home_side = self.home[self.indices]
        self.joint_index = {
            name.removeprefix(self.hand_side + "_"): i
            for i, name in enumerate(self.robot.actuator_joints)
        }
        self.palm = PalmFirstGrasp(env.model.opt.timestep)
        self.audit = ContactAudit(env, side, cube)
        self.start = env.data.site(self.robot.hand_site).xpos.copy()
        self.home_rotation = (
            env.data.site(self.robot.hand_site).xmat.reshape(3, 3).copy()
        )
        yaw = np.deg2rad(yaw_degrees)
        # Swapping handedness reverses the lateral palm geometry. Mirror the
        # calibrated approach's small tilt as well; the saved home stays exact
        # and the approach smoothly blends from it. With these mirrored homes
        # this is the opposite native hand's validated orientation yawed 180°.
        grasp_rotation = np.diag([1, -1, 1]) @ self.home_rotation @ np.diag([-1, 1, 1])
        self.rotation = (
            np.array(
                [
                    [np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1],
                ]
            )
            @ grasp_rotation
        )
        # Native left/right meshes reflect about local x=-0.01, not x=0.
        offset = np.array(
            offset
            if offset is not None
            else [0.01 if self.hand_side == "right" else -0.03, 0.15, -0.048]
        )
        if (
            offset.shape != (3,)
            or not np.isfinite(offset).all()
            or not np.isfinite(yaw_degrees)
        ):
            raise ValueError("Use three finite grasp offsets and a finite yaw")
        self.offset = offset
        self.grasp = env.data.body(cube).xpos - self.rotation @ offset
        pre = self.grasp + [0, 0, 0.16]
        points = [self.start, *[np.asarray(p, dtype=float) for p in waypoints], pre]
        lengths = [np.linalg.norm(b - a) for a, b in zip(points, points[1:])]
        total = sum(lengths)
        seed = env.data.qpos[self.robot.arm_qadr].copy()
        path, amounts = [seed.copy()], [0.0]
        travelled = 0.0
        for a, b, length in zip(points, points[1:], lengths):
            for fraction in np.linspace(
                0, 1, max(3, int(80 * length / max(total, 1e-9)))
            )[1:]:
                amount = (travelled + fraction * length) / max(total, 1e-9)
                rotation = blend_rotation(self.home_rotation, self.rotation, amount)
                seed = env.solve_ik(
                    side, a * (1 - fraction) + b * fraction, rotation, seed=seed
                )
                path.append(seed.copy())
                amounts.append(amount)
            travelled += length
        self.path, self.amounts = np.array(path), np.array(amounts)
        self.zs = np.linspace(-0.03, 0.26, 146)
        solutions = []
        for z in reversed(self.zs):
            seed = env.solve_ik(side, self.grasp + [0, 0, z], self.rotation, seed=seed)
            solutions.append(seed.copy())
        self.solutions = np.array(solutions[::-1])

    def command(self, t):
        targets = np.zeros(24)
        fraction = 0.0
        if t < 1:
            return self.home.copy(), "SETTLE"
        if t < 5:
            amount = smooth((t - 1) / 4)
            arm = np.array(
                [np.interp(amount, self.amounts, self.path[:, j]) for j in range(7)]
            )
            phase = "APPROACH"
        else:
            phase, z, fraction = self.palm.command(t, self.audit.grasp_feedback())
            arm = np.array(
                [np.interp(z, self.zs, self.solutions[:, j]) for j in range(7)]
            )
        for name, i in self.joint_index.items():
            if name in {
                f"{digit}-{joint}" for digit in "imrp" for joint in ("mcp", "pip")
            }:
                targets[i] = 1.45 * fraction
            if name in ("t-cmc", "t-mcp", "t-pip"):
                targets[i] = {"t-cmc": 0.55, "t-mcp": 0.3, "t-pip": 0.8}[
                    name
                ] * fraction
        closure = (
            0.0
            if self.palm.close_time is None
            else smooth((t - self.palm.close_time - 1) / 2)
        )
        targets[self.joint_index["t-abd"]] = 0.9 * (1 - closure)
        amount = smooth((t - 1) / 4)
        if t < 5:
            targets[7:] = self.home_side[7:] * (1 - amount)
            i = self.joint_index["t-abd"]
            targets[i] = self.home_side[i] * (1 - amount) + 0.9 * amount
        i = self.joint_index["wrist"]
        targets[i] = self.home_side[i] * (1 - amount) + 0.06 * amount
        i = self.joint_index["t-cmc"]
        targets[i] = (
            self.home_side[i] * (1 - amount)
            + (-0.5 * (1 - closure) + 0.55 * closure) * amount
        )
        # Fold the distal thumb before lowering; sweeping an extended thumb
        # through opposition otherwise brushes the tabletop beside a small cube.
        for joint, value in (("t-mcp", 0.3), ("t-pip", 0.8)):
            i = self.joint_index[joint]
            targets[i] = self.home_side[i] * (1 - amount) + value * amount
        if self.palm.close_time is not None:
            age = t - self.palm.close_time
            # Curl distal joints first so fingertips sweep above the table,
            # then bring proximal joints around the cube's sides.
            for digit in "imrp":
                # The index shares the thumb's opposition space; leave it
                # partly open while the other three fingers wrap the cube.
                scale = 0.5 if digit == "i" else 1.0
                targets[self.joint_index[digit + "-pip"]] = (
                    scale * 1.45 * smooth(age / 1.3)
                )
                targets[self.joint_index[digit + "-mcp"]] = (
                    scale * 1.45 * smooth((age - 0.5) / 2.5)
                )
            spread = smooth((t - self.palm.close_time) / 0.6)
            targets[self.joint_index["p-abd"]] = -0.4 * spread
            targets[self.joint_index["r-abd"]] = -0.15 * spread
        targets[:7] = arm
        return self.env.side_action(self.side, targets, action=self.home), phase


def run(
    *,
    side,
    output_dir,
    video=False,
    yaw_degrees=0,
    offset=None,
    waypoints=(),
    duration=28.0,
    object_layout=None,
):
    if side not in SIDES:
        raise ValueError("Choose right or left")
    if not np.isfinite(duration) or duration < 28:
        raise ValueError("Allow at least 28 seconds for approach, grasp, lift and hold")
    cube = "red_cube" if side == "right" else "blue_cube"
    other_cube = "blue_cube" if side == "right" else "red_cube"
    other_side = "left" if side == "right" else "right"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = OrcaBimanualPandaCubeStackingMujoco(
        render_mode="rgb_array" if video else None,
        frame_skip=1,
        object_layout=object_layout,
    )
    m, d = env.model, env.data
    cube_start = d.body(cube).xpos.copy()
    other_start = d.body(other_cube).xpos.copy()
    home = d.ctrl.copy()
    inactive = env.robots[other_side].actuator_indices
    inactive_qadr = env.qadr[inactive]
    inactive_home = d.qpos[inactive_qadr].copy()
    skins = skin_geoms(m)
    cube_ids = [m.geom(n + "_geom").id for n in (cube, other_cube)]
    writer = None
    trace, hold, forbidden = [], [], []
    max_depth = max_skin = max_limit = max_hand_torque = 0.0
    max_other_motion = inactive_change = 0.0
    inactive_drift = 0.0
    checks = 0
    forces_zero = True
    command_hash = hashlib.sha256()
    error = None
    controller = None
    result = {}
    try:
        (output_dir / "scene.xml").write_text(env.xml)
        if video:
            import imageio.v2 as imageio
            from PIL import Image

            Image.fromarray(env.render()).save(output_dir / "home.png")
            writer = imageio.get_writer(
                output_dir / "rollout.mp4", fps=20, codec="libx264", quality=8
            )
        controller = PalmLift(
            env, side, cube, yaw_degrees=yaw_degrees, offset=offset, waypoints=waypoints
        )
        audit = controller.audit
        last_phase = None
        next_frame = 0.0
        saved_phases = set()
        for step in range(round(duration / m.opt.timestep)):
            action, phase = controller.command(d.time)
            if phase != last_phase:
                print(f"{side} {d.time:5.1f}s | {phase}", flush=True)
                last_phase = phase
            command_hash.update(action.astype("<f8").tobytes())
            env.step(action)
            contact = audit.inspect()
            checks += 1
            max_depth = max(max_depth, contact["contact_depth_m"])
            inactive_change = max(
                inactive_change,
                float(np.max(np.abs(d.ctrl[inactive] - home[inactive]))),
            )
            inactive_drift = max(
                inactive_drift,
                float(np.max(np.abs(d.qpos[inactive_qadr] - inactive_home))),
            )
            max_other_motion = max(
                max_other_motion,
                float(np.linalg.norm(d.body(other_cube).xpos - other_start)),
            )
            hand_ids = np.r_[np.arange(7, 24), np.arange(31, 48)]
            max_hand_torque = max(
                max_hand_torque, float(np.max(np.abs(d.actuator_force[hand_ids])))
            )
            q = d.qpos[env.qadr]
            max_limit = max(
                max_limit,
                float(np.max(env.action_space.low - q)),
                float(np.max(q - env.action_space.high)),
            )
            forces_zero &= not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied)
            if contact["forbidden"]:
                forbidden.extend(
                    dict(t=float(d.time), **c) for c in contact["forbidden"]
                )
                raise RuntimeError(
                    "Unintended contact: " + str(contact["forbidden"][0])
                )
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                raise RuntimeError("Non-finite physics state")
            if d.time >= duration - 5:
                hold.append(
                    dict(lift=float(d.body(cube).xpos[2] - cube_start[2]), **contact)
                )
            if step % 10 == 0:
                skin = max(skin_vertex_depth(m, d, g, skins) for g in cube_ids)
                max_skin = max(max_skin, skin)
                trace.append(
                    dict(
                        t=float(d.time),
                        phase=phase,
                        cube_pos=d.body(cube).xpos.tolist(),
                        ctrl=d.ctrl.tolist(),
                        qpos=d.qpos.tolist(),
                        skin_depth_m=skin,
                        **contact,
                    )
                )
            if writer is not None and d.time >= next_frame:
                from PIL import Image, ImageDraw, ImageFont

                frame = Image.fromarray(env.render())
                saved_camera = (
                    env.camera.lookat.copy(),
                    env.camera.distance,
                    env.camera.azimuth,
                    env.camera.elevation,
                )
                env.camera.lookat[:] = d.body(cube).xpos + [0, 0, 0.03]
                env.camera.distance = 0.50
                env.camera.azimuth = 60 if side == "right" else 120
                env.camera.elevation = -15
                detail = Image.fromarray(env.render()).resize((480, 270))
                frame.paste(detail, (784, 434))
                env.camera.lookat[:] = saved_camera[0]
                env.camera.distance, env.camera.azimuth, env.camera.elevation = (
                    saved_camera[1:]
                )
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 1280, 86), fill=(20, 24, 31))
                font = ImageFont.load_default(size=22)
                draw.text(
                    (16, 10),
                    f"MUJOCO | {side.upper()} ARM / {env.robots[side].hand_side.upper()} HAND | {cube.upper()} | {phase} | t={d.time:.1f}s",
                    font=font,
                )
                draw.text(
                    (16, 46),
                    f"Lift {(d.body(cube).xpos[2] - cube_start[2]) * 100:.1f} cm | intended contacts {contact['hand_contacts']} | unwanted contacts {len(forbidden)}",
                    font=font,
                )
                if phase not in saved_phases:
                    frame.save(output_dir / (phase.lower().replace(" ", "_") + ".png"))
                    saved_phases.add(phase)
                writer.append_data(np.asarray(frame))
                next_frame += 1 / 20
    except RuntimeError as exc:
        error = str(exc)
        print(error, flush=True)
    finally:
        warnings = {str(i): int(w.number) for i, w in enumerate(d.warning) if w.number}
        minimum_lift = min((s["lift"] for s in hold), default=0.0)
        palm_verified = bool(
            controller
            and controller.palm.close_time is not None
            and controller.palm.rest_duration >= 0.6
        )
        result = dict(
            side=side,
            cube=cube,
            engine="MuJoCo",
            version=mujoco.__version__,
            duration_s=float(d.time),
            timestep_s=float(m.opt.timestep),
            yaw_degrees=yaw_degrees,
            start_pose=env.start_pose,
            hand_assignment=env.hand_assignment,
            collision_checks=checks,
            skin_probe_interval_s=0.01,
            grasp_offset_m=None if controller is None else controller.offset.tolist(),
            grasp_rotation=None if controller is None else controller.rotation.tolist(),
            object_layout=env.object_layout,
            minimum_hold_lift_m=minimum_lift,
            hold_validation_seconds=5.0,
            hold_samples=len(hold),
            max_hand_cube_depth_m=max_depth,
            max_skin_vertex_depth_m=max_skin,
            forbidden_contacts=forbidden[:20],
            forbidden_contact_count=len(forbidden),
            max_other_cube_displacement_m=max_other_motion,
            max_inactive_command_change=inactive_change,
            max_inactive_joint_drift_rad=inactive_drift,
            minimum_hold_contacts=min((s["hand_contacts"] for s in hold), default=0),
            continuously_loaded_hold_digits=sorted(
                set.intersection(*(set(s["digits"]) for s in hold))
            )
            if hold
            else [],
            peak_palm_load_N=controller.palm.peak_palm_load if controller else None,
            max_hand_torque_Nm=max_hand_torque,
            max_joint_limit_violation_rad=max_limit,
            equalities=int(m.neq),
            external_forces_zero=bool(forces_zero),
            warnings=warnings,
            command_sha256=command_hash.hexdigest(),
            palm_before_closure_verified=palm_verified,
            error=error,
            success=bool(
                error is None
                and d.time >= duration - 1e-8
                and minimum_lift > 0.12
                and len(hold) >= 4999
                and max_depth < 0.001
                and max_skin < 0.0001
                and not forbidden
                and max_other_motion < 0.002
                and inactive_change == 0
                and max_limit < 0.03
                and max_hand_torque <= 1 + 1e-10
                and m.neq == 0
                and forces_zero
                and not warnings
                and palm_verified
                and all(
                    s["hand_contacts"] >= 2
                    and s["table_contacts"] == 0
                    and "t" in s["digits"]
                    and len(s["digits"]) >= 2
                    for s in hold
                )
            ),
        )
        (output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        (output_dir / "trace.json").write_text(json.dumps(trace) + "\n")
        if writer is not None:
            writer.close()
        env.close()
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=(*SIDES, "both"), default="both")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("renders/bimanual_grasps")
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--yaw", type=float, default=0)
    parser.add_argument(
        "--object-layout",
        type=Path,
        help="Explicit layout override; never writes saved scene defaults",
    )
    args = parser.parse_args()
    results = [
        run(
            side=side,
            output_dir=args.output_dir / side,
            video=args.video,
            yaw_degrees=args.yaw,
            object_layout=args.object_layout,
        )
        for side in (SIDES if args.side == "both" else (args.side,))
    ]
    if args.check and not all(r["success"] for r in results):
        raise SystemExit("Independent grasp verification failed; see reports")


if __name__ == "__main__":
    main()
