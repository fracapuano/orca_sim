"""Joint-target-only, two-arm pickup/place/stack sequence in native MuJoCo."""

from __future__ import annotations

import mujoco
import numpy as np

from orca_sim.demos.bimanual_grasp import ContactAudit, PalmLift
from orca_sim.mujoco_panda import PACKAGE, TABLE_HEIGHT
from orca_sim.panda_stacking import blend_rotation, smooth

STACK_LAYOUT = PACKAGE / "assets/mujoco/bimanual_stack_layout.json"


def level_rotation(rotation):
    """Closest cube-face-horizontal orientation, retaining its in-plane yaw."""
    axis = np.argmax(np.abs(rotation[2]))
    normal = rotation[:, axis] * np.sign(rotation[2, axis])
    x, y, z = np.cross(normal, [0, 0, 1])
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return (np.eye(3) + skew + skew @ skew / (1 + normal[2])) @ rotation


class PlaceCube:
    """Transfer a held cube, release onto real support, then withdraw.

    Uses measured grasp geometry and slow pose feedback. No cube state writes,
    kinematic attachment, collision exclusions, force or actuator-limit changes.
    """

    def __init__(self, env, side, cube, destination_xy, support=None):
        self.env, self.side, self.cube, self.support = env, side, cube, support
        self.robot = env.robots[side]
        m, d = env.model, env.data
        self.start_time = float(d.time)
        self.start = d.body(cube).xpos.copy()
        if self.start[2] < TABLE_HEIGHT + 0.13:
            raise RuntimeError(f"{side}: cannot transfer a cube that was not lifted")
        self.start_rotation = d.body(cube).xmat.reshape(3, 3).copy()
        # Preserve the sideways grasp instead of rolling the forearm down
        # toward the table to force a face-horizontal held cube.
        self.rotation = self.start_rotation.copy()
        site = d.site(self.robot.hand_site)
        self.relative_pos = self.start_rotation.T @ (site.xpos - self.start)
        self.relative_rot = self.start_rotation.T @ site.xmat.reshape(3, 3)
        self.arm = d.ctrl[self.robot.actuator_indices[:7]].copy()
        self.closed = d.ctrl[self.robot.actuator_indices[7:]].copy()
        self.open = m.key("orcapanda_home").ctrl[self.robot.actuator_indices[7:]].copy()
        self.wrist = self.robot.actuator_joints[7:].index(
            self.robot.hand_side + "_wrist"
        )
        self.open[self.wrist] = self.closed[self.wrist]
        self.half_size = m.geom(cube + "_geom").size.copy()
        # A tilted cube pivots about its lowest corner as it settles. Aim the
        # held centre upstream of that geometric shift instead of deliberately
        # rolling the Panda wrist down toward the tabletop. This is only a
        # placement prediction: final support/position are measured after release.
        corner = -np.sign(self.rotation[2]) * self.half_size
        self.landing_shift = (self.rotation - level_rotation(self.rotation)) @ corner
        self.xy = np.array(destination_xy, dtype=float)
        self.above = np.r_[self.xy, max(self.start[2], TABLE_HEIGHT + 0.24)]
        scratch = mujoco.MjData(m)
        scratch.qpos[:] = d.qpos
        scratch.qpos[self.robot.arm_qadr] = self.arm
        mujoco.mj_forward(m, scratch)
        self.correction = scratch.site(self.robot.hand_site).xpos - site.xpos
        self.rotation_correction = site.xmat.reshape(3, 3).T @ scratch.site(
            self.robot.hand_site
        ).xmat.reshape(3, 3)
        self.rotation_feedback = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_update = -np.inf
        self.placed_pos = self.placed_rot = self.relaxed = self.retreat = None
        # Preflight complete path on scratch state, preserving the IK branch.
        q = self.arm.copy()
        for amount in np.linspace(0, 1, 41):
            rot = blend_rotation(self.start_rotation, self.rotation, amount)
            pos = self.start * (1 - amount) + self.above * amount
            q = self.solve(pos, rot, q)
        for amount in np.linspace(0, 1, 31):
            pos = self.above * (1 - amount) + self.destination() * amount
            q = self.solve(pos, self.rotation, q)

    def destination(self):
        d, m = self.env.data, self.env.model
        surface = TABLE_HEIGHT
        xy = self.xy
        if self.support is not None:
            body = d.body(self.support)
            xy = body.xpos[:2]
            surface = (
                body.xpos[2]
                + np.abs(body.xmat.reshape(3, 3)[2])
                @ m.geom(self.support + "_geom").size
            )
        # The wrapped thumb extends below the held cube. Leave space for it
        # while releasing onto another cube; never use the base block as a
        # finger support. Both bodies settle under normal gravity/contact.
        clearance = 0.014 if self.support else 0.005
        return np.r_[
            xy - self.landing_shift[:2],
            surface + np.abs(self.rotation[2]) @ self.half_size + clearance,
        ]

    def solve(self, position, rotation, seed):
        return self.env.solve_ik(
            self.side,
            position + rotation @ self.relative_pos + self.correction,
            rotation @ self.relative_rot @ self.rotation_correction,
            seed=seed,
        )

    def choose_retreat(self):
        for back, up in ((0.12, 0.12), (0.08, 0.15), (0.04, 0.18), (0, 0.18)):
            vector = -back * self.placed_rot[:, 1] + [0, 0, up]
            q = self.arm.copy()
            try:
                for s in np.linspace(0, 1, 41):
                    q = self.env.solve_ik(
                        self.side, self.placed_pos + s * vector, self.placed_rot, seed=q
                    )
                return vector
            except RuntimeError:
                continue
        raise RuntimeError(f"{self.side}: no continuous withdrawal path")

    def command(self):
        e, d = self.env, self.env.data
        age = d.time - self.start_time
        hand = self.closed.copy()
        if age < 6:
            amount = smooth(age / 6)
            pos = self.start * (1 - amount) + self.above * amount
            rot = blend_rotation(self.start_rotation, self.rotation, amount)
            phase = "TRANSFER"
        elif age < 10:
            amount = smooth((age - 6) / 4)
            pos = self.above * (1 - amount) + self.destination() * amount
            rot = self.rotation
            phase = "LOWER"
        else:
            phase = (
                "UNLOAD"
                if age < 11
                else "RELEASE AND RETREAT"
                if age < 16
                else "SETTLE"
            )
            if self.relaxed is None:
                self.relaxed = np.clip(
                    d.qpos[self.robot.hand_qadr],
                    self.robot.action_space.low[7:],
                    self.robot.action_space.high[7:],
                )
                self.relaxed[self.wrist] = self.closed[self.wrist]
            if age < 11:
                amount = smooth(age - 10)
                hand = self.closed * (1 - amount) + self.relaxed * amount
            else:
                amount = smooth((age - 11) / 5)
                hand = self.relaxed * (1 - amount) + self.open * amount
        if d.time - self.last_update >= 0.02 - 1e-8:
            if age < 10:
                self.correction = np.clip(
                    self.correction + 0.003 * (pos - d.body(self.cube).xpos),
                    -0.08,
                    0.08,
                )
                target_q, actual_q, error, feedback = (
                    np.empty(4),
                    np.empty(4),
                    np.empty(3),
                    np.empty(9),
                )
                actual = d.body(self.cube).xmat.reshape(3, 3)
                mujoco.mju_mat2Quat(target_q, rot.ravel())
                mujoco.mju_mat2Quat(actual_q, actual.ravel())
                mujoco.mju_subQuat(error, target_q, actual_q)
                mujoco.mju_quat2Mat(feedback, self.rotation_feedback)
                local_error = feedback.reshape(3, 3).T @ actual @ error
                if age < 6 or d.body(self.cube).xpos[2] > self.destination()[2] + 0.03:
                    mujoco.mju_quatIntegrate(self.rotation_feedback, local_error, 0.012)
                mujoco.mju_quat2Mat(feedback, self.rotation_feedback)
                self.placed_pos = pos + rot @ self.relative_pos + self.correction
                self.placed_rot = (
                    feedback.reshape(3, 3)
                    @ rot
                    @ self.relative_rot
                    @ self.rotation_correction
                )
                self.arm = e.solve_ik(
                    self.side, self.placed_pos, self.placed_rot, seed=self.arm
                )
            else:
                pos = self.placed_pos.copy()
                if age >= 11:
                    if self.retreat is None:
                        self.retreat = self.choose_retreat()
                    pos += smooth((age - 11) / 5) * self.retreat
                self.arm = e.solve_ik(self.side, pos, self.placed_rot, seed=self.arm)
            self.last_update = float(d.time)
        return np.r_[self.arm, hand], phase


class BimanualStack:
    """Serialized shared-workspace access; inactive hands keep their targets."""

    def __init__(self, env):
        self.env = env
        self.phase = "RIGHT PICK RED"
        self.start_time = float(env.data.time)
        # Calibrate both grasp orientations from the same saved home, before
        # gravity/servo deflection during the other hand's turn. Execution is
        # still fully dynamic and uses fresh palm contact feedback every step.
        self.pickers = {
            side: PalmLift(env, side, cube)
            for side, cube in (("right", "red_cube"), ("left", "blue_cube"))
        }
        self.controller = self.pickers["right"]
        self.action = env.data.ctrl.copy()
        self.events = [dict(t=self.start_time, phase=self.phase)]
        self.done = False
        self.park_start = None

    def transition(self, phase):
        self.events.append(dict(t=float(self.env.data.time), phase=phase))
        self.phase, self.start_time = phase, float(self.env.data.time)

    def command(self):
        e, d = self.env, self.env.data
        age = d.time - self.start_time
        if self.phase in ("RIGHT PICK RED", "LEFT PICK BLUE"):
            side = "right" if self.phase == "RIGHT PICK RED" else "left"
            action, detail = self.controller.command(age)
            indices = e.robots[side].actuator_indices
            if side == "left" and self.controller.palm.close_time is not None:
                # Give the middle finger time to wrap before thumb opposition;
                # otherwise their tips can cross in this two-loaded-hand state.
                # Final targets and all native gains/force limits are unchanged.
                closure = smooth((age - self.controller.palm.close_time - 1.7) / 2)
                for joint, target in (
                    ("t-abd", 0.9 * (1 - closure)),
                    ("t-cmc", -0.5 * (1 - closure) + 0.55 * closure),
                ):
                    action[indices[self.controller.joint_index[joint]]] = target
            self.action[indices] = action[indices]
            if age >= 23:
                cube = "red_cube" if side == "right" else "blue_cube"
                contact = self.controller.audit.inspect()
                if (
                    d.body(cube).xpos[2] < TABLE_HEIGHT + 0.16
                    or contact["hand_contacts"] < 2
                ):
                    raise RuntimeError(f"{side}: grasp/lift not established")
                if side == "right":
                    self.transition("LEFT PICK BLUE")
                    self.controller = self.pickers["left"]
                else:
                    self.transition("LEFT PLACE BLUE ON X")
                    self.controller = PlaceCube(
                        e, "left", "blue_cube", d.body("stack_target").xpos[:2]
                    )
        elif self.phase in ("LEFT PLACE BLUE ON X", "RIGHT STACK RED ON BLUE"):
            side = "left" if self.phase == "LEFT PLACE BLUE ON X" else "right"
            action, detail = self.controller.command()
            self.action[e.robots[side].actuator_indices] = action
            if age >= 18:
                if side == "left":
                    self.transition("LEFT CLEAR WORKSPACE")
                    self.park_start = self.action[
                        e.robots["left"].actuator_indices
                    ].copy()
                else:
                    self.transition("SETTLE STACK")
        elif self.phase == "LEFT CLEAR WORKSPACE":
            indices = e.robots["left"].actuator_indices
            amount = smooth(age / 5)
            home = e.model.key("orcapanda_home").ctrl[indices]
            self.action[indices] = self.park_start * (1 - amount) + home * amount
            detail = "RETURN LEFT HOME"
            if age >= 5:
                xy_error = np.linalg.norm(
                    d.body("blue_cube").xpos[:2] - d.body("stack_target").xpos[:2]
                )
                if xy_error > 0.015:
                    raise RuntimeError(f"Blue cube missed X by {xy_error:.4f} m")
                self.transition("RIGHT STACK RED ON BLUE")
                self.controller = PlaceCube(
                    e,
                    "right",
                    "red_cube",
                    d.body("blue_cube").xpos[:2],
                    support="blue_cube",
                )
        else:
            detail = "BOTH HANDS RELEASED"
            self.done = age >= 5
        return self.action.copy(), f"{self.phase} | {detail}"


class StackContactAudit:
    """Allow only each hand's assigned cube and real cube support contacts."""

    def __init__(self, env):
        self.env = env
        self.roles = ContactAudit(env, "right", "red_cube").roles
        self.allowed = {
            frozenset(pair)
            for pair in (
                ("right_hand", "red_cube"),
                ("left_hand", "blue_cube"),
                ("red_cube", "table"),
                ("blue_cube", "table"),
                ("red_cube", "blue_cube"),
            )
        }

    def inspect(self):
        m, d = self.env.model, self.env.data
        forbidden, depth, hand_contacts = [], 0.0, 0
        force = np.empty(6)
        for i, c in enumerate(d.contact):
            mujoco.mj_contactForce(m, d, i, force)
            roles = frozenset((self.roles[c.geom1], self.roles[c.geom2]))
            if c.dist < -1e-7 or force[0] > 1e-5:
                if roles not in self.allowed:
                    forbidden.append(
                        dict(
                            roles=sorted(roles),
                            bodies=[
                                m.body(m.geom_bodyid[g]).name
                                for g in (c.geom1, c.geom2)
                            ],
                            depth_m=max(0.0, -float(c.dist)),
                            force_N=float(force[0]),
                        )
                    )
                depth = max(depth, -float(c.dist))
                hand_contacts += int(bool(roles & {"right_hand", "left_hand"}))
        return dict(forbidden=forbidden, max_depth_m=depth, hand_contacts=hand_contacts)
