"""Pose-feedback placement controller; commands joints, never cube state."""

import itertools

import mujoco
import numpy as np

from orca_sim.mujoco_panda import inverse_kinematics


def smooth(value):
    x = np.clip(value, 0, 1)
    return x * x * (3 - 2 * x)


def blend_rotation(first, second, weight):
    a, b = np.empty(4), np.empty(4)
    mujoco.mju_mat2Quat(a, first.ravel())
    mujoco.mju_mat2Quat(b, second.ravel())
    if a @ b < 0:
        b *= -1
    q = (1 - weight) * a + weight * b
    q /= np.linalg.norm(q)
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, q)
    return result.reshape(3, 3)


class StackController:
    """Transfer, align, lower, open, retreat and settle after a real grasp."""

    def __init__(self, env, keep_grasp_orientation=False):
        self.env = env
        self.keep_grasp_orientation = keep_grasp_orientation
        m, d = env.model, env.data
        self.red_half_size = m.geom("red_cube_geom").size.copy()
        self.blue_half_height = float(m.geom("blue_cube_geom").size[2])
        self.start_time = float(d.time)
        self.cube_start = d.body("red_cube").xpos.copy()
        self.blue_start = d.body("blue_cube").xpos.copy()
        if self.cube_start[2] < self.blue_start[2] + 0.1:
            raise RuntimeError("Cannot stack: cube was not lifted")
        self.start_rotation = d.body("red_cube").xmat.reshape(3, 3).copy()
        hand_rotation = d.site("hand_frame").xmat.reshape(3, 3).copy()
        self.relative_pos = self.start_rotation.T @ (
            d.site("hand_frame").xpos - self.cube_start
        )
        self.relative_rotation = self.start_rotation.T @ hand_rotation
        self.arm = d.ctrl[:7].copy()
        self.closed_hand = d.ctrl[7:].copy()
        self.open_hand = np.array(
            [env.start_pose["hand_qpos"][n] for n in env.actuator_joints[7:]]
        )
        # Opening fingers must not simultaneously bend the supporting wrist.
        self.wrist_index = env.actuator_joints[7:].index("right_wrist")
        self.open_hand[self.wrist_index] = self.closed_hand[self.wrist_index]
        self.rotation_feedback = np.array([1.0, 0.0, 0.0, 0.0])
        scratch = mujoco.MjData(m)
        scratch.qpos[:] = d.qpos
        scratch.qpos[env.arm_qadr] = self.arm
        mujoco.mj_forward(m, scratch)
        # Account for the measured position-servo deflection under gravity.
        self.correction = scratch.site("hand_frame").xpos - d.site("hand_frame").xpos
        self.rotation_correction = hand_rotation.T @ scratch.site(
            "hand_frame"
        ).xmat.reshape(3, 3)
        self.target_rotation = self.choose_orientation()
        self.last_update = -1.0
        self.placed_pos = self.placed_rotation = None
        self.relaxed_hand = None
        self.retreat_vector = None

    def choose_orientation(self):
        """Level a cube face without needlessly forcing its yaw to world axes."""
        candidates = []
        vertical = np.array([0.0, 0.0, 1.0])
        for axis in range(3):
            for sign in (-1, 1):
                normal = sign * self.start_rotation[:, axis]
                cross = np.cross(normal, vertical)
                cosine = float(normal @ vertical)
                if cosine < -0.999999:
                    continue  # The opposite face supplies the nearby solution.
                x, y, z = cross
                skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                level = (
                    np.eye(3) + skew + skew @ skew / (1 + cosine)
                ) @ self.start_rotation
                for yaw in (0, -np.pi / 6, np.pi / 6, -np.pi / 3, np.pi / 3):
                    c, s = np.cos(yaw), np.sin(yaw)
                    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ level
                    angle = np.arccos(
                        np.clip(
                            (np.trace(rotation @ self.start_rotation.T) - 1) / 2, -1, 1
                        )
                    )
                    candidates.append((angle, rotation))
        # Axis-aligned alternatives remain available if the nearby yaw is not
        # reachable, but an arbitrary in-plane cube yaw is valid for stacking.
        for permutation in itertools.permutations(range(3)):
            for signs in itertools.product((-1, 1), repeat=3):
                rotation = np.eye(3)[:, permutation] * signs
                if np.linalg.det(rotation) > 0.5:
                    angle = np.arccos(
                        np.clip(
                            (np.trace(rotation @ self.start_rotation.T) - 1) / 2, -1, 1
                        )
                    )
                    candidates.append((angle, rotation))
        if self.keep_grasp_orientation:
            candidates = [(0.0, self.start_rotation.copy())]
        for _, rotation in sorted(candidates, key=lambda item: item[0]):
            q = self.arm.copy()
            try:
                for s in np.linspace(0, 1, 31):
                    pos = (1 - s) * self.cube_start + s * (
                        self.blue_start + [0, 0, 0.2]
                    )
                    rot = blend_rotation(self.start_rotation, rotation, s)
                    q = inverse_kinematics(
                        self.env,
                        pos + rot @ self.relative_pos + self.correction,
                        rot @ self.relative_rotation @ self.rotation_correction,
                        seed=q,
                    )
                for z in np.linspace(0.2, self.placement_height(rotation), 21):
                    q = inverse_kinematics(
                        self.env,
                        self.blue_start
                        + [0, 0, z]
                        + rotation @ self.relative_pos
                        + self.correction,
                        rotation @ self.relative_rotation @ self.rotation_correction,
                        seed=q,
                    )
                return rotation
            except RuntimeError:
                continue
        raise RuntimeError("No continuous face-aligned stacking path from this grasp")

    def placement_height(self, rotation):
        # Use actual cube extents, including the held cube's tilt.
        return self.blue_half_height + np.abs(rotation[2]) @ self.red_half_size + 0.001

    def choose_retreat(self):
        """Check the entire withdrawal before moving into a joint-limit trap.

        A nearby stack can leave the elbow almost fully folded. In that case
        pulling back toward the base is unreachable even though placement is
        reachable; a more vertical withdrawal can retain the same hand pose.
        All IK trials use scratch data, never live robot or object state.
        """
        for backward, upward in ((0.12, 0.12), (0.08, 0.15), (0.04, 0.18), (0, 0.18)):
            vector = -backward * self.placed_rotation[:, 1] + [0, 0, upward]
            q = self.arm.copy()
            try:
                for amount in np.linspace(0, 1, 41):
                    q = inverse_kinematics(
                        self.env,
                        self.placed_pos + amount * vector,
                        self.placed_rotation,
                        seed=q,
                    )
                return vector
            except RuntimeError:
                continue
        raise RuntimeError("No continuous withdrawal path from the placed hand pose")

    def action(self):
        e, d = self.env, self.env.data
        t = d.time - self.start_time
        blue = d.body("blue_cube").xpos.copy()
        above = blue + [0, 0, 0.20]
        destination = blue + [0, 0, self.placement_height(self.target_rotation)]
        if t < 6:
            if t > 0.5 and d.body("red_cube").xpos[2] < self.blue_start[2] + 0.09:
                raise RuntimeError("Grasp lost during transfer")
            s = smooth(t / 6)
            phase = "TRANSFER"
            cube_target = self.cube_start * (1 - s) + above * s
            cube_rotation = blend_rotation(self.start_rotation, self.target_rotation, s)
        elif t < 10:
            s = smooth((t - 6) / 4)
            phase = "LOWER ON BLUE"
            cube_target = above * (1 - s) + destination * s
            cube_rotation = self.target_rotation
        else:
            phase = (
                "UNLOAD GRIP"
                if t < 11
                else "RELEASE & RETREAT"
                if t < 16
                else "SETTLE STACK"
            )
        hand = self.closed_hand.copy()
        if t >= 10:
            if self.relaxed_hand is None:
                self.relaxed_hand = np.clip(
                    d.qpos[e.qadr[7:]], e.action_space.low[7:], e.action_space.high[7:]
                )
                self.relaxed_hand[self.wrist_index] = self.closed_hand[self.wrist_index]
            if t < 11:
                opening = smooth(t - 10)
                hand = self.closed_hand * (1 - opening) + self.relaxed_hand * opening
            else:
                opening = smooth((t - 11) / 5)
                hand = self.relaxed_hand * (1 - opening) + self.open_hand * opening
        if d.time - self.last_update >= 0.02 - 1e-8:
            if t < 10:
                error = cube_target - d.body("red_cube").xpos
                self.correction = np.clip(self.correction + 0.003 * error, -0.08, 0.08)
                hand_pos = (
                    cube_target + cube_rotation @ self.relative_pos + self.correction
                )
                hand_rot = (
                    cube_rotation @ self.relative_rotation @ self.rotation_correction
                )
                # The compliant wrist and grasp can deflect as gravity changes
                # direction during transfer. Correct measured cube orientation,
                # just as position feedback corrects translational deflection.
                target_q, actual_q, error = np.empty(4), np.empty(4), np.empty(3)
                actual_rot = d.body("red_cube").xmat.reshape(3, 3)
                mujoco.mju_mat2Quat(target_q, cube_rotation.ravel())
                mujoco.mju_mat2Quat(actual_q, actual_rot.ravel())
                mujoco.mju_subQuat(error, target_q, actual_q)
                feedback_rot = np.empty(9)
                mujoco.mju_quat2Mat(feedback_rot, self.rotation_feedback)
                local_error = feedback_rot.reshape(3, 3).T @ actual_rot @ error
                if not self.keep_grasp_orientation and (
                    t < 6 or d.body("red_cube").xpos[2] > blue[2] + 0.09
                ):
                    mujoco.mju_quatIntegrate(self.rotation_feedback, local_error, 0.012)
                mujoco.mju_quat2Mat(feedback_rot, self.rotation_feedback)
                hand_rot = feedback_rot.reshape(3, 3) @ hand_rot
                self.placed_pos, self.placed_rotation = hand_pos.copy(), hand_rot.copy()
            else:
                hand_pos, hand_rot = self.placed_pos.copy(), self.placed_rotation.copy()
                if t >= 11:
                    if self.retreat_vector is None:
                        self.retreat_vector = self.choose_retreat()
                    retreat = smooth((t - 11) / 5)
                    hand_pos += retreat * self.retreat_vector
            self.arm = inverse_kinematics(e, hand_pos, hand_rot, seed=self.arm)
            self.last_update = float(d.time)
        return np.r_[self.arm, hand], phase


def stack_metrics(env):
    m, d = env.model, env.data
    red, blue, table = (
        m.geom(n).id for n in ("red_cube_geom", "blue_cube_geom", "tabletop")
    )
    cube_contacts = blue_support = 0
    max_depth = 0.0
    force = np.zeros(6)
    for i, c in enumerate(d.contact):
        pair = {c.geom1, c.geom2}
        if pair in ({red, blue}, {blue, table}):
            mujoco.mj_contactForce(m, d, i, force)
            if pair == {red, blue}:
                cube_contacts += int(force[0] > 1e-4)
                max_depth = max(max_depth, -float(c.dist))
            else:
                blue_support += int(force[0] > 1e-4)
    delta = d.body("red_cube").xpos - d.body("blue_cube").xpos
    velocity = np.empty(6)
    mujoco.mj_objectVelocity(
        m, d, mujoco.mjtObj.mjOBJ_BODY, m.body("red_cube").id, velocity, 0
    )
    return dict(
        blue_cube_pos=d.body("blue_cube").xpos.tolist(),
        stack_contacts=cube_contacts,
        blue_table_contacts=blue_support,
        stack_contact_depth_m=max_depth,
        stack_xy_error_m=float(np.linalg.norm(delta[:2])),
        stack_height_m=float(delta[2]),
        cube_linear_speed_m_s=float(np.linalg.norm(velocity[3:])),
        cube_angular_speed_rad_s=float(np.linalg.norm(velocity[:3])),
    )
