"""Explicit, bounded preload control for an already acquired native v2 grasp.

This is not a grasp planner, slip compensator, or object attachment. It reads
only hand joint positions and actuator targets, and outputs position targets.
Latch after settling with opposing contacts; ramp the requested multiplier.
Arbitrary retargeted finger motion is still allowed to move/release an object.
"""

from __future__ import annotations

import mujoco
import numpy as np


class GraspPreload:
    """Scale the existing position-servo preload with a common saturation bound.

    Independent motor clipping changes the torque pattern. Instead, the first
    motor to run out of torque/target headroom limits the *whole* increase.
    Wrist/arm actuators are not modified. Native gains and limits are unchanged.
    Re-latch after changing the grasp; this is intended for a static squeeze,
    not compensation for arbitrary arm accelerations or changing contacts.
    """

    def __init__(self, model, data, side, *, force_headroom=0.95):
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        if not np.isfinite(force_headroom) or not 0 < force_headroom <= 1:
            raise ValueError("force_headroom must be in (0, 1]")
        selected = []
        for i in range(model.nu):
            if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            joint = model.joint(int(model.actuator_trnid[i, 0]))
            if joint.name.startswith(side + "_") and "-" in joint.name:
                selected.append(i)
        self.indices = np.asarray(selected, dtype=int)
        if len(self.indices) != 16:
            raise ValueError("Expected the sixteen native v2 finger actuators")
        i = self.indices
        kp = model.actuator_gainprm[i, 0]
        expected_bias = np.zeros((len(i), 10))
        expected_bias[:, 1] = -kp
        if (
            np.any(kp <= 0)
            or not np.isfinite(kp).all()
            or not np.isfinite(model.actuator_forcerange[i]).all()
            or not np.isfinite(model.actuator_ctrlrange[i]).all()
            or not np.all(model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_FIXED)
            or not np.all(model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE)
            or not np.all(model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_NONE)
            or not np.allclose(model.actuator_biasprm[i], expected_bias)
            or not np.all(model.actuator_forcelimited[i])
            or not np.all(model.actuator_ctrllimited[i])
            or not np.allclose(model.actuator_gear[i, 0], 1)
            or not np.allclose(model.actuator_gear[i, 1:], 0)
        ):
            raise ValueError(
                "Requires native unit-gear position servos with finite limits"
            )
        joints = model.actuator_trnid[i, 0]
        if not np.all(model.jnt_type[joints] == mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Requires hinge joint actuators")
        self.base = data.ctrl.copy()
        self.position = data.qpos[model.jnt_qposadr[joints]].copy()
        self.preload = self.base[i] - self.position
        if not np.isfinite(self.base).all() or not np.isfinite(self.position).all():
            raise ValueError("Cannot latch non-finite state or commands")
        torque = kp * self.preload
        active = np.abs(self.preload) > 1e-10
        if not np.any(active):
            raise ValueError("No existing preload to increase")
        force_limit = np.where(
            torque > 0, model.actuator_forcerange[i, 1], model.actuator_forcerange[i, 0]
        )
        target_limit = np.where(
            self.preload > 0,
            model.actuator_ctrlrange[i, 1],
            model.actuator_ctrlrange[i, 0],
        )
        self.max_multiplier = float(
            min(
                np.min(force_headroom * force_limit[active] / torque[active]),
                np.min(
                    (target_limit[active] - self.position[active])
                    / self.preload[active]
                ),
            )
        )
        if self.max_multiplier < 1:
            raise ValueError(
                "Grasp already exceeds the reserved torque/target headroom"
            )
        self.applied_multiplier = 1.0

    def command(self, multiplier, *, other_targets=None):
        """Return actuator-ordered targets; report actual scale in applied_multiplier."""
        if not np.isfinite(multiplier) or multiplier < 1:
            raise ValueError("Squeeze multiplier must be finite and at least one")
        target = (
            self.base.copy()
            if other_targets is None
            else np.array(other_targets, dtype=float, copy=True)
        )
        if target.shape != self.base.shape or not np.isfinite(target).all():
            raise ValueError("Invalid actuator target vector")
        self.applied_multiplier = min(float(multiplier), self.max_multiplier)
        target[self.indices] = self.position + self.applied_multiplier * self.preload
        return target
