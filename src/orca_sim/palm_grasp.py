"""Contact-gated palm-first grasp schedule, with bounded vertical admittance.

Outputs only a hand-height offset and closure fraction. The caller resolves
arm IK on scratch data and applies ordinary native position-actuator targets.
"""

import numpy as np


def smooth(value):
    x = np.clip(value, 0, 1)
    return x * x * (3 - 2 * x)


class PalmFirstGrasp:
    def __init__(self, dt, grip=1.0):
        self.dt = dt
        self.grip = grip
        self.z = 0.025
        self.load = 0.0
        self.rest_duration = 0.0
        self.first_contact_time = None
        self.close_time = None
        self.lift_z = None
        self.peak_palm_load = 0.0
        self.rest_min_force = self.rest_max_force = None

    @property
    def stack_time(self):
        return float("inf") if self.close_time is None else self.close_time + 10

    @property
    def hold_time(self):
        return float("inf") if self.close_time is None else self.close_time + 7

    def command(self, time, contacts):
        if time < 8:
            return "LOWER OPEN PALM", 0.16 - 0.135 * smooth((time - 5) / 3), 0.0
        top = contacts["palm_top_contacts"] > 0
        force = contacts["palm_top_load_N"]
        self.peak_palm_load = max(self.peak_palm_load, force)
        self.load += (force - self.load) * min(1.0, self.dt / 0.03)
        if self.first_contact_time is None and top and force > 0.2:
            self.first_contact_time = float(time)
        if self.close_time is None or time < self.close_time + 3:
            if self.first_contact_time is None:
                self.z -= 0.006 * self.dt
            else:
                # Rest against the top rather than keep pushing the descent
                # target through it. This does not change gains or force caps.
                self.z += np.clip(0.0002 * (force - 2.0), -0.0004, 0.008) * self.dt
            if not -0.025 <= self.z <= 0.07:
                raise RuntimeError(
                    "Palm contact search exceeded its bounded height range"
                )
        if self.close_time is None:
            good = top and 0.2 <= self.load <= 5.0
            self.rest_duration = self.rest_duration + self.dt if good else 0.0
            if good:
                self.rest_min_force = (
                    force
                    if self.rest_min_force is None
                    else min(self.rest_min_force, force)
                )
                self.rest_max_force = (
                    force
                    if self.rest_max_force is None
                    else max(self.rest_max_force, force)
                )
            else:
                self.rest_min_force = self.rest_max_force = None
            if self.rest_duration >= 0.6:
                self.close_time = float(time)
            elif time >= 15.5:
                raise RuntimeError("No stable palm-on-top contact; fingers remain open")
            return (
                "PALM REST"
                if self.first_contact_time is not None
                else "SEEK PALM CONTACT",
                self.z,
                0.0,
            )
        age = time - self.close_time
        if age < 3:
            return "WRAP FINGERS", self.z, self.grip * smooth(age / 3)
        if self.lift_z is None:
            self.lift_z = self.z
        if age < 7:
            return "LIFT", self.lift_z + 0.18 * smooth((age - 3) / 4), self.grip
        return "HOLD", self.lift_z + 0.18, self.grip
