import numpy as np
import pytest

from orca_sim.palm_grasp import PalmFirstGrasp


def contacts(count=1, force=2.0):
    return {"palm_top_contacts": count, "palm_top_load_N": force}


def test_fingers_wait_for_sustained_palm_contact():
    controller = PalmFirstGrasp(0.001)
    for time in np.arange(8, 8.5, 0.001):
        phase, _, closure = controller.command(time, contacts(0, 0))
        assert phase == "SEEK PALM CONTACT" and closure == 0
    for time in np.arange(8.5, 9.4, 0.001):
        phase, _, closure = controller.command(time, contacts())
        if controller.close_time is None:
            assert closure == 0
    assert controller.close_time >= controller.first_contact_time + 0.6
    assert controller.rest_duration >= 0.6
    assert phase == "WRAP FINGERS" and closure > 0


@pytest.mark.parametrize("reading", [contacts(0, 0), contacts(0, 2), contacts(1, 10)])
def test_no_top_contact_or_excess_pressure_does_not_trigger_closure(reading):
    controller = PalmFirstGrasp(0.001)
    with pytest.raises(RuntimeError, match="fingers remain open"):
        for time in np.arange(8, 15.502, 0.001):
            _, _, closure = controller.command(time, reading)
            assert closure == 0
    assert controller.close_time is None


def test_contact_loss_restarts_rest_timer():
    controller = PalmFirstGrasp(0.001)
    for time in np.arange(8, 8.4, 0.001):
        controller.command(time, contacts())
    controller.command(8.4, contacts(0, 0))
    assert controller.rest_duration == 0
    for time in np.arange(8.401, 8.9, 0.001):
        _, _, closure = controller.command(time, contacts())
        assert closure == 0
    assert controller.close_time is None


def test_wrap_finishes_before_lift():
    controller = PalmFirstGrasp(0.001, grip=0.8)
    for time in np.arange(8, 9, 0.001):
        controller.command(time, contacts())
    phase, z, closure = controller.command(controller.close_time + 3, contacts())
    assert phase == "LIFT" and closure == pytest.approx(0.8)
    assert z == controller.lift_z
    phase, z, closure = controller.command(controller.hold_time, contacts())
    assert phase == "HOLD" and z == pytest.approx(controller.lift_z + 0.18)
    assert controller.stack_time == controller.hold_time + 3
