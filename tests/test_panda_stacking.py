"""Placement must level a cube face, not needlessly snap its in-plane yaw."""

import numpy as np
import pytest

from orca_sim import panda_stacking


@pytest.mark.parametrize("tilt", [0.0, 0.14, -0.2])
@pytest.mark.parametrize("keep", [False, True])
def test_placement_orientation_preserves_yaw_or_full_grasp_pose(
    monkeypatch, tilt, keep
):
    yaw = 0.61
    c, s = np.cos(yaw), np.sin(yaw)
    yaw_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    c, s = np.cos(tilt), np.sin(tilt)
    tilt_rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    controller = panda_stacking.StackController.__new__(panda_stacking.StackController)
    controller.start_rotation = yaw_rotation @ tilt_rotation
    controller.keep_grasp_orientation = keep
    controller.arm = np.zeros(7)
    controller.cube_start = np.array([0.16, 0.30, 0.55])
    controller.blue_start = np.array([0.0, 0.30, 0.353])
    controller.red_half_size = np.full(3, 0.028)
    controller.blue_half_height = 0.028
    controller.relative_pos = np.array([0.0, -0.15, 0.1])
    controller.correction = np.zeros(3)
    controller.relative_rotation = controller.rotation_correction = np.eye(3)
    controller.env = None
    monkeypatch.setattr(
        panda_stacking, "inverse_kinematics", lambda *args, seed: seed.copy()
    )
    rotation = controller.choose_orientation()
    np.testing.assert_allclose(
        rotation, controller.start_rotation if keep else yaw_rotation, atol=1e-12
    )


@pytest.mark.parametrize("tilt", [0.0, 0.2, -0.3])
@pytest.mark.parametrize("half_size", [0.028, 0.035])
def test_tilted_cube_placement_height_clears_blue_top(tilt, half_size):
    from itertools import product

    c, s = np.cos(tilt), np.sin(tilt)
    rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    controller = panda_stacking.StackController.__new__(panda_stacking.StackController)
    controller.red_half_size = np.full(3, half_size)
    controller.blue_half_height = half_size
    height = controller.placement_height(rotation)
    lowest_corner = min(
        (rotation @ (half_size * np.array(p)))[2] for p in product((-1, 1), repeat=3)
    )
    assert height + lowest_corner == pytest.approx(half_size + 0.001)


def test_retreat_checks_complete_path_and_uses_vertical_fallback(monkeypatch):
    controller = panda_stacking.StackController.__new__(panda_stacking.StackController)
    controller.env = None
    controller.arm = np.zeros(7)
    controller.placed_pos = np.array([0.0, 0.1, 0.55])
    controller.placed_rotation = np.eye(3)
    checked = []

    def ik(env, position, rotation, *, seed):
        checked.append(position.copy())
        if position[1] < 0.1 - 1e-8:
            raise RuntimeError("Elbow limit")
        return seed.copy()

    monkeypatch.setattr(panda_stacking, "inverse_kinematics", ik)
    np.testing.assert_allclose(controller.choose_retreat(), [0, 0, 0.18])
    np.testing.assert_allclose(checked[-1], [0, 0.1, 0.73])
    np.testing.assert_array_equal(controller.arm, np.zeros(7))


def test_retreat_refuses_when_no_complete_path_is_reachable(monkeypatch):
    controller = panda_stacking.StackController.__new__(panda_stacking.StackController)
    controller.env = None
    controller.arm = np.zeros(7)
    controller.placed_pos = np.array([0.0, 0.1, 0.55])
    controller.placed_rotation = np.eye(3)

    def ik(*args, **kwargs):
        raise RuntimeError("Unreachable")

    monkeypatch.setattr(panda_stacking, "inverse_kinematics", ik)
    with pytest.raises(RuntimeError, match="No continuous withdrawal"):
        controller.choose_retreat()
