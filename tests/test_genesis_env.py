from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("genesis")
pytest.importorskip("orca_arm")

from orca_sim.genesis_env import (
    LEFT_ARM_DOFS,
    READY_LEFT_ARM_QPOS,
    READY_QPOS,
    READY_RIGHT_ARM_QPOS,
    RIGHT_ARM_DOFS,
    BimanualOrcaPandaCubeGenesisEnv,
)


def test_ready_pose_is_a_mirrored_bimanual_configuration() -> None:
    assert READY_QPOS.shape == (48,)
    np.testing.assert_allclose(READY_QPOS[LEFT_ARM_DOFS], READY_LEFT_ARM_QPOS)
    np.testing.assert_allclose(READY_QPOS[RIGHT_ARM_DOFS], READY_RIGHT_ARM_QPOS)
    np.testing.assert_allclose(
        READY_RIGHT_ARM_QPOS,
        [
            -READY_LEFT_ARM_QPOS[0],
            READY_LEFT_ARM_QPOS[1],
            -READY_LEFT_ARM_QPOS[2],
            READY_LEFT_ARM_QPOS[3],
            -READY_LEFT_ARM_QPOS[4],
            READY_LEFT_ARM_QPOS[5],
            1.6 - READY_LEFT_ARM_QPOS[6],
        ],
    )


@pytest.mark.slow
def test_scripted_grasp_lifts_cube_stably() -> None:
    env = BimanualOrcaPandaCubeGenesisEnv()
    try:
        obs, info = env.reset()
        assert obs.shape == (109,)
        assert env.action_space.shape == (48,)
        assert info["finite"]
        np.testing.assert_allclose(obs[:48], READY_QPOS, atol=1e-7)

        result = env.run_scripted_grasp()
        assert result.success, result
        assert result.finite
        assert result.lift_height_m >= 0.12
        assert result.final_hand_cube_contacts >= 4
        assert result.final_linear_speed_m_s < 0.05
        assert np.all(np.isfinite(result.final_cube_position))
    finally:
        env.close()
