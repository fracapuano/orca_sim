from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("genesis")
pytest.importorskip("orca_arm")

from orca_sim.genesis_env import BimanualOrcaPandaCubeGenesisEnv


@pytest.mark.slow
def test_scripted_grasp_lifts_cube_stably() -> None:
    env = BimanualOrcaPandaCubeGenesisEnv()
    try:
        obs, info = env.reset()
        assert obs.shape == (109,)
        assert env.action_space.shape == (48,)
        assert info["finite"]

        result = env.run_scripted_grasp()
        assert result.success, result
        assert result.finite
        assert result.lift_height_m >= 0.12
        assert result.final_hand_cube_contacts >= 4
        assert result.final_linear_speed_m_s < 0.05
        assert np.all(np.isfinite(result.final_cube_position))
    finally:
        env.close()
