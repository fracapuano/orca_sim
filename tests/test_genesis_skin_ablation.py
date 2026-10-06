"""The contact-only ablation must leave all unrelated collision pairs intact."""

import numpy as np

from scripts.genesis_skin_collision_ablation import without_skin_pairs


def test_skin_ablation_preserves_other_pairs_and_source():
    original = np.array(
        [[-1, 0, 1, 2], [-1, -1, 3, 4], [-1, -1, -1, 5], [-1, -1, -1, -1]]
    )
    saved = original.copy()
    disabled = without_skin_pairs(original, [1, 3])
    np.testing.assert_array_equal(original, saved)
    assert np.all(disabled[[1, 3], :] == -1)
    assert np.all(disabled[:, [1, 3]] == -1)
    np.testing.assert_array_equal(
        disabled[np.ix_([0, 2], [0, 2])], original[np.ix_([0, 2], [0, 2])]
    )


def test_empty_skin_selection_is_identity():
    original = np.array([[-1, 0], [-1, -1]])
    np.testing.assert_array_equal(without_skin_pairs(original, []), original)
