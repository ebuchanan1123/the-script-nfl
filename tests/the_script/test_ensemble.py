import numpy as np
import pytest

from the_script.models.ensemble import _convex_weights


def test_convex_weights_are_nonnegative_and_favor_accurate_model() -> None:
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    predictions = np.column_stack(
        [actual, actual + 20.0, np.full_like(actual, 5.0)]
    )

    weights = _convex_weights(predictions, actual)

    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights >= 0)
    assert weights[0] > 0.99