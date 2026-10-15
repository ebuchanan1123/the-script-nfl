import polars as pl
import pytest

from the_script.models.preprocessing import NumericPreprocessor


def test_preprocessing_statistics_use_training_rows_only() -> None:
    train = pl.DataFrame({"feature": [1.0, None, 3.0]})
    future = pl.DataFrame({"feature": [1000.0, None]})
    preprocessor = NumericPreprocessor.fit(train, ["feature"])

    transformed = preprocessor.transform(future)

    assert preprocessor.imputer.statistics_.tolist() == [2.0]
    assert preprocessor.scaler.mean_.tolist() == [2.0]
    assert transformed[1, 0] == pytest.approx(0.0)


def test_current_game_outcomes_are_rejected_as_features() -> None:
    frame = pl.DataFrame({"receiving_yards": [10.0]})

    with pytest.raises(ValueError, match="Current-game"):
        NumericPreprocessor.fit(frame, ["receiving_yards"])