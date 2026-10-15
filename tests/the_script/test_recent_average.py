import polars as pl
import pytest

from the_script.evaluation.backtest import chronological_splits
from the_script.evaluation.metrics import holdout_metrics
from the_script.models.baseline import evaluate_recent_average


def test_chronological_splits_keep_future_seasons_separate() -> None:
    frame = pl.DataFrame({"season": [2016, 2023, 2024, 2025], "value": [1, 2, 3, 4]})

    splits = chronological_splits(frame)

    assert splits["train"]["value"].to_list() == [1, 2]
    assert splits["validation"]["value"].to_list() == [3]
    assert splits["test"]["value"].to_list() == [4]


def test_recent_average_reports_expected_errors() -> None:
    frame = pl.DataFrame(
        {
            "season": [2023, 2024, 2024, 2025, 2025],
            "receiving_yards": [10, 20, 40, 30, 50],
            "yards_avg_last_5": [None, 10.0, 50.0, 20.0, 70.0],
        }
    )

    results = evaluate_recent_average(frame)

    assert results["train_rows"] == 1
    assert results["validation"] == {
        "rows": 2,
        "mae": 10.0,
        "rmse": 10.0,
        "median_absolute_error": 10.0,
    }
    assert results["test"]["rows"] == 2
    assert results["test"]["mae"] == 15.0
    assert results["test"]["rmse"] == pytest.approx(10 * (2.5**0.5))
    assert results["test"]["median_absolute_error"] == 15.0


def test_holdout_metrics_reports_common_history_cohort() -> None:
    frame = pl.DataFrame(
        {
            "receiving_yards": [10.0, 20.0],
            "yards_avg_last_5": [None, 15.0],
        }
    )

    metrics = holdout_metrics(frame, pl.Series([12.0, 18.0]))

    assert metrics["rows"] == 2
    assert metrics["common_history_cohort"] == {
        "rows": 1,
        "mae": 2.0,
        "rmse": 2.0,
        "median_absolute_error": 2.0,
    }