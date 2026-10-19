import polars as pl
import pytest

from the_script.evaluation.wr_rolling import MODEL_COLUMNS, evaluate_rolling_validation


def test_expanding_validation_reports_fold_and_usage_metrics(tmp_path):
    rows = []
    features = []
    for week in range(1, 19):
        for player in range(4):
            actual = float(week * 3 + player)
            rows.append({
                "game_id": f"2024_{week:02d}_BUF_NYJ",
                "player_id": f"wr-{player}",
                "receiving_yards": actual,
                "ridge": actual + 2,
                "xgboost": actual + 5,
                "numerical_mlp": actual + 1,
                "decomposition": actual + 3,
            })
            features.append({
                "game_id": f"2024_{week:02d}_BUF_NYJ",
                "player_id": f"wr-{player}",
                "targets_avg_last_5": [2.0, 5.0, 9.0, None][player],
                "yards_avg_last_5": actual - 10,
            })
    predictions_path = tmp_path / "validation.parquet"
    features_path = tmp_path / "features.parquet"
    output_path = tmp_path / "diagnostics.json"
    pl.DataFrame(rows).write_parquet(predictions_path)
    pl.DataFrame(features).write_parquet(features_path)

    result = evaluate_rolling_validation(predictions_path, features_path, output_path)

    assert len(MODEL_COLUMNS) == 4
    assert [fold["score_weeks"] for fold in result["folds"]] == ["7-10", "11-14", "15-18"]
    assert result["scored_rows"] == 48
    assert result["common_cohort_rows"] == 48
    assert set(result["by_pregame_usage_tier"]) == {
        "high (8+ targets/game)", "low (<4 targets/game)",
        "mid (4-8 targets/game)", "unknown",
    }
    assert result["overall"]["rolling_blend"]["mae"] < result["overall"]["recent_average"]["mae"]
    assert result["folds"][0]["bias_calibration"] is None
    assert result["folds"][1]["bias_calibration"]["fit_on_prior_scored_weeks"]
    assert result["bias_calibration_on_later_folds"]["rows"] == 32
    assert output_path.exists()


def test_rejects_non_2024_rows(tmp_path):
    predictions_path = tmp_path / "predictions.parquet"
    features_path = tmp_path / "features.parquet"
    prediction = {
        "game_id": ["2025_07_BUF_NYJ"], "player_id": ["wr-1"],
        "receiving_yards": [20.0],
        **{name: [20.0] for name in MODEL_COLUMNS},
    }
    pl.DataFrame(prediction).write_parquet(predictions_path)
    pl.DataFrame({
        "game_id": ["2025_07_BUF_NYJ"], "player_id": ["wr-1"],
        "targets_avg_last_5": [5.0], "yards_avg_last_5": [15.0],
    }).write_parquet(features_path)

    with pytest.raises(ValueError, match="2024-only"):
        evaluate_rolling_validation(predictions_path, features_path)