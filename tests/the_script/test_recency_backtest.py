import joblib
import polars as pl
from sklearn.linear_model import Ridge

from the_script.evaluation.recency_backtest import run_recency_backtest
from the_script.features.nextgen import NGS_RECEIVING_FEATURE_COLUMNS
from the_script.models.preprocessing import NUMERIC_FEATURES, NumericPreprocessor


def test_backtest_uses_same_artifact_features_and_does_not_mutate_models(tmp_path):
    assert not set(NGS_RECEIVING_FEATURE_COLUMNS).intersection(NUMERIC_FEATURES)
    processed = tmp_path / "processed"
    models = tmp_path / "models"
    processed.mkdir()
    rows = []
    for season in range(2016, 2026):
        for week in range(1, 5):
            for player in range(12):
                recent = float((player + week) % 5)
                share = recent / 20
                rz = float((player + 2 * week) % 4) / 10
                target = 2 * recent + 5 * share + 3 * rz + (season - 2016) * .1
                rows.append({
                    "game_id": f"{season}_{week:02d}_BUF_NYJ",
                    "game_date": f"{season}-09-{week + 1:02d}",
                    "season": season, "week": week, "player_id": f"wr-{player}",
                    "targets": recent, "target_share": share,
                    "receptions": recent * 0.5,
                    "red_zone_target_share": rz,
                    "targets_avg_last_5": recent * 0.8,
                    "target_share_avg_last_3": share * 0.8,
                    "red_zone_target_share_avg_last_3": rz * 0.8,
                    "targets_recency_5": recent,
                    "target_share_recency_5": share,
                    "red_zone_target_share_recency_5": rz,
                    "receiving_yards": target,
                })
    frame = pl.DataFrame(rows)
    frame.write_parquet(processed / "wr_player_features.parquet")
    baseline_features = [
        "targets_avg_last_5", "target_share_avg_last_3", "red_zone_target_share_avg_last_3"
    ]
    preprocessor = NumericPreprocessor.fit(frame.filter(pl.col("season") <= 2023), baseline_features)
    model = Ridge(alpha=10).fit(
        preprocessor.transform(frame.filter(pl.col("season") <= 2023)),
        frame.filter(pl.col("season") <= 2023)["receiving_yards"].to_numpy(),
    )
    artifacts = [
        models / "wr_opponent" / "ridge",
        models / "wr_volume" / "targets" / "ridge",
        models / "wr_volume" / "receptions" / "ridge",
    ]
    for artifact in artifacts:
        artifact.mkdir(parents=True)
        joblib.dump(model, artifact / "model.joblib")
        joblib.dump(preprocessor, artifact / "preprocessor.joblib")
    old_model = (artifacts[0] / "model.joblib").read_bytes()

    result = run_recency_backtest(
        processed_directory=processed, models_directory=models,
        positions=("WR",), output_path=tmp_path / "report.json",
    )

    experiment = result["positions"]["WR"]["receiving_yards"]
    assert experiment["added_features"] == [
        "targets_recency_5", "target_share_recency_5", "red_zone_target_share_recency_5"
    ]
    assert experiment["splits"]["validation"]["baseline"]["rows"] == 48
    assert experiment["splits"]["test"]["recency"]["rows"] == 48
    assert (artifacts[0] / "model.joblib").read_bytes() == old_model
