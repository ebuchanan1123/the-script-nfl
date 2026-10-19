from datetime import date, timedelta

import polars as pl
import pytest

from the_script.features.advanced import add_usage_quality_features


def _usage_frame() -> tuple[pl.DataFrame, pl.DataFrame]:
    frame = pl.DataFrame({
        "game_id": ["g1", "g2", "g3"],
        "game_date": [date(2024, 9, 1) + timedelta(days=7 * i) for i in range(3)],
        "player_id": ["wr-1"] * 3,
        "team": ["BUF"] * 3,
    })
    plays = []
    # Older game: player has two of four team targets; both are in the red zone.
    plays.extend([
        {"game_id": "g1", "receiver_player_id": "wr-1", "posteam": "BUF", "air_yards": 3, "yardline_100": 10},
        {"game_id": "g1", "receiver_player_id": "wr-1", "posteam": "BUF", "air_yards": 5, "yardline_100": 15},
        {"game_id": "g1", "receiver_player_id": "wr-2", "posteam": "BUF", "air_yards": 7, "yardline_100": 40},
        {"game_id": "g1", "receiver_player_id": "wr-2", "posteam": "BUF", "air_yards": None, "yardline_100": 30},
    ])
    # Most recent completed game: one of four targets, none in the red zone.
    plays.extend([
        {"game_id": "g2", "receiver_player_id": "wr-1", "posteam": "BUF", "air_yards": 5, "yardline_100": 40},
        {"game_id": "g2", "receiver_player_id": "wr-2", "posteam": "BUF", "air_yards": 7, "yardline_100": 10},
        {"game_id": "g2", "receiver_player_id": "wr-2", "posteam": "BUF", "air_yards": 7, "yardline_100": 20},
        {"game_id": "g2", "receiver_player_id": "wr-2", "posteam": "BUF", "air_yards": 7, "yardline_100": 40},
    ])
    # Target-game outcomes must not leak into its features.
    plays.extend([
        {"game_id": "g3", "receiver_player_id": "wr-2", "posteam": "BUF", "air_yards": 7, "yardline_100": 10},
    ])
    return frame, pl.DataFrame(plays)


def test_target_share_and_red_zone_recency_use_prior_games_only():
    frame, plays = _usage_frame()
    result = add_usage_quality_features(frame, plays).sort("game_id")
    row = result.filter(pl.col("game_id") == "g3").row(0, named=True)

    expected_target_share = (0.25 + 0.75 * 0.5) / (0.75 + 1)
    expected_red_zone_share = (0.0 + 0.75 * 1.0) / (0.75 + 1)
    assert row["target_share_recency_5"] == pytest.approx(expected_target_share)
    assert row["red_zone_target_share_recency_5"] == pytest.approx(expected_red_zone_share)
    assert row["pbp_target_share_avg_last_3"] == pytest.approx((0.25 + 0.5) / 2)

    changed_target_game = plays.with_columns(
        pl.when(pl.col("game_id") == "g3")
        .then(pl.lit(1))
        .otherwise(pl.col("yardline_100"))
        .alias("yardline_100")
    )
    changed = add_usage_quality_features(frame, changed_target_game).filter(
        pl.col("game_id") == "g3"
    ).row(0, named=True)
    assert changed["target_share_recency_5"] == row["target_share_recency_5"]
    assert changed["red_zone_target_share_recency_5"] == row["red_zone_target_share_recency_5"]
