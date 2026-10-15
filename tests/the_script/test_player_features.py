from datetime import date, timedelta

import polars as pl
import pytest

from the_script.features.player import PLAYER_FEATURE_COLUMNS, add_player_form_features


def _player_games() -> pl.DataFrame:
    start = date(2023, 9, 1)
    rows = []
    for game_number, yards in enumerate([10, 20, 30, 40, 50, 60]):
        rows.append(
            {
                "game_id": f"g-{game_number + 1}",
                "game_date": start + timedelta(days=7 * game_number),
                "season": 2023,
                "player_id": "wr-1",
                "receiving_yards": yards,
                "targets": game_number + 2,
                "receptions": game_number + 1,
                "target_share": 0.1 + game_number * 0.02,
                "air_yards_share": 0.15 + game_number * 0.03,
                "wopr": 0.2 + game_number * 0.04,
                "snap_share": 0.5 + game_number * 0.05,
            }
        )
    rows.append(
        {
            "game_id": "other-1",
            "game_date": start,
            "season": 2023,
            "player_id": "wr-2",
            "receiving_yards": 200,
            "targets": 15,
            "receptions": 12,
            "target_share": 0.4,
            "air_yards_share": 0.5,
            "wopr": 0.6,
            "snap_share": 0.95,
        }
    )
    return pl.DataFrame(rows)


def test_shifted_windows_use_only_prior_player_games() -> None:
    featured = add_player_form_features(_player_games()).filter(pl.col("player_id") == "wr-1")

    assert featured[0, "yards_last_game"] is None
    assert featured[0, "yards_avg_last_5"] is None
    assert featured[1, "yards_last_game"] == 10
    assert featured[3, "yards_avg_last_3"] == 20
    assert featured[5, "yards_avg_last_5"] == 30
    assert featured[5, "yards_avg_last_16"] == 30
    assert featured[5, "targets_avg_last_5"] == 4
    assert featured[5, "targets_avg_last_16"] == 4
    assert featured[5, "target_share_last_game"] == 0.18
    assert featured[5, "target_share_avg_last_5"] == pytest.approx(0.14)
    assert featured[5, "air_yards_share_avg_last_3"] == pytest.approx(0.24)
    assert featured[5, "wopr_avg_last_5"] == pytest.approx(0.28)
    assert featured[5, "receptions_avg_last_5"] == 3
    assert featured[5, "receptions_avg_last_16"] == 3
    assert featured[5, "snap_share_avg_last_3"] == 0.65
    assert featured[5, "season_yards_per_game_before_game"] == 30
    assert featured[5, "games_played_before_game"] == 5


def test_current_target_cannot_change_current_features() -> None:
    original = _player_games()
    mutated = original.with_columns(
        pl.when(pl.col("game_id") == "g-6")
        .then(pl.lit(999))
        .otherwise(pl.col("receiving_yards"))
        .alias("receiving_yards")
    )

    original_row = add_player_form_features(original).filter(pl.col("game_id") == "g-6")
    mutated_row = add_player_form_features(mutated).filter(pl.col("game_id") == "g-6")

    assert original_row.select(PLAYER_FEATURE_COLUMNS).equals(
        mutated_row.select(PLAYER_FEATURE_COLUMNS)
    )


def test_future_rows_do_not_change_historical_features() -> None:
    complete = add_player_form_features(_player_games())
    historical_source = _player_games().filter(pl.col("game_date") <= date(2023, 9, 22))
    truncated = add_player_form_features(historical_source)

    historical_complete = complete.filter(pl.col("game_date") <= date(2023, 9, 22))
    assert historical_complete.equals(truncated)


def test_player_histories_are_isolated() -> None:
    first_game = add_player_form_features(_player_games()).filter(pl.col("game_id") == "other-1")

    assert first_game[0, "yards_last_game"] is None
    assert first_game[0, "yards_avg_last_3"] is None