from datetime import date, timedelta

import polars as pl
import pytest

from the_script.features.matchup import add_matchup_history_features


def _frame() -> tuple[pl.DataFrame, pl.DataFrame]:
    base = date(2026, 9, 1)
    games = pl.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "game_date": [base + timedelta(days=7 * index) for index in range(3)],
            "season": [2026] * 3,
            "week": [1, 2, 3],
            "team": ["AAA"] * 3,
            "opponent": ["BBB"] * 3,
            "player_id": ["rb1"] * 3,
            "position": ["RB"] * 3,
        }
    )
    plays = []
    team_games = [
        ("g1", "AAA", 4, 0, 2, 5),
        ("g2", "AAA", 8, 1, 4, 7),
        ("g3", "AAA", 0, 0, 1, 1),
        ("g1", "BBB", 0, 0, 2, 5),
        ("g2", "BBB", 0, 1, 2, 5),
        ("g3", "BBB", 0, 0, 1, 5),
    ]
    for game, team, trailing_count, touchdowns, drives, dropbacks in team_games:
        home, away = ("AAA", "BBB")
        for index in range(10):
            series = min(drives, (index * drives // 10) + 1)
            plays.append(
                {
                    "game_id": game,
                    "game_date": base + timedelta(days=7 * int(game[-1]) - 7),
                    "season": 2026,
                    "week": int(game[-1]),
                    "posteam": team,
                    "home_team": home,
                    "away_team": away,
                    "series": float(series),
                    "series_result": "Touchdown" if series <= touchdowns else "Punt",
                    "qb_dropback": 1.0 if index < dropbacks else 0.0,
                    "rush_attempt": 0.0 if index < dropbacks else 1.0,
                    "qb_kneel": 0.0,
                    "qb_spike": 0.0,
                    "score_differential": -1.0 if index < trailing_count else 1.0,
                    "passer_player_id": f"{team}_qb" if index < dropbacks else None,
                    "passing_yards": 20.0 if index < dropbacks else 0.0,
                    "pass_touchdown": 0.0,
                    "interception": 0.0,
                    "rusher_player_id": None,
                    "rushing_yards": 0.0,
                    "rush_touchdown": 0.0,
                }
            )
    return games, pl.DataFrame(plays)


def test_matchup_history_is_shifted_and_game_specific() -> None:
    games, plays = _frame()
    result = add_matchup_history_features(games, plays).sort("week")

    assert result[0, "team_trailing_play_rate_avg_last_3"] is None
    assert result[1, "team_trailing_play_rate_avg_last_3"] == pytest.approx(0.4)
    assert result[2, "team_trailing_play_rate_avg_last_3"] == pytest.approx(0.6)
    assert result[1, "team_td_drive_rate_avg_last_3"] == pytest.approx(0.0)
    assert result[2, "team_td_drive_rate_avg_last_3"] == pytest.approx(0.125)
    assert result[2, "team_dropbacks_avg_last_3"] == pytest.approx(6.0)
    assert result[2, "opponent_td_drive_rate_allowed_avg_last_3"] == pytest.approx(0.25)


def test_qb_efficiency_and_dropbacks_are_rolling_prior_game_features() -> None:
    games, plays = _frame()
    qb_games = games.with_columns(
        pl.lit("AAA_qb").alias("player_id"),
        pl.lit("QB").alias("position"),
    )
    result = add_matchup_history_features(qb_games, plays).sort("week")

    assert result[0, "qb_dropbacks_avg_last_3"] is None
    assert result[1, "qb_dropbacks_avg_last_3"] == pytest.approx(5.0)
    assert result[2, "qb_dropbacks_avg_last_3"] == pytest.approx(6.0)
    assert result[2, "qb_fantasy_points_per_dropback_avg_last_3"] == pytest.approx(0.8)


def test_future_play_changes_do_not_change_earlier_matchup_features() -> None:
    games, plays = _frame()
    original = add_matchup_history_features(games, plays).sort("week")
    altered_plays = plays.with_columns(
        pl.when(pl.col("game_id") == "g3")
        .then(pl.lit("Touchdown"))
        .otherwise(pl.col("series_result"))
        .alias("series_result"),
        pl.when(pl.col("game_id") == "g3")
        .then(pl.lit(-10.0))
        .otherwise(pl.col("score_differential"))
        .alias("score_differential"),
    )
    altered = add_matchup_history_features(games, altered_plays).sort("week")

    for column in (
        "team_trailing_play_rate_avg_last_5",
        "team_td_drive_rate_avg_last_5",
        "team_dropbacks_avg_last_5",
        "opponent_td_drive_rate_allowed_avg_last_5",
    ):
        assert altered[1, column] == original[1, column]
        assert altered[2, column] == original[2, column]
