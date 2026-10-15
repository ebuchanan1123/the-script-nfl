import polars as pl
import pytest

from the_script.features.coaching import add_qb_playcall_features


def test_playcall_features_use_only_prior_games_and_current_coach():
    games = pl.DataFrame({
        "game_id": ["g1", "g2", "g3", "g4"], "season": [2025] * 4,
        "week": [1, 2, 3, 4], "team": ["BUF"] * 4,
        "opponent": ["NYJ"] * 4, "team_implied_margin": [-4.0] * 4,
    })
    schedules = games.select("game_id").with_columns(
        pl.lit("BUF").alias("home_team"), pl.lit("NYJ").alias("away_team"),
        pl.Series("home_coach", ["A", "A", "B", "B"]), pl.lit("C").alias("away_coach"),
    )
    pbp = pl.DataFrame({
        "game_id": ["g1", "g1", "g2", "g2", "g3", "g3"],
        "season": [2025] * 6, "week": [1, 1, 2, 2, 3, 3],
        "posteam": ["BUF"] * 6, "qb_dropback": [1, 0, 1, 1, 0, 0],
        "rush_attempt": [0, 1, 0, 0, 1, 1], "score_differential": [-3, 3, -3, 3, -3, 3],
        "game_seconds_remaining": [1800, 1700] * 3, "play_id": [1, 2] * 3,
    })
    result = add_qb_playcall_features(games, pbp, schedules).sort("week")
    assert result[0, "playcall_neutral_pass_rate_avg_last_5"] is None
    assert result[1, "playcall_neutral_pass_rate_avg_last_5"] == 0.5
    assert result[2, "playcall_neutral_pass_rate_avg_last_5"] == 0.75
    assert result[2, "coach_neutral_pass_rate_avg_last_16"] is None
    assert result[3, "coach_neutral_pass_rate_avg_last_16"] == 0
    assert result[1, "underdog_neutral_pass_interaction"] == 2
    changed = pbp.with_columns(pl.when(pl.col("week") == 3).then(1).otherwise(pl.col("qb_dropback")).alias("qb_dropback"))
    after = add_qb_playcall_features(games, changed, schedules).sort("week")
    assert after[2, "playcall_neutral_pass_rate_avg_last_5"] == result[2, "playcall_neutral_pass_rate_avg_last_5"]
    assert result[3, "playcall_plays_avg_last_5"] == pytest.approx(2)