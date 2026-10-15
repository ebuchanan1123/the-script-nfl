from datetime import date, timedelta

import polars as pl
import pytest

from the_script.features.tendencies import add_team_tendency_features


def _games() -> pl.DataFrame:
    start = date(2025, 9, 1)
    return pl.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "game_date": [start + timedelta(days=7 * index) for index in range(3)],
            "season": [2025, 2025, 2025],
            "week": [1, 2, 3],
            "team": ["PHI", "PHI", "PHI"],
            "player_id": ["wr1", "wr1", "wr1"],
        }
    )


def _stats() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["g1", "g1", "g1", "g2", "g2", "g2"],
            "season_type": ["REG"] * 6,
            "team": ["PHI"] * 6,
            "position": ["WR", "WR", "TE", "WR", "WR", "TE"],
            "targets": [6, 2, 2, 4, 4, 2],
        }
    )


def _plays() -> pl.DataFrame:
    rows = []
    for game_id, dropbacks in (("g1", 6), ("g2", 4)):
        for index in range(10):
            rows.append(
                {
                    "game_id": game_id,
                    "posteam": "PHI",
                    "qb_dropback": float(index < dropbacks),
                    "rush_attempt": float(index >= dropbacks),
                    "qb_kneel": 0.0,
                    "qb_spike": 0.0,
                    "score_differential": 0.0,
                    "wp": 0.5,
                }
            )
    return pl.DataFrame(rows)


def test_tendencies_use_only_prior_team_games() -> None:
    result = add_team_tendency_features(_games(), _stats(), _plays()).sort("week")

    assert result[0, "offense_pass_rate_avg_last_3"] is None
    assert result[1, "offense_pass_rate_avg_last_3"] == pytest.approx(0.6)
    assert result[2, "offense_pass_rate_avg_last_3"] == pytest.approx(0.5)
    assert result[2, "neutral_pass_rate_avg_last_5"] == pytest.approx(0.5)
    assert result[1, "team_wr_target_share_avg_last_3"] == pytest.approx(0.8)
    assert result[2, "team_wr_target_share_avg_last_5"] == pytest.approx(0.8)
    assert result[2, "team_wr_target_concentration_avg_last_5"] == pytest.approx(0.5625)


def test_future_game_has_tendencies_without_current_outcomes() -> None:
    result = add_team_tendency_features(_games(), _stats(), _plays()).sort("week")

    assert result[2, "offense_pass_rate_season_to_date"] == pytest.approx(0.5)
    assert result[2, "team_wr_target_share_season_to_date"] == pytest.approx(0.8)