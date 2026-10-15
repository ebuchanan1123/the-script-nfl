from datetime import date, timedelta

import polars as pl

from the_script.features.advanced import ADVANCED_FEATURE_COLUMNS
from the_script.features.coaching import QB_PLAYCALL_FEATURE_COLUMNS
from the_script.features.matchup import MATCHUP_FEATURE_COLUMNS
from the_script.models.multiposition import (
    POSITION_CONFIGS,
    add_position_form_features,
    position_feature_columns,
    position_ranking_candidates,
)


def _qb_games() -> pl.DataFrame:
    start = date(2024, 9, 1)
    return pl.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "game_date": [start + timedelta(days=7 * index) for index in range(3)],
            "season": [2024] * 3,
            "week": [1, 2, 3],
            "player_id": ["qb-1"] * 3,
            "team": ["BUF"] * 3,
            "opponent": ["NYJ", "MIA", "JAX"],
            "home_game": [True, False, True],
            "days_rest": [7, 7, 7],
            "position": ["QB"] * 3,
            "attempts": [30, 40, 999],
            "passing_yards": [200, 300, 999],
            "passing_tds": [1, 2, 99],
            "passing_interceptions": [0, 1, 99],
            "carries": [5, 6, 99],
            "rushing_yards": [30, 40, 999],
            "rushing_tds": [0, 1, 99],
            "snap_share": [0.95, 1.0, 1.0],
        }
    )


def test_qb_form_features_exclude_current_game_outcomes() -> None:
    featured = add_position_form_features(_qb_games(), "QB").sort("week")

    assert featured[2, "passing_yards_avg_last_3"] == 250
    assert featured[2, "rushing_yards_avg_last_16"] == 35
    assert featured[2, "attempts_avg_last_5"] == 35
    assert featured[2, "games_played_before_game"] == 2


def test_required_multi_position_outcomes_and_features_are_configured() -> None:
    assert {"attempts", "passing_yards", "rushing_yards", "rushing_tds"}.issubset(
        POSITION_CONFIGS["QB"].outcomes
    )
    assert {"rushing_yards", "receiving_yards", "receptions"}.issubset(
        POSITION_CONFIGS["RB"].outcomes
    )
    assert {"receiving_yards", "receptions", "receiving_tds"}.issubset(
        POSITION_CONFIGS["TE"].outcomes
    )
    for position in POSITION_CONFIGS:
        features = position_feature_columns(position)
        assert "snap_share_avg_last_3" in features
        assert len(features) == len(set(features))
        assert set(ADVANCED_FEATURE_COLUMNS).issubset(features)
        assert set(MATCHUP_FEATURE_COLUMNS).issubset(features)
    assert {
        "team_trailing_play_rate_avg_last_3",
        "team_td_drive_rate_avg_last_3",
        "team_dropbacks_avg_last_3",
        "opponent_td_drive_rate_allowed_avg_last_3",
    }.issubset(position_feature_columns("RB"))
    assert {
        "qb_dropbacks_avg_last_3",
        "qb_fantasy_points_per_dropback_avg_last_3",
    }.issubset(position_feature_columns("QB"))
    assert set(QB_PLAYCALL_FEATURE_COLUMNS).issubset(position_feature_columns("QB"))


def test_depth_chart_candidates_include_returning_player_without_stats() -> None:
    stats = pl.DataFrame(
        {
            "season": [2026],
            "season_type": ["REG"],
            "week": [1],
            "player_id": ["active-te-1"],
            "player_display_name": ["Active Tight End"],
            "position": ["TE"],
            "team": ["LV"],
        }
    )
    depth_charts = pl.DataFrame(
        {
            "dt": ["2026-09-23T12:43:59Z"],
            "team": ["LV"],
            "gsis_id": ["returning-te-1"],
            "pos_abb": ["TE"],
            "pos_rank": [1],
            "player_name": ["Returning Tight End"],
        }
    )
    schedules = pl.DataFrame(
        {
            "season": [2026],
            "week": [3],
            "game_type": ["REG"],
            "gameday": ["2026-09-24"],
        }
    )

    candidates = position_ranking_candidates(
        stats, depth_charts, schedules, "TE", 2026, 3
    )

    assert candidates.filter(pl.col("player_id") == "returning-te-1").height == 1