from datetime import date

import polars as pl
import pytest

from the_script.features.advanced import add_qb_context_features
from the_script.features.role import add_role_features


def _games() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "game_date": [date(2025, 9, 7), date(2025, 9, 14)],
            "season": [2025, 2025],
            "week": [1, 2],
            "player_id": ["smith", "smith"],
            "team": ["PHI", "PHI"],
        }
    )


def _prior_stats() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "season_type": ["REG", "REG", "REG"],
            "team": ["PHI", "PHI", "PHI"],
            "player_id": ["smith", "brown", "goedert"],
            "targets": [25, 30, 45],
        }
    )


def _legacy_depth_charts() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 1, 2, 2],
            "club_code": ["PHI", "PHI", "PHI", "PHI"],
            "gsis_id": ["smith", "goedert", "smith", "goedert"],
            "position": ["WR", "TE", "WR", "TE"],
            "depth_team": ["2", "1", "1", "1"],
            "game_type": ["REG", "REG", "REG", "REG"],
        }
    )


def test_role_features_capture_promotion_and_vacated_opportunity() -> None:
    result = add_role_features(_games(), _prior_stats(), _legacy_depth_charts())

    assert result[0, "depth_chart_rank"] == 2
    assert result[1, "depth_chart_rank"] == 1
    assert result[1, "depth_chart_rank_change"] == 1
    assert result[0, "depth_chart_available"] == 1
    assert result[0, "returning_team_target_share"] == pytest.approx(0.7)
    assert result[0, "vacated_team_target_share"] == pytest.approx(0.3)
    assert result[0, "active_teammate_target_share"] == pytest.approx(0.45)
    assert result[0, "team_role_available"] == 1


def test_missing_depth_chart_is_explicit_and_does_not_invent_role() -> None:
    result = add_role_features(_games(), _prior_stats(), pl.DataFrame())

    assert result["depth_chart_rank"].null_count() == result.height
    assert result["vacated_team_target_share"].null_count() == result.height
    assert result["depth_chart_available"].to_list() == [0, 0]
    assert result["team_role_available"].to_list() == [0, 0]


def test_timestamped_depth_chart_uses_latest_snapshot_before_game() -> None:
    games = _games().head(1)
    depth_charts = pl.DataFrame(
        {
            "dt": ["2025-09-01T12:00:00Z", "2025-09-06T12:00:00Z"],
            "team": ["PHI", "PHI"],
            "gsis_id": ["smith", "smith"],
            "pos_abb": ["WR", "WR"],
            "pos_rank": [2, 1],
        }
    )

    result = add_role_features(games, _prior_stats(), depth_charts)

    assert result[0, "depth_chart_rank"] == 1
    assert result[0, "depth_chart_available"] == 1


def test_prior_team_data_is_required_for_vacated_share() -> None:
    result = add_role_features(_games(), pl.DataFrame(), _legacy_depth_charts())

    assert result["vacated_team_target_share"].null_count() == result.height
    assert result["team_role_available"].to_list() == [0, 0]


def test_role_rank_baseline_uses_only_observations_before_game_date() -> None:
    games = pl.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "game_date": [date(2025, 9, 7), date(2025, 9, 14), date(2025, 9, 21)],
            "season": [2025, 2025, 2025],
            "week": [1, 2, 3],
            "player_id": ["smith", "smith", "smith"],
            "team": ["PHI", "PHI", "PHI"],
            "target_share": [0.2, 0.3, 0.4],
        }
    )
    depth_charts = pl.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "week": [1, 2, 3],
            "club_code": ["PHI", "PHI", "PHI"],
            "gsis_id": ["smith", "smith", "smith"],
            "position": ["WR", "WR", "WR"],
            "depth_team": ["1", "1", "1"],
            "game_type": ["REG", "REG", "REG"],
        }
    )

    baseline = add_role_features(games, _prior_stats(), depth_charts)
    changed_future = add_role_features(
        games.with_columns(
            pl.when(pl.col("week") == 3)
            .then(0.99)
            .otherwise(pl.col("target_share"))
            .alias("target_share")
        ),
        _prior_stats(),
        depth_charts,
    )

    assert baseline[1, "role_rank_target_share_delta"] == pytest.approx(0.0)
    assert changed_future[1, "role_rank_target_share_delta"] == pytest.approx(
        baseline[1, "role_rank_target_share_delta"]
    )


def test_add_qb_context_features_handles_multi_qb_receiver_targets() -> None:
    player_games = pl.DataFrame(
        {
            "game_id": ["g1", "g1", "g1"],
            "game_date": [date(2025, 9, 7), date(2025, 9, 7), date(2025, 9, 7)],
            "season": [2025, 2025, 2025],
            "week": [1, 1, 1],
            "team": ["PHI", "PHI", "PHI"],
            "player_id": ["receiver", "receiver", "rb"],
        }
    )
    play_by_play = pl.DataFrame(
        {
            "game_id": ["g1", "g1", "g1", "g1"],
            "posteam": ["PHI", "PHI", "PHI", "PHI"],
            "passer_player_id": ["qb_a", "qb_b", "qb_a", "qb_b"],
            "receiver_player_id": ["receiver", "receiver", "receiver", "receiver"],
            "qb_dropback": [1, 1, 1, 1],
            "qb_scramble": [0, 0, 0, 0],
            "qb_hit": [0, 0, 0, 0],
            "sack": [0, 0, 0, 0],
            "qb_epa": [0.1, -0.2, -0.1, 0.3],
            "cpoe": [0.5, 0.2, 0.7, -0.1],
            "air_yards": [5, 8, 7, 9],
            "complete_pass": [1, 0, 1, 0],
            "yards_gained": [10, 5, 11, 8],
            "pass_attempt": [1, 1, 1, 1],
        }
    )

    result = add_qb_context_features(player_games, play_by_play)

    assert result.height == 3
    assert "qb_receiver_target_share_avg_last_5" in result.columns