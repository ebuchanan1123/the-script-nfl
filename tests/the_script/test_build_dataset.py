import polars as pl
import pytest

from the_script.data.build_dataset import build_wr_player_games, validate_wr_player_games


def _schedules() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["2024_01_JAX_MIA", "2024_02_JAX_BUF"],
            "season": [2024, 2024],
            "week": [1, 2],
            "game_type": ["REG", "REG"],
            "gameday": ["2024-09-08", "2024-09-15"],
            "home_team": ["MIA", "BUF"],
            "away_team": ["JAX", "JAX"],
            "home_rest": [7, 7],
            "away_rest": [7, 7],
        }
    )


def _player_stats() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["2024_01_JAX_MIA", "2024_02_JAX_BUF", "2024_02_JAX_BUF"],
            "player_id": ["wr-1", "wr-1", "qb-1"],
            "position": ["WR", "WR", "QB"],
            "season": [2024, 2024, 2024],
            "week": [1, 2, 2],
            "season_type": ["REG", "REG", "REG"],
            "team": ["JAC", "JAC", "JAC"],
            "opponent_team": ["MIA", "BUF", "BUF"],
            "receiving_yards": [50, 70, 0],
            "targets": [6, 8, 0],
            "receptions": [4, 5, 0],
            "receiving_tds": [0, 1, 0],
            "target_share": [0.2, 0.25, 0.0],
            "air_yards_share": [0.3, 0.35, 0.0],
            "wopr": [0.4, 0.45, 0.0],
        }
    )


def _players() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "pfr_id": ["pfr-wr-1", "pfr-qb-1"],
            "gsis_id": ["wr-1", "qb-1"],
        }
    )


def _snap_counts() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["2024_01_JAX_MIA", "2024_02_JAX_BUF", "2024_02_JAX_BUF"],
            "pfr_player_id": ["pfr-wr-1", "pfr-wr-1", "pfr-qb-1"],
            "offense_snaps": [45, 50, 60],
            "offense_pct": [0.75, 0.80, 1.0],
        }
    )


def _build(player_stats: pl.DataFrame | None = None) -> pl.DataFrame:
    return build_wr_player_games(
        _player_stats() if player_stats is None else player_stats,
        _schedules(),
        _snap_counts(),
        _players(),
    )


def test_builds_one_canonical_regular_season_wr_row_per_game() -> None:
    result = _build()

    assert result.height == 2
    assert result["team"].to_list() == ["JAX", "JAX"]
    assert result["opponent"].to_list() == ["MIA", "BUF"]
    assert result["home_game"].to_list() == [False, False]
    assert result["days_rest"].to_list() == [7, 7]
    assert result["receiving_yards"].to_list() == [50, 70]
    assert result["target_share"].to_list() == [0.2, 0.25]
    assert result["snap_share"].to_list() == [0.75, 0.80]


def test_future_source_rows_do_not_change_historical_base_rows() -> None:
    complete = _build()
    truncated = build_wr_player_games(
        _player_stats().head(1), _schedules().head(1), _snap_counts().head(1), _players()
    )

    assert complete.head(1).equals(truncated)


def test_duplicate_player_game_is_rejected() -> None:
    valid = _build()
    duplicate = pl.concat([valid, valid.head(1)])

    with pytest.raises(ValueError, match="Duplicate"):
        validate_wr_player_games(duplicate)


def test_unmatched_schedule_team_is_rejected() -> None:
    invalid = _player_stats().with_columns(pl.lit("KC").alias("team"))

    with pytest.raises(ValueError, match="did not match"):
        _build(invalid)


def test_source_and_schedule_opponents_must_agree() -> None:
    invalid = _player_stats().with_columns(pl.lit("KC").alias("opponent_team"))

    with pytest.raises(ValueError, match="opponents disagree"):
        _build(invalid)


def test_wr_without_positive_offensive_snaps_is_excluded() -> None:
    snaps = _snap_counts().with_columns(pl.lit(0).alias("offense_snaps"))

    result = build_wr_player_games(_player_stats(), _schedules(), snaps, _players())

    assert result.is_empty()