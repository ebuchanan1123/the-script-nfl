"""Canonical NFL identifiers and source-level cleaning rules."""

import polars as pl

TEAM_ALIASES = {
    "JAC": "JAX",
    "LA": "LAR",
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LAR",
}


def canonical_team_code(team: str) -> str:
    """Normalize one team code using the canonical historical alias contract."""
    return TEAM_ALIASES.get(team, team)


def canonical_team(column: str) -> pl.Expr:
    """Normalize historical aliases without rewriting the raw source files."""
    return pl.col(column).replace_strict(TEAM_ALIASES, default=pl.col(column))


def clean_player_stats(frame: pl.DataFrame) -> pl.DataFrame:
    required = {
        "game_id",
        "player_id",
        "position",
        "season",
        "week",
        "season_type",
        "team",
        "opponent_team",
        "receiving_yards",
        "targets",
        "receptions",
        "receiving_tds",
        "target_share",
        "air_yards_share",
        "wopr",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"player_stats is missing columns: {', '.join(missing)}")
    return (
        frame.filter(
            (pl.col("season_type") == "REG")
            & (pl.col("position") == "WR")
            & pl.col("season").is_between(2016, 2025)
        )
        .with_columns(
            canonical_team("team").alias("team"),
            canonical_team("opponent_team").alias("source_opponent"),
        )
        .unique(["game_id", "player_id"], keep="none")
    )


def clean_schedules(
    frame: pl.DataFrame, min_season: int = 2016, max_season: int = 2025
) -> pl.DataFrame:
    required = {
        "game_id",
        "season",
        "week",
        "game_type",
        "gameday",
        "home_team",
        "away_team",
        "home_rest",
        "away_rest",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"schedules is missing columns: {', '.join(missing)}")
    game_date = (
        pl.col("gameday")
        if frame.schema["gameday"] == pl.Date
        else pl.col("gameday").str.to_date(strict=False)
    )
    return (
        frame.filter(
            (pl.col("game_type") == "REG")
            & pl.col("season").is_between(min_season, max_season)
        )
        .with_columns(
            canonical_team("home_team").alias("home_team"),
            canonical_team("away_team").alias("away_team"),
            game_date.alias("game_date"),
        )
        .unique("game_id", keep="none")
    )


def clean_snap_counts(snap_counts: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    """Map PFR snap IDs to GSIS IDs and retain verified offensive participation."""
    snap_required = {"game_id", "pfr_player_id", "offense_snaps", "offense_pct"}
    player_required = {"pfr_id", "gsis_id"}
    missing_snaps = sorted(snap_required.difference(snap_counts.columns))
    missing_players = sorted(player_required.difference(players.columns))
    if missing_snaps:
        raise ValueError(f"snap_counts is missing columns: {', '.join(missing_snaps)}")
    if missing_players:
        raise ValueError(f"players is missing columns: {', '.join(missing_players)}")

    id_map = (
        players.select("pfr_id", "gsis_id")
        .drop_nulls()
        .unique("pfr_id", keep="none")
    )
    return (
        snap_counts.filter(pl.col("offense_snaps") > 0)
        .join(id_map, left_on="pfr_player_id", right_on="pfr_id", how="inner", validate="m:1")
        .select("game_id", pl.col("gsis_id").alias("player_id"), pl.col("offense_pct").alias("snap_share"))
        .unique(["game_id", "player_id"], keep="none")
    )
