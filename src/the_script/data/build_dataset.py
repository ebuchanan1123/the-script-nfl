"""Build and validate the canonical regular-season WR-game table."""

from pathlib import Path

import polars as pl

from the_script.data.clean import clean_player_stats, clean_schedules, clean_snap_counts

CANONICAL_COLUMNS = [
    "game_id",
    "game_date",
    "season",
    "week",
    "player_id",
    "team",
    "opponent",
    "home_game",
    "days_rest",
    "position",
    "receiving_yards",
    "targets",
    "receptions",
    "receiving_touchdowns",
    "target_share",
    "air_yards_share",
    "wopr",
    "snap_share",
]


def validate_wr_player_games(frame: pl.DataFrame) -> None:
    """Fail fast when the canonical observation unit or join keys are invalid."""
    identifier_columns = ["game_id", "game_date", "player_id", "team", "opponent"]
    null_counts = frame.select(pl.col(identifier_columns).null_count()).row(0, named=True)
    invalid_nulls = {name: count for name, count in null_counts.items() if count}
    if invalid_nulls:
        raise ValueError(f"Null canonical identifiers: {invalid_nulls}")
    if frame.select(pl.struct(["game_id", "player_id"]).n_unique()).item() != frame.height:
        raise ValueError("Duplicate (game_id, player_id) rows in canonical table")
    if not frame["season"].is_between(2016, 2025).all():
        raise ValueError("Canonical table contains a season outside 2016-2025")
    if frame.filter(pl.col("team") == pl.col("opponent")).height:
        raise ValueError("Canonical table contains a team playing itself")


def build_wr_player_games(
    player_stats: pl.DataFrame,
    schedules: pl.DataFrame,
    snap_counts: pl.DataFrame,
    player_registry: pl.DataFrame,
) -> pl.DataFrame:
    """Create one row per snap-verified regular-season WR-game with schedule context."""
    player_rows = clean_player_stats(player_stats)
    participation = clean_snap_counts(snap_counts, players=player_registry)
    games = clean_schedules(schedules).select(
        "game_id",
        "game_date",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_rest",
        "away_rest",
    )
    joined = (
        player_rows.join(games, on=["game_id", "season", "week"], how="left", validate="m:1")
        .join(participation, on=["game_id", "player_id"], how="inner", validate="1:1")
    )
    is_home = pl.col("team") == pl.col("home_team")
    is_away = pl.col("team") == pl.col("away_team")
    unmatched = joined.filter(~(is_home | is_away) | pl.col("game_date").is_null())
    if unmatched.height:
        examples = unmatched.select("game_id", "player_id", "team").head(5).to_dicts()
        raise ValueError(f"WR rows did not match a scheduled team: {examples}")

    derived_opponent = pl.when(is_home).then(pl.col("away_team")).otherwise(pl.col("home_team"))
    opponent_mismatch = joined.filter(pl.col("source_opponent") != derived_opponent)
    if opponent_mismatch.height:
        examples = opponent_mismatch.select(
            "game_id", "player_id", "source_opponent", "home_team", "away_team"
        ).head(5).to_dicts()
        raise ValueError(f"Source and schedule opponents disagree: {examples}")

    result = (
        joined.with_columns(
            is_home.alias("home_game"),
            derived_opponent.alias("opponent"),
            pl.when(is_home).then(pl.col("home_rest")).otherwise(pl.col("away_rest")).alias("days_rest"),
            pl.col("receiving_tds").alias("receiving_touchdowns"),
        )
        .select(CANONICAL_COLUMNS)
        .sort(["game_date", "game_id", "player_id"])
    )
    validate_wr_player_games(result)
    return result


def build_wr_player_games_from_parquet(
    raw_directory: Path, output_path: Path
) -> pl.DataFrame:
    frame = build_wr_player_games(
        pl.read_parquet(raw_directory / "player_stats.parquet"),
        pl.read_parquet(raw_directory / "schedules.parquet"),
        pl.read_parquet(raw_directory / "snap_counts.parquet"),
        pl.read_parquet(raw_directory / "players.parquet"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path)
    return frame
