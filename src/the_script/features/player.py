"""Leakage-safe, prior-game WR form features."""

from pathlib import Path

import polars as pl

from the_script.features.advanced import add_advanced_features
from the_script.features.injuries import add_injury_features
from the_script.features.opponent import add_opponent_wr_features
from the_script.features.role import add_role_features
from the_script.features.tendencies import add_team_tendency_features

PLAYER_FEATURE_COLUMNS = [
	"yards_last_game",
	"yards_avg_last_3",
	"yards_avg_last_5",
	"yards_avg_last_16",
	"targets_avg_last_3",
	"targets_avg_last_5",
	"targets_avg_last_16",
	"target_share_last_game",
	"target_share_avg_last_3",
	"target_share_avg_last_5",
	"target_share_avg_last_16",
	"air_yards_share_last_game",
	"air_yards_share_avg_last_3",
	"air_yards_share_avg_last_5",
	"wopr_last_game",
	"wopr_avg_last_3",
	"wopr_avg_last_5",
	"receptions_avg_last_5",
	"receptions_avg_last_16",
	"snap_share_last_game",
	"snap_share_avg_last_3",
	"season_yards_per_game_before_game",
	"games_played_before_game",
]


def _prior_rolling_mean(column: str, window_size: int, group: str | list[str]) -> pl.Expr:
	return (
		pl.col(column)
		.shift(1)
		.rolling_mean(window_size=window_size, min_samples=1)
		.over(group)
	)


def add_player_form_features(player_games: pl.DataFrame) -> pl.DataFrame:
	"""Add shifted player history; no current-game outcome enters a feature."""
	required = {
		"game_id",
		"game_date",
		"season",
		"player_id",
		"receiving_yards",
		"targets",
		"receptions",
		"target_share",
		"air_yards_share",
		"wopr",
		"snap_share",
	}
	missing = sorted(required.difference(player_games.columns))
	if missing:
		raise ValueError(f"player_games is missing columns: {', '.join(missing)}")

	ordered = player_games.sort(["player_id", "game_date", "game_id"])
	season_group = ["player_id", "season"]
	return ordered.with_columns(
		pl.col("receiving_yards").shift(1).over("player_id").alias("yards_last_game"),
		_prior_rolling_mean("receiving_yards", 3, "player_id").alias("yards_avg_last_3"),
		_prior_rolling_mean("receiving_yards", 5, "player_id").alias("yards_avg_last_5"),
		_prior_rolling_mean("receiving_yards", 16, "player_id").alias("yards_avg_last_16"),
		_prior_rolling_mean("targets", 3, "player_id").alias("targets_avg_last_3"),
		_prior_rolling_mean("targets", 5, "player_id").alias("targets_avg_last_5"),
		_prior_rolling_mean("targets", 16, "player_id").alias("targets_avg_last_16"),
		pl.col("target_share").shift(1).over("player_id").alias("target_share_last_game"),
		_prior_rolling_mean("target_share", 3, "player_id").alias("target_share_avg_last_3"),
		_prior_rolling_mean("target_share", 5, "player_id").alias("target_share_avg_last_5"),
		_prior_rolling_mean("target_share", 16, "player_id").alias("target_share_avg_last_16"),
		pl.col("air_yards_share").shift(1).over("player_id").alias("air_yards_share_last_game"),
		_prior_rolling_mean("air_yards_share", 3, "player_id").alias("air_yards_share_avg_last_3"),
		_prior_rolling_mean("air_yards_share", 5, "player_id").alias("air_yards_share_avg_last_5"),
		pl.col("wopr").shift(1).over("player_id").alias("wopr_last_game"),
		_prior_rolling_mean("wopr", 3, "player_id").alias("wopr_avg_last_3"),
		_prior_rolling_mean("wopr", 5, "player_id").alias("wopr_avg_last_5"),
		_prior_rolling_mean("receptions", 5, "player_id").alias("receptions_avg_last_5"),
		_prior_rolling_mean("receptions", 16, "player_id").alias("receptions_avg_last_16"),
		pl.col("snap_share").shift(1).over("player_id").alias("snap_share_last_game"),
		_prior_rolling_mean("snap_share", 3, "player_id").alias("snap_share_avg_last_3"),
		_prior_rolling_mean("receiving_yards", 100, season_group).alias(
			"season_yards_per_game_before_game"
		),
		pl.int_range(0, pl.len()).over(season_group).alias("games_played_before_game"),
	)


def build_player_features_from_parquet(
	input_path: Path,
	output_path: Path,
	player_stats_path: Path = Path("data/raw/player_stats.parquet"),
	depth_charts_path: Path = Path("data/raw/depth_charts.parquet"),
	play_by_play_path: Path = Path("data/raw/play_by_play.parquet"),
	schedules_path: Path = Path("data/raw/schedules.parquet"),
	injuries_path: Path = Path("data/raw/injuries.parquet"),
) -> pl.DataFrame:
	"""Build and persist shifted player features separately from the canonical table."""
	player_games = pl.read_parquet(input_path)
	with_form = add_player_form_features(player_games)
	with_role = add_role_features(
		with_form,
		pl.read_parquet(player_stats_path),
		pl.read_parquet(depth_charts_path),
	)
	with_tendencies = add_team_tendency_features(
		with_role,
		pl.read_parquet(player_stats_path),
		pl.read_parquet(play_by_play_path),
	)
	with_injuries = add_injury_features(
		with_tendencies,
		pl.read_parquet(player_stats_path),
		pl.read_parquet(injuries_path),
	)
	with_advanced = add_advanced_features(
		with_injuries,
		pl.read_parquet(schedules_path),
		pl.read_parquet(play_by_play_path),
	)
	featured = add_opponent_wr_features(with_advanced)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	featured.write_parquet(output_path)
	return featured
