"""Leakage-safe WR production allowed by the upcoming opponent."""

import polars as pl

OPPONENT_FEATURE_COLUMNS = [
	"opponent_wr_yards_allowed_last_game",
	"opponent_wr_yards_allowed_avg_last_3",
	"opponent_wr_yards_allowed_avg_last_5",
	"opponent_wr_yards_allowed_season_to_date",
	"opponent_wr_ppr_points_allowed_last_game",
	"opponent_wr_ppr_points_allowed_avg_last_3",
	"opponent_wr_ppr_points_allowed_avg_last_5",
	"opponent_wr_ppr_points_allowed_season_to_date",
]


def _prior_average(column: str, window: int, group: str | list[str]) -> pl.Expr:
	return (
		pl.col(column)
		.shift(1)
		.rolling_mean(window_size=window, min_samples=1)
		.over(group)
	)


def build_opponent_wr_history(player_games: pl.DataFrame) -> pl.DataFrame:
	"""Aggregate one row per defense-game, then shift before rolling."""
	required = {
		"game_id",
		"game_date",
		"season",
		"week",
		"opponent",
		"receiving_yards",
		"receptions",
		"receiving_touchdowns",
	}
	missing = sorted(required.difference(player_games.columns))
	if missing:
		raise ValueError(f"player_games is missing opponent inputs: {', '.join(missing)}")

	game_totals = (
		player_games.with_columns(
			(
				pl.col("receptions")
				+ pl.col("receiving_yards") / 10
				+ pl.col("receiving_touchdowns") * 6
			).alias("wr_ppr_points_allowed")
		)
		.group_by("opponent", "game_id", "game_date", "season", "week")
		.agg(
			pl.col("receiving_yards").sum().alias("wr_yards_allowed"),
			pl.col("wr_ppr_points_allowed").sum().alias("wr_ppr_points_allowed"),
		)
		.sort(["opponent", "game_date", "game_id"])
	)
	season_group = ["opponent", "season"]
	return game_totals.with_columns(
		pl.col("wr_yards_allowed")
		.shift(1)
		.over("opponent")
		.alias("opponent_wr_yards_allowed_last_game"),
		_prior_average("wr_yards_allowed", 3, "opponent").alias(
			"opponent_wr_yards_allowed_avg_last_3"
		),
		_prior_average("wr_yards_allowed", 5, "opponent").alias(
			"opponent_wr_yards_allowed_avg_last_5"
		),
		_prior_average("wr_yards_allowed", 100, season_group).alias(
			"opponent_wr_yards_allowed_season_to_date"
		),
		pl.col("wr_ppr_points_allowed")
		.shift(1)
		.over("opponent")
		.alias("opponent_wr_ppr_points_allowed_last_game"),
		_prior_average("wr_ppr_points_allowed", 3, "opponent").alias(
			"opponent_wr_ppr_points_allowed_avg_last_3"
		),
		_prior_average("wr_ppr_points_allowed", 5, "opponent").alias(
			"opponent_wr_ppr_points_allowed_avg_last_5"
		),
		_prior_average("wr_ppr_points_allowed", 100, season_group).alias(
			"opponent_wr_ppr_points_allowed_season_to_date"
		),
	).select("game_id", "opponent", *OPPONENT_FEATURE_COLUMNS)


def add_opponent_wr_features(player_games: pl.DataFrame) -> pl.DataFrame:
	"""Join pregame opponent WR-allowed history onto every player-game row."""
	history = build_opponent_wr_history(player_games)
	return player_games.join(history, on=["game_id", "opponent"], how="left", validate="m:1")
