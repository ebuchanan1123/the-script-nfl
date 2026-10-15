from datetime import date

import polars as pl

from the_script.features.opponent import OPPONENT_FEATURE_COLUMNS, add_opponent_wr_features


def _games() -> pl.DataFrame:
	return pl.DataFrame(
		{
			"game_id": ["g1", "g1", "g2", "g2", "g3"],
			"game_date": [
				date(2024, 9, 1),
				date(2024, 9, 1),
				date(2024, 9, 8),
				date(2024, 9, 8),
				date(2024, 9, 15),
			],
			"season": [2024] * 5,
			"week": [1, 1, 2, 2, 3],
			"player_id": ["a", "b", "c", "d", "e"],
			"opponent": ["DEF", "DEF", "DEF", "DEF", "DEF"],
			"receiving_yards": [40, 60, 30, 20, 999],
			"receptions": [4, 6, 3, 2, 20],
			"receiving_touchdowns": [0, 1, 0, 0, 5],
		}
	)


def test_opponent_features_aggregate_all_wrs_then_shift_games() -> None:
	featured = add_opponent_wr_features(_games())
	week_three = featured.filter(pl.col("game_id") == "g3")

	assert week_three[0, "opponent_wr_yards_allowed_last_game"] == 50
	assert week_three[0, "opponent_wr_yards_allowed_avg_last_3"] == 75
	assert week_three[0, "opponent_wr_yards_allowed_avg_last_5"] == 75
	assert week_three[0, "opponent_wr_yards_allowed_season_to_date"] == 75
	assert week_three[0, "opponent_wr_ppr_points_allowed_last_game"] == 10
	assert week_three[0, "opponent_wr_ppr_points_allowed_avg_last_3"] == 18


def test_current_game_allowed_production_cannot_change_current_features() -> None:
	original = _games()
	mutated = original.with_columns(
		pl.when(pl.col("game_id") == "g3")
		.then(pl.lit(9999))
		.otherwise(pl.col("receiving_yards"))
		.alias("receiving_yards")
	)

	original_row = add_opponent_wr_features(original).filter(pl.col("game_id") == "g3")
	mutated_row = add_opponent_wr_features(mutated).filter(pl.col("game_id") == "g3")

	assert original_row.select(OPPONENT_FEATURE_COLUMNS).equals(
		mutated_row.select(OPPONENT_FEATURE_COLUMNS)
	)


def test_first_opponent_game_has_no_prior_allowed_history() -> None:
	first_game = add_opponent_wr_features(_games()).filter(pl.col("game_id") == "g1")

	assert first_game["opponent_wr_yards_allowed_avg_last_5"].null_count() == 2