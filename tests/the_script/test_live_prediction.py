from datetime import date
import json
import os
import time

import polars as pl
import pytest

from the_script import ros
from the_script.predict import (
	build_live_feature_row,
	build_live_feature_rows,
	build_upcoming_player_row,
)


def test_upcoming_row_contains_no_current_game_outcomes() -> None:
	schedules = pl.DataFrame(
		{
			"game_id": ["2026_03_NYJ_DET"],
			"season": [2026],
			"week": [3],
			"gameday": ["2026-09-27"],
			"home_team": ["DET"],
			"away_team": ["NYJ"],
			"home_rest": [10],
			"away_rest": [7],
		}
	)

	row = build_upcoming_player_row(
		schedules, season=2026, week=3, team="DET", player_id="wr-1"
	)

	assert row[0, "opponent"] == "NYJ"
	assert row[0, "home_game"]
	assert row[0, "days_rest"] == 10
	assert row[0, "receiving_yards"] is None
	assert row[0, "snap_share"] is None


def test_week_two_blind_row_uses_schedule_without_week_one_outcomes() -> None:
	schedules = pl.DataFrame(
		{
			"game_id": ["2026_02_PHI_TEN"],
			"season": [2026],
			"week": [2],
			"gameday": ["2026-09-20"],
			"home_team": ["TEN"],
			"away_team": ["PHI"],
			"home_rest": [7],
			"away_rest": [7],
		}
	)

	row = build_upcoming_player_row(
		schedules, season=2026, week=2, team="PHI", player_id="wr-1"
	)

	assert not row[0, "home_game"]
	assert row[0, "opponent"] == "TEN"
	assert row[0, "receiving_yards"] is None


def test_upcoming_row_normalizes_rams_team_alias() -> None:
	schedules = pl.DataFrame(
		{
			"game_id": ["2026_03_LA_DEN"],
			"season": [2026],
			"week": [3],
			"gameday": ["2026-09-27"],
			"home_team": ["DEN"],
			"away_team": ["LA"],
			"home_rest": [7],
			"away_rest": [6],
		}
	)

	row = build_upcoming_player_row(
		schedules, season=2026, week=3, team="LA", player_id="wr-1"
	)

	assert row[0, "team"] == "LAR"
	assert row[0, "opponent"] == "DEN"


def test_future_feature_row_uses_completed_history_only() -> None:
	columns = {
		"game_id": ["g1", "g2"],
		"game_date": [date(2026, 9, 7), date(2026, 9, 14)],
		"season": [2026, 2026],
		"week": [1, 2],
		"player_id": ["wr-1", "wr-1"],
		"team": ["DET", "DET"],
		"opponent": ["NO", "BUF"],
		"home_game": [True, False],
		"days_rest": [7, 7],
		"position": ["WR", "WR"],
		"receiving_yards": [67, 142],
		"targets": [14, 13],
		"receptions": [10, 9],
		"receiving_touchdowns": [0, 1],
		"target_share": [0.3, 0.28],
		"air_yards_share": [0.4, 0.38],
		"wopr": [0.45, 0.42],
		"snap_share": [0.9, 0.95],
	}
	completed = pl.DataFrame(columns)
	upcoming = pl.DataFrame(
		{
			name: [value]
			for name, value in {
				"game_id": "g3",
				"game_date": date(2026, 9, 27),
				"season": 2026,
				"week": 3,
				"player_id": "wr-1",
				"team": "DET",
				"opponent": "NYJ",
				"home_game": True,
				"days_rest": 10,
				"position": "WR",
				"receiving_yards": None,
				"targets": None,
				"receptions": None,
				"receiving_touchdowns": None,
				"target_share": None,
				"air_yards_share": None,
				"wopr": None,
				"snap_share": None,
			}.items()
		}
	)

	row = build_live_feature_row(pl.DataFrame(schema=completed.schema), completed, upcoming)

	assert row[0, "yards_last_game"] == 142
	assert row[0, "yards_avg_last_3"] == 104.5
	assert row[0, "targets_avg_last_3"] == 13.5
	assert row[0, "target_share_avg_last_3"] == pytest.approx(0.29)
	assert row[0, "air_yards_share_last_game"] == 0.38
	assert row[0, "wopr_avg_last_5"] == pytest.approx(0.435)
	assert row[0, "snap_share_last_game"] == 0.95

	# A partially completed cutoff week can put a played game in both the
	# completed frame and the synthetic upcoming frame.
	placeholder = upcoming.with_columns(
		pl.lit("g2").alias("game_id"),
		pl.lit(date(2026, 9, 14)).alias("game_date"),
		pl.lit(2).alias("week"),
	)
	rows = build_live_feature_rows(
		pl.DataFrame(schema=completed.schema),
		completed,
		pl.concat([placeholder, upcoming], how="vertical_relaxed"),
	)
	assert rows.height == 2
	assert rows.filter(pl.col("game_id") == "g2")[0, "receiving_yards"] == 142
	assert rows.filter(pl.col("game_id") == "g3")[0, "yards_last_game"] == 142


def test_weekly_rankings_reload_when_ros_cache_is_newer(tmp_path, monkeypatch) -> None:
	cache_dir = tmp_path / "cache"
	cache_dir.mkdir()
	weekly_path = cache_dir / "weekly_v5_2026_3_3.json"
	ros_path = cache_dir / "ros_game_by_game_v5_2026_3.json"

	weekly_path.write_text(json.dumps([{"player_name": "Old Ranker", "team": "DET"}]))
	ros_path.write_text(json.dumps({"fresh": True}))

	now = time.time()
	os.utime(weekly_path, (now - 2 * 60 * 60, now - 2 * 60 * 60))
	os.utime(ros_path, (now - 1 * 60 * 60, now - 1 * 60 * 60))

	def fake_predict_ros_rankings(*args, **kwargs):
		fresh = [{"player_name": "Christian Watson", "team": "GB", "projected_ppr_points": 18.7}]
		weekly_path.write_text(json.dumps(fresh))
		return fresh

	monkeypatch.setattr(ros, "predict_ros_rankings", fake_predict_ros_rankings)

	result = ros.predict_weekly_rankings(
		2026,
		3,
		data_cutoff_week=3,
		historical_path=tmp_path / "not_used.parquet",
		cache_directory=cache_dir,
		cache_max_age_seconds=6 * 60 * 60,
	)

	assert result == [{"player_name": "Christian Watson", "team": "GB", "projected_ppr_points": 18.7}]