"""Pregame WR receiving-yards inference from current nflverse data."""

import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from the_script.data import live as nfl
from the_script.data.build_dataset import CANONICAL_COLUMNS
from the_script.data.clean import canonical_team, canonical_team_code, clean_snap_counts
from the_script.features.advanced import add_advanced_features
from the_script.features.injuries import add_injury_features
from the_script.features.opponent import add_opponent_wr_features
from the_script.features.player import add_player_form_features
from the_script.features.role import add_role_features
from the_script.features.tendencies import add_team_tendency_features
from the_script.models.preprocessing import NUMERIC_FEATURES


def build_completed_live_wr_games(
	player_stats: pl.DataFrame,
	schedules: pl.DataFrame,
	snap_counts: pl.DataFrame,
	player_registry: pl.DataFrame,
) -> pl.DataFrame:
	"""Normalize completed current-season rows using the training-table contract."""
	stats = (
		player_stats.filter(
			(pl.col("season_type") == "REG")
			& (pl.col("position") == "WR")
			& pl.col("player_id").is_not_null()
		)
		.with_columns(
			canonical_team("team").alias("team"),
			canonical_team("opponent_team").alias("source_opponent"),
		)
		.unique(["game_id", "player_id"], keep="none")
	)
	participation = clean_snap_counts(snap_counts, player_registry)
	games = (
		schedules.filter((pl.col("game_type") == "REG") & pl.col("home_score").is_not_null())
		.with_columns(
			canonical_team("home_team").alias("home_team"),
			canonical_team("away_team").alias("away_team"),
			pl.col("gameday").str.to_date(strict=False).alias("game_date"),
		)
		.select(
			"game_id",
			"game_date",
			"season",
			"week",
			"home_team",
			"away_team",
			"home_rest",
			"away_rest",
		)
	)
	joined = (
		stats.join(games, on=["game_id", "season", "week"], how="inner", validate="m:1")
		.join(participation, on=["game_id", "player_id"], how="inner", validate="1:1")
	)
	is_home = pl.col("team") == pl.col("home_team")
	return (
		joined.with_columns(
			is_home.alias("home_game"),
			pl.when(is_home)
			.then(pl.col("away_team"))
			.otherwise(pl.col("home_team"))
			.alias("opponent"),
			pl.when(is_home)
			.then(pl.col("home_rest"))
			.otherwise(pl.col("away_rest"))
			.alias("days_rest"),
			pl.col("receiving_tds").alias("receiving_touchdowns"),
		)
		.select(CANONICAL_COLUMNS)
		.sort(["game_date", "game_id", "player_id"])
	)


def build_upcoming_player_row(
	schedules: pl.DataFrame,
	*,
	season: int,
	week: int,
	team: str,
	player_id: str,
) -> pl.DataFrame:
	"""Create an outcome-null row from pregame schedule information."""
	game = schedules.filter(
		(pl.col("season") == season)
		& (pl.col("week") == week)
		& ((pl.col("home_team") == team) | (pl.col("away_team") == team))
	)
	if game.height != 1:
		raise ValueError(f"Expected one {team} game in {season} week {week}, found {game.height}")
	record = game.row(0, named=True)
	home_game = record["home_team"] == team
	canonical_player_team = canonical_team_code(team)
	return pl.DataFrame(
		{
			"game_id": [record["game_id"]],
			"game_date": [record["gameday"]],
			"season": [season],
			"week": [week],
			"player_id": [player_id],
			"team": [canonical_player_team],
			"opponent": [
				canonical_team_code(
					record["away_team"] if home_game else record["home_team"]
				)
			],
			"home_game": [home_game],
			"days_rest": [record["home_rest"] if home_game else record["away_rest"]],
			"position": ["WR"],
			"receiving_yards": [None],
			"targets": [None],
			"receptions": [None],
			"receiving_touchdowns": [None],
			"target_share": [None],
			"air_yards_share": [None],
			"wopr": [None],
			"snap_share": [None],
		}
	).with_columns(pl.col("game_date").str.to_date(strict=False))


def build_live_feature_rows(
	historical_games: pl.DataFrame,
	completed_live_games: pl.DataFrame,
	upcoming_rows: pl.DataFrame,
	player_stats: pl.DataFrame | None = None,
	depth_charts: pl.DataFrame | None = None,
	play_by_play: pl.DataFrame | None = None,
	schedules: pl.DataFrame | None = None,
	injuries: pl.DataFrame | None = None,
	nextgen_receiving: pl.DataFrame | None = None,
) -> pl.DataFrame:
	"""Append outcome-null future rows and derive shifted pregame features."""
	all_games = pl.concat(
		[historical_games, completed_live_games, upcoming_rows], how="vertical_relaxed"
	)
	# A game that has already been played by the data cutoff can appear both as
	# real completed data and as a synthetic upcoming placeholder for the same
	# (game_id, player_id). Prefer the real data (concat order above puts it
	# first) so every downstream join keyed on (game_id, player_id) stays 1:1.
	all_games = all_games.unique(
		subset=["game_id", "player_id"], keep="first", maintain_order=True
	)
	with_form = add_player_form_features(all_games)
	with_role = add_role_features(
		with_form,
		player_stats if player_stats is not None else pl.DataFrame(),
		depth_charts if depth_charts is not None else pl.DataFrame(),
	)
	with_tendencies = add_team_tendency_features(
		with_role,
		player_stats if player_stats is not None else pl.DataFrame(),
		play_by_play if play_by_play is not None else pl.DataFrame(),
	)
	with_injuries = add_injury_features(
		with_tendencies,
		player_stats if player_stats is not None else pl.DataFrame(),
		injuries if injuries is not None else pl.DataFrame(),
	)
	with_advanced = add_advanced_features(
		with_injuries,
		schedules if schedules is not None else pl.DataFrame(),
		play_by_play if play_by_play is not None else pl.DataFrame(),
		nextgen_receiving,
	)
	featured = add_opponent_wr_features(with_advanced)
	keys = upcoming_rows.select("game_id", "player_id").unique()
	return featured.join(
		keys, on=["game_id", "player_id"], how="inner", validate="1:1"
	)


def build_live_feature_row(
	historical_games: pl.DataFrame,
	completed_live_games: pl.DataFrame,
	upcoming_row: pl.DataFrame,
	player_stats: pl.DataFrame | None = None,
	depth_charts: pl.DataFrame | None = None,
	play_by_play: pl.DataFrame | None = None,
	schedules: pl.DataFrame | None = None,
	injuries: pl.DataFrame | None = None,
	nextgen_receiving: pl.DataFrame | None = None,
) -> pl.DataFrame:
	"""Build one live feature row through the shared batched implementation."""
	return build_live_feature_rows(
		historical_games,
		completed_live_games,
		upcoming_row,
		player_stats=player_stats,
		depth_charts=depth_charts,
		play_by_play=play_by_play,
		schedules=schedules,
		injuries=injuries,
		nextgen_receiving=nextgen_receiving,
	)


def _predict_joblib(directory: Path, frame: pl.DataFrame) -> float:
	model = joblib.load(directory / "model.joblib")
	preprocessor = joblib.load(directory / "preprocessor.joblib")
	return float(model.predict(preprocessor.transform(frame))[0])


def _predict_neural(directory: Path, frame: pl.DataFrame) -> float:
	import torch

	from the_script.models.neural_network import NumericalWRModel

	preprocessor = joblib.load(directory / "preprocessor.joblib")
	model = NumericalWRModel(len(preprocessor.features))
	model.load_state_dict(
		torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
	)
	model.eval()
	with torch.no_grad():
		prediction = model(torch.from_numpy(preprocessor.transform(frame)))[0]
	return float(prediction)


def _predict_worker(model_type: str, directory: Path, row: dict[str, Any]) -> float:
	"""Load exactly one native model runtime inside an isolated process."""
	frame = pl.DataFrame([row])
	if model_type == "neural":
		return _predict_neural(directory, frame)
	return _predict_joblib(directory, frame)


def predict_model_artifacts(
	models_directory: Path, frame: pl.DataFrame
) -> dict[str, float]:
	"""Score each artifact in a spawned process to isolate OpenMP and PyTorch."""
	jobs = {
		"ridge": ("joblib", models_directory / "wr_opponent" / "ridge"),
		"xgboost": ("joblib", models_directory / "wr_opponent" / "xgboost"),
		"numerical_mlp": ("neural", models_directory / "wr_opponent" / "neural"),
	}
	row = frame.row(0, named=True)
	context = multiprocessing.get_context("spawn")
	with ProcessPoolExecutor(max_workers=3, mp_context=context) as executor:
		futures = {
			name: executor.submit(_predict_worker, model_type, directory, row)
			for name, (model_type, directory) in jobs.items()
		}
		return {name: future.result() for name, future in futures.items()}


def predict_volume_artifacts(
	models_directory: Path, frame: pl.DataFrame
) -> dict[str, dict[str, float]]:
	"""Score target and reception regressors in isolated native processes."""
	jobs = {
		(target, model): ("joblib", models_directory / "wr_volume" / target / model)
		for target in ("targets", "receptions")
		for model in ("ridge", "xgboost")
	}
	row = frame.row(0, named=True)
	context = multiprocessing.get_context("spawn")
	with ProcessPoolExecutor(max_workers=4, mp_context=context) as executor:
		futures = {
			key: executor.submit(_predict_worker, model_type, directory, row)
			for key, (model_type, directory) in jobs.items()
		}
		raw = {key: max(0.0, future.result()) for key, future in futures.items()}
	return {
		target: {model: raw[(target, model)] for model in ("ridge", "xgboost")}
		for target in ("targets", "receptions")
	}


def predict_touchdown_artifacts(
	models_directory: Path, frame: pl.DataFrame
) -> tuple[dict[str, float], str]:
	"""Score both touchdown families and return the validation-selected model."""
	directory = models_directory / "wr_touchdowns"
	model_names = ("poisson", "xgboost_poisson")
	row = frame.row(0, named=True)
	context = multiprocessing.get_context("spawn")
	with ProcessPoolExecutor(max_workers=2, mp_context=context) as executor:
		futures = {
			name: executor.submit(_predict_worker, "joblib", directory / name, row)
			for name in model_names
		}
		predictions = {
			name: max(0.0, future.result()) for name, future in futures.items()
		}
	comparison = json.loads((directory / "comparison.json").read_text())
	return predictions, comparison["selected_model"]


def predict_quantile_artifacts(
	models_directory: Path, frame: pl.DataFrame
) -> dict[str, float]:
	"""Score and monotonically order the receiving-yards quantiles."""
	directory = models_directory / "wr_quantiles"
	names = ("q10", "q50", "q90")
	row = frame.row(0, named=True)
	context = multiprocessing.get_context("spawn")
	with ProcessPoolExecutor(max_workers=3, mp_context=context) as executor:
		futures = {
			name: executor.submit(_predict_worker, "joblib", directory / name, row)
			for name in names
		}
		ordered = sorted(max(0.0, future.result()) for future in futures.values())
	return {name: ordered[index] for index, name in enumerate(names)}


def predict_decomposition_artifact(
	models_directory: Path, frame: pl.DataFrame
) -> float:
	"""Score the structured targets × catch rate × yards/reception challenger."""
	directory = models_directory / "wr_decomposition"
	preprocessor = joblib.load(directory / "preprocessor.joblib")
	values = preprocessor.transform(frame)
	targets = np.clip(
		joblib.load(directory / "targets_model.joblib").predict(values), 0.0, 20.0
	)
	catch_rate = np.clip(
		joblib.load(directory / "catch_rate_model.joblib").predict(values), 0.0, 1.0
	)
	yards_per_reception = np.clip(
		joblib.load(directory / "yards_per_reception_model.joblib").predict(values),
		0.0,
		40.0,
	)
	return float((targets * catch_rate * yards_per_reception)[0])


def predict_live_wr(
	player_name: str,
	season: int,
	week: int,
	*,
	team: str | None = None,
	exclude_current_season_data: bool = False,
	historical_path: Path = Path("data/processed/wr_player_games.parquet"),
	models_directory: Path = Path("models"),
	output_directory: Path = Path("predictions"),
) -> dict[str, Any]:
	"""Download current sources and predict one upcoming WR-game."""
	schedules = nfl.load_schedules([season])
	lookup_seasons = [season - 1] if exclude_current_season_data else [season - 1, season]
	player_stats = nfl.load_player_stats(lookup_seasons, summary_level="week")
	matches = player_stats.filter(pl.col("player_display_name") == player_name)
	player_ids = matches["player_id"].drop_nulls().unique().to_list()
	latest = matches.sort(["season", "week"]).tail(1)
	resolved_team = team or (latest[0, "team"] if latest.height else None)
	if len(player_ids) != 1 or resolved_team is None:
		raise ValueError(
			f"Could not uniquely resolve {player_name}: IDs={player_ids}, team={resolved_team}"
		)
	historical = pl.read_parquet(historical_path)
	if exclude_current_season_data:
		completed = historical.head(0)
	else:
		snap_counts = nfl.load_snap_counts([season])
		player_registry = nfl.load_players()
		completed = build_completed_live_wr_games(
			player_stats, schedules, snap_counts, player_registry
		)
	upcoming = build_upcoming_player_row(
		schedules,
		season=season,
		week=week,
		team=resolved_team,
		player_id=player_ids[0],
	)
	depth_charts = nfl.load_depth_charts([season])
	play_by_play = nfl.load_pbp(lookup_seasons)
	injuries = nfl.load_injuries([season])
	ngs_loader = getattr(nfl, "load_nextgen_receiving_optional", None)
	nextgen_receiving = ngs_loader(list(range(2016, season + 1))) if ngs_loader else pl.DataFrame()
	features = build_live_feature_row(
		historical,
		completed,
		upcoming,
		player_stats=player_stats,
		depth_charts=depth_charts,
		play_by_play=play_by_play,
		schedules=schedules,
		injuries=injuries,
		nextgen_receiving=nextgen_receiving,
	)
	predictions = predict_model_artifacts(models_directory, features)
	volume_predictions = predict_volume_artifacts(models_directory, features)
	expected_targets = volume_predictions["targets"]["xgboost"]
	expected_receptions = min(
		volume_predictions["receptions"]["xgboost"], expected_targets
	)
	touchdown_predictions, touchdown_model = predict_touchdown_artifacts(
		models_directory, features
	)
	expected_receiving_touchdowns = touchdown_predictions[touchdown_model]
	anytime_touchdown_probability = 1.0 - math.exp(-expected_receiving_touchdowns)
	receiving_yards_quantiles = predict_quantile_artifacts(models_directory, features)
	decomposition_prediction = predict_decomposition_artifact(models_directory, features)
	ensemble_metadata = json.loads(
		(models_directory / "wr_ensemble" / "comparison.json").read_text()
	)
	ensemble_components = {
		**predictions,
		"decomposition": decomposition_prediction,
	}
	ensemble_prediction = sum(
		ensemble_metadata["weights"][name] * value
		for name, value in ensemble_components.items()
	)
	record = {
		"player_name": player_name,
		"player_id": player_ids[0],
		"game_id": upcoming[0, "game_id"],
		"game_date": str(upcoming[0, "game_date"]),
		"team": resolved_team,
		"opponent": upcoming[0, "opponent"],
		"prediction_time_contract": "24 hours before kickoff",
		"data_cutoff": (
			f"{season - 1} regular season"
			if exclude_current_season_data
			else f"{season} games completed before week {week}"
		),
		"excluded_current_season_data": exclude_current_season_data,
		"features": {
			name: (None if value is None else float(value))
			for name, value in features.select(
				NUMERIC_FEATURES
			).row(0, named=True).items()
		},
		"predictions": predictions,
		"volume_predictions": volume_predictions,
		"expected_targets": expected_targets,
		"expected_receptions": expected_receptions,
		"touchdown_predictions": touchdown_predictions,
		"touchdown_model": touchdown_model,
		"expected_receiving_touchdowns": expected_receiving_touchdowns,
		"anytime_touchdown_probability": anytime_touchdown_probability,
		"receiving_yards_quantiles": receiving_yards_quantiles,
		"decomposition_prediction": decomposition_prediction,
		"ensemble_weights": ensemble_metadata["weights"],
		"ensemble_prediction": ensemble_prediction,
	}
	output_directory.mkdir(parents=True, exist_ok=True)
	suffix = "_blind" if exclude_current_season_data else ""
	(output_directory / f"{upcoming[0, 'game_id']}_{player_ids[0]}{suffix}.json").write_text(
		json.dumps(record, indent=2)
	)
	return record