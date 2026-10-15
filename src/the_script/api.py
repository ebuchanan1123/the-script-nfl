"""FastAPI service for The Script's WR prediction dashboard."""

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from the_script.data import live as nfl
from the_script.features.injuries import apply_availability, injury_risk_tags
from the_script.models.multiposition import (
	POSITION_CONFIGS,
	predict_live_position,
)
from the_script.predict import predict_live_wr
from the_script.ranking_cache import ranking_snapshot
from the_script.refresh import start_daily_refresh
from the_script.ros import predict_ros_rankings
from the_script.week import get_current_week, start_week_rollover

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIRECTORY = PROJECT_ROOT / "models"
PREDICTIONS_DIRECTORY = PROJECT_ROOT / "predictions"
CURRENT_WEEK_PATH = PROJECT_ROOT / "data" / "current_week.json"


class PredictionRequest(BaseModel):
	player_name: str = Field(min_length=2)
	team: str = Field(min_length=2, max_length=3)
	season: int = Field(ge=2016, le=2100)
	week: int = Field(ge=1, le=22)
	exclude_current_season_data: bool = False


class PositionPredictionRequest(BaseModel):
	player_name: str = Field(min_length=2)
	position: str
	team: str | None = Field(default=None, min_length=2, max_length=3)
	season: int = Field(ge=2016, le=2100)
	week: int = Field(ge=1, le=22)


@lru_cache(maxsize=8)
def _player_stats(season: int) -> pl.DataFrame:
	return nfl.load_player_stats([season], summary_level="week")


@lru_cache(maxsize=8)
def _schedules(season: int) -> pl.DataFrame:
	return nfl.load_schedules([season])


def _ros_rankings(season: int, week: int) -> list[dict[str, Any]]:
	return predict_ros_rankings(
		season,
		week,
		historical_path=PROJECT_ROOT / "data" / "processed" / "wr_player_games.parquet",
		models_directory=MODELS_DIRECTORY,
		cache_directory=PROJECT_ROOT / "data" / "cache",
	)


def _json(path: Path) -> dict[str, Any]:
	return json.loads(path.read_text())


def _as_float(value: Any) -> float | None:
	return None if value is None else float(value)


def _player_rows(player_id: str, season: int) -> pl.DataFrame:
	return _player_stats(season).filter(pl.col("player_id") == player_id).sort("week")


def _resolve_player(player_id: str, season: int) -> dict[str, Any]:
	rows = _player_rows(player_id, season)
	if rows.is_empty():
		raise HTTPException(status_code=404, detail="Player was not found in that season")
	latest = rows.tail(1).row(0, named=True)
	return {
		"player_id": player_id,
		"name": latest["player_display_name"],
		"team": latest["team"],
		"position": latest["position"],
		"headshot_url": latest.get("headshot_url"),
		"games_played": rows.height,
		"latest_week": int(rows["week"].max()),
		"season_receiving_yards": int(rows["receiving_yards"].sum()),
		"season_targets": int(rows["targets"].sum()),
		"season_receptions": int(rows["receptions"].sum()),
	}


def _game_rows(rows: pl.DataFrame) -> list[dict[str, Any]]:
	return [
		{
			"week": int(row["week"]),
			"game_id": row["game_id"],
			"opponent": row["opponent_team"],
			"yards": int(row["receiving_yards"]),
			"targets": int(row["targets"]),
			"receptions": int(row["receptions"]),
			"touchdowns": int(row["receiving_tds"]),
		}
		for row in rows.iter_rows(named=True)
	]


def _matchup(player_id: str, season: int, week: int) -> dict[str, Any]:
	player = _resolve_player(player_id, season)
	schedule = _schedules(season)
	game = schedule.filter(
		(pl.col("week") == week)
		& (
			(pl.col("home_team") == player["team"])
			| (pl.col("away_team") == player["team"])
		)
	)
	if game.height != 1:
		raise HTTPException(status_code=404, detail="Upcoming matchup was not found")
	record = game.row(0, named=True)
	is_home = record["home_team"] == player["team"]
	return {
		"game_id": record["game_id"],
		"season": season,
		"week": week,
		"game_date": record["gameday"],
		"game_time": record.get("gametime"),
		"team": player["team"],
		"opponent": record["away_team"] if is_home else record["home_team"],
		"home_game": is_home,
		"days_rest": record["home_rest"] if is_home else record["away_rest"],
		"stadium": record.get("stadium"),
		"roof": record.get("roof"),
		"temperature": record.get("temp"),
		"wind": record.get("wind"),
	}


def _prediction_payload(record: dict[str, Any]) -> dict[str, Any]:
	predictions = record["predictions"]
	values = list(predictions.values())
	quantiles = record.get("receiving_yards_quantiles")
	features = record["features"]
	season = int(record["game_id"].split("_")[0])
	week = int(record["game_id"].split("_")[1])
	rows = _player_rows(record["player_id"], season).filter(pl.col("week") < week)
	return {
		**record,
		"primary_model": "weighted_ensemble",
		"predicted_yards": record["ensemble_prediction"],
		"model_range": {
			"low": quantiles["q10"] if quantiles else min(values),
			"high": quantiles["q90"] if quantiles else max(values),
		},
		"recent_games": _game_rows(rows.tail(5)),
		"opponent_context": {
			"yards_allowed_last_game": features.get("opponent_wr_yards_allowed_last_game"),
			"yards_allowed_avg_last_5": features.get("opponent_wr_yards_allowed_avg_last_5"),
			"ppr_points_allowed_avg_last_5": features.get(
				"opponent_wr_ppr_points_allowed_avg_last_5"
			),
		},
		"generated_at": datetime.now(UTC).isoformat(),
	}


@asynccontextmanager
async def lifespan(_: FastAPI):
	def clear_source_caches() -> None:
		_player_stats.cache_clear()
		_schedules.cache_clear()

	start_daily_refresh(PROJECT_ROOT, clear_source_caches)
	yield


app = FastAPI(title="The Script API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
	CORSMiddleware,
	allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.exception_handler(ConnectionError)
async def data_source_unavailable(request, error: ConnectionError) -> JSONResponse:
	return JSONResponse(
		status_code=503,
		content={
			"detail": (
				"NFL data source is unreachable. Check your internet connection or DNS "
				"access to github.com, then retry. No saved snapshot is available "
				"for this request."
			),
		},
	)


@app.get("/api/health")
def health(season: int = 2026) -> dict[str, Any]:
	stats = _player_stats(season)
	return {
		"status": "ok",
		"service": "the-script-api",
		"season": season,
		"current_week": get_current_week(CURRENT_WEEK_PATH),
		"latest_completed_week": int(stats["week"].max()) if stats.height else None,
		"models_ready": all(
			path.exists()
			for path in [
				*(MODELS_DIRECTORY / "wr_opponent" / model for model in ("ridge", "xgboost", "neural")),
				*(
					MODELS_DIRECTORY / "wr_volume" / target / model
					for target in ("targets", "receptions")
					for model in ("ridge", "xgboost")
				),
				*(
					MODELS_DIRECTORY / "wr_touchdowns" / model
					for model in ("poisson", "xgboost_poisson")
				),
				*(
					MODELS_DIRECTORY / "wr_quantiles" / quantile
					for quantile in ("q10", "q50", "q90")
				),
			]
		),
		"checked_at": datetime.now(UTC).isoformat(),
	}


@app.get("/api/injuries")
def injuries(season: int = 2026, week: int | None = None) -> dict[str, Any]:
	week = week or get_current_week(CURRENT_WEEK_PATH)
	return injury_risk_tags(nfl.load_injuries([season]), season, week)


@app.get("/api/players")
def search_players(
	query: str = Query(default="", max_length=80),
	season: int = 2026,
	limit: int = 12,
	position: str = "WR",
) -> list[dict[str, Any]]:
	position = position.upper()
	if position not in {"WR", *POSITION_CONFIGS.keys()}:
		raise HTTPException(status_code=422, detail="Position must be QB, RB, WR, or TE")
	stats = _player_stats(season).filter(pl.col("position") == position)
	if query.strip():
		stats = stats.filter(
			pl.col("player_display_name")
			.str.to_lowercase()
			.str.contains(query.strip().lower(), literal=True)
		)
	players = (
		stats.sort(["player_id", "week"])
		.group_by("player_id", maintain_order=True)
		.agg(
			pl.col("player_display_name").last().alias("name"),
			pl.col("team").last().alias("team"),
			pl.col("position").last().alias("position"),
			pl.col("headshot_url").last().alias("headshot_url"),
			pl.len().alias("games_played"),
			pl.col("receiving_yards").sum().alias("season_receiving_yards"),
			pl.col("receiving_tds").sum().alias("season_receiving_tds"),
			pl.col("targets").sum().alias("season_targets"),
			pl.col("receptions").sum().alias("season_receptions"),
			pl.col("passing_yards").sum().alias("season_passing_yards"),
			pl.col("rushing_yards").sum().alias("season_rushing_yards"),
			pl.col("rushing_tds").sum().alias("season_rushing_tds"),
			pl.col("carries").sum().alias("season_carries"),
			pl.col("fantasy_points_ppr").sum().alias("season_ppr_points"),
		)
		.sort(
			"season_ppr_points" if position != "WR" else "season_receiving_yards",
			descending=True,
		)
		.head(limit)
	)
	return players.to_dicts()


@app.get("/api/players/{player_id}")
def player_profile(player_id: str, season: int = 2026) -> dict[str, Any]:
	rows = _player_rows(player_id, season)
	return {**_resolve_player(player_id, season), "recent_games": _game_rows(rows.tail(5))}


@app.get("/api/matchup/{player_id}")
def matchup(player_id: str, season: int = 2026, week: int | None = None) -> dict[str, Any]:
	week = week or get_current_week(CURRENT_WEEK_PATH)
	return _matchup(player_id, season, week)


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
	try:
		record = predict_live_wr(
			request.player_name,
			request.season,
			request.week,
			team=request.team,
			exclude_current_season_data=request.exclude_current_season_data,
			historical_path=PROJECT_ROOT / "data" / "processed" / "wr_player_games.parquet",
			models_directory=MODELS_DIRECTORY,
			output_directory=PREDICTIONS_DIRECTORY,
		)
	except ValueError as error:
		raise HTTPException(status_code=422, detail=str(error)) from error
	return _prediction_payload(record)


@app.post("/api/predict-position")
def predict_position(request: PositionPredictionRequest) -> dict[str, Any]:
	position = request.position.upper()
	if position not in POSITION_CONFIGS:
		raise HTTPException(status_code=422, detail="Position must be QB, RB, or TE")
	try:
		return predict_live_position(
			request.player_name,
			position,
			request.season,
			request.week,
			team=request.team,
			processed_directory=PROJECT_ROOT / "data" / "processed",
			models_directory=MODELS_DIRECTORY,
		)
	except ValueError as error:
		raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/predictions")
def prediction_history(limit: int = 20) -> list[dict[str, Any]]:
	if not PREDICTIONS_DIRECTORY.exists():
		return []
	paths = sorted(
		PREDICTIONS_DIRECTORY.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True
	)
	return [
		{
			"id": path.stem,
			"file_name": path.name,
			"created_at": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
			**_json(path),
		}
		for path in paths[:limit]
	]


@app.get("/api/ros-rankings")
def ros_rankings(season: int = 2026, week: int | None = None) -> list[dict[str, Any]]:
	week = week or get_current_week(CURRENT_WEEK_PATH)
	if week < 1 or week > 18:
		raise HTTPException(status_code=422, detail="Week must be between 1 and 18")
	return _ros_rankings(season, week)


@app.get("/api/position-ros-rankings")
def position_ros_rankings(
	position: str = "WR", season: int = 2026, week: int | None = None
) -> list[dict[str, Any]]:
	week = week or get_current_week(CURRENT_WEEK_PATH)
	position = position.upper()
	if position == "WR":
		return ranking_snapshot(PROJECT_ROOT, position, season, week, ros=True)
	if position not in POSITION_CONFIGS:
		raise HTTPException(status_code=422, detail="Position must be QB, RB, WR, or TE")
	return ranking_snapshot(PROJECT_ROOT, position, season, week, ros=True)


@app.get("/api/weekly-rankings")
def weekly_rankings(
	season: int = 2026,
	week: int | None = None,
	data_cutoff_week: int | None = None,
	position: str = "WR",
) -> list[dict[str, Any]]:
	current_week = get_current_week(CURRENT_WEEK_PATH)
	week = week or current_week
	data_cutoff_week = data_cutoff_week or current_week
	position = position.upper()
	if week < data_cutoff_week or week > 18:
		raise HTTPException(
			status_code=422,
			detail=f"Week must be between {data_cutoff_week} and 18",
		)
	if position != "WR" and position not in POSITION_CONFIGS:
		raise HTTPException(status_code=422, detail="Position must be QB, RB, WR, or TE")
	rows = ranking_snapshot(PROJECT_ROOT, position, season, week, ros=False, cutoff=data_cutoff_week)
	return apply_availability(rows, injury_risk_tags(nfl.load_injuries([season]), season, week))


@app.get("/api/predictions/{prediction_id}")
def prediction_detail(prediction_id: str) -> dict[str, Any]:
	path = PREDICTIONS_DIRECTORY / f"{prediction_id}.json"
	if not path.exists() or path.parent != PREDICTIONS_DIRECTORY:
		raise HTTPException(status_code=404, detail="Prediction was not found")
	return _json(path)


@app.get("/api/models/metrics")
def model_metrics() -> dict[str, Any]:
	comparison = _json(MODELS_DIRECTORY / "wr_model_comparison.json")
	opponent = _json(MODELS_DIRECTORY / "wr_opponent" / "comparison.json")
	return {
		"base": comparison,
		"opponent_experiment": opponent,
		"production_model": "validation_weighted_ensemble",
		"training_seasons": "2016-2023",
		"validation_season": 2024,
		"test_season": 2025,
	}


def run() -> None:
	import uvicorn

	start_week_rollover(CURRENT_WEEK_PATH)
	uvicorn.run("the_script.api:app", host="127.0.0.1", port=8000, reload=False)