"""Cached rest-of-season WR projections from batched production inference."""

import json
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from the_script.data import live as nfl
from the_script.data.clean import canonical_team_code
from the_script.models.multiposition import position_ranking_candidates
from the_script.predict import (
    build_completed_live_wr_games,
    build_live_feature_rows,
    build_upcoming_player_row,
)

_NEURAL_INFERENCE_LOCK = threading.Lock()


def _score_artifact(model_type: str, directory: Path, frame_path: Path) -> np.ndarray:
    frame = pl.read_parquet(frame_path)
    preprocessor = joblib.load(directory / "preprocessor.joblib")
    values = preprocessor.transform(frame)
    if model_type == "neural":
        import torch

        from the_script.models.neural_network import NumericalWRModel

        model = NumericalWRModel(len(preprocessor.features))
        model.load_state_dict(
            torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        )
        model.eval()
        with torch.no_grad():
            return model(torch.from_numpy(values)).numpy()
    model = joblib.load(directory / "model.joblib")
    return model.predict(values)


def _isolated_score(model_type: str, directory: Path, frame_path: Path) -> np.ndarray:
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        return np.asarray(
            executor.submit(_score_artifact, model_type, directory, frame_path).result()
        )


def _score_neural_safely(directory: Path, frame_path: Path) -> np.ndarray:
    with _NEURAL_INFERENCE_LOCK:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        import torch

        torch.set_num_threads(1)
        return _score_artifact("neural", directory, frame_path)


def _decomposition(models_directory: Path, features: pl.DataFrame) -> np.ndarray:
    directory = models_directory / "wr_decomposition"
    preprocessor = joblib.load(directory / "preprocessor.joblib")
    values = preprocessor.transform(features)
    targets = np.clip(joblib.load(directory / "targets_model.joblib").predict(values), 0.0, 20.0)
    catch_rate = np.clip(
        joblib.load(directory / "catch_rate_model.joblib").predict(values), 0.0, 1.0
    )
    yards_per_reception = np.clip(
        joblib.load(directory / "yards_per_reception_model.joblib").predict(values),
        0.0,
        40.0,
    )
    return targets * catch_rate * yards_per_reception


def standard_ppr_points(yards: float, receptions: float, touchdowns: float) -> float:
    return receptions + yards / 10.0 + touchdowns * 6.0


def _remaining_games(schedules: pl.DataFrame, season: int, week: int) -> dict[str, int]:
    games = schedules.filter(
        (pl.col("season") == season) & (pl.col("game_type") == "REG") & (pl.col("week") >= week)
    )
    teams = pl.concat(
        [
            games.select(pl.col("home_team").alias("team")),
            games.select(pl.col("away_team").alias("team")),
        ]
    ).with_columns(pl.col("team").map_elements(canonical_team_code, return_dtype=pl.String))
    return dict(teams.group_by("team").len().iter_rows())


def predict_ros_rankings(
    season: int,
    week: int,
    *,
    historical_path: Path = Path("data/processed/wr_player_games.parquet"),
    models_directory: Path = Path("models"),
    cache_directory: Path = Path("data/cache"),
    cache_max_age_seconds: int = 6 * 60 * 60,
) -> list[dict[str, Any]]:
    """Project current per-game roles over each team's remaining schedule."""
    cache_path = cache_directory / f"ros_game_by_game_v5_{season}_{week}.json"
    if cache_path.exists() and time.time() - cache_path.stat().st_mtime < cache_max_age_seconds:
        return json.loads(cache_path.read_text())

    schedules = nfl.load_schedules([season])
    player_stats = nfl.load_player_stats([season - 1, season], summary_level="week")
    depth_charts = nfl.load_depth_charts([season])
    players = position_ranking_candidates(player_stats, depth_charts, schedules, "WR", season, week)
    snap_counts = nfl.load_snap_counts([season])
    player_registry = nfl.load_players()
    completed = build_completed_live_wr_games(player_stats, schedules, snap_counts, player_registry)
    historical = pl.read_parquet(historical_path)
    play_by_play = nfl.load_pbp([season - 1, season])
    injuries = nfl.load_injuries([season])
    ngs_loader = getattr(nfl, "load_nextgen_receiving_optional", None)
    nextgen_receiving = ngs_loader(list(range(2016, season + 1))) if ngs_loader else pl.DataFrame()
    upcoming_by_week: dict[int, list[pl.DataFrame]] = {}
    metadata = []
    for player in players.iter_rows(named=True):
        team = player["team"]
        future_games = schedules.filter(
            (pl.col("season") == season)
            & (pl.col("game_type") == "REG")
            & (pl.col("week") >= week)
            & pl.col("home_score").is_null()
            & ((pl.col("home_team") == team) | (pl.col("away_team") == team))
        ).sort("week")
        if future_games.is_empty():
            continue
        for projection_week in future_games["week"].to_list():
            row = build_upcoming_player_row(
                schedules,
                season=season,
                week=int(projection_week),
                team=team,
                player_id=player["player_id"],
            )
            upcoming_by_week.setdefault(int(projection_week), []).append(row)
            metadata.append(
                {
                    **player,
                    "team": canonical_team_code(team),
                    "projection_week": int(projection_week),
                }
            )

    feature_groups = []
    for next_week, rows in upcoming_by_week.items():
        upcoming = pl.concat(rows, how="vertical_relaxed")
        feature_groups.append(
            build_live_feature_rows(
                historical,
                completed,
                upcoming,
                player_stats=player_stats,
                depth_charts=depth_charts,
                play_by_play=play_by_play,
                schedules=schedules,
                injuries=injuries,
                nextgen_receiving=nextgen_receiving,
            ).with_columns(pl.lit(next_week).alias("projection_week"))
        )
    features = pl.concat(feature_groups, how="vertical_relaxed").join(
        pl.DataFrame(metadata).select("player_id", "projection_week", "player_name"),
        on=["player_id", "projection_week"],
        how="inner",
        validate="1:1",
    )
    cache_directory.mkdir(parents=True, exist_ok=True)
    frame_path = cache_directory / f"ros_features_{season}_{week}.parquet"
    features.write_parquet(frame_path)

    direct = {
        "ridge": _isolated_score("joblib", models_directory / "wr_opponent" / "ridge", frame_path),
        "xgboost": _isolated_score(
            "joblib", models_directory / "wr_opponent" / "xgboost", frame_path
        ),
    }
    direct["numerical_mlp"] = _score_neural_safely(
        models_directory / "wr_opponent" / "neural", frame_path
    )
    decomposition = _decomposition(models_directory, features)
    weights = json.loads((models_directory / "wr_ensemble" / "comparison.json").read_text())[
        "weights"
    ]
    yards_per_game = sum(weights[name] * values for name, values in direct.items())
    yards_per_game += weights["decomposition"] * decomposition
    receptions_per_game = np.clip(
        _isolated_score(
            "joblib",
            models_directory / "wr_volume" / "receptions" / "xgboost",
            frame_path,
        ),
        0.0,
        None,
    )
    touchdown_model = json.loads(
        (models_directory / "wr_touchdowns" / "comparison.json").read_text()
    )["selected_model"]
    touchdowns_per_game = np.clip(
        _isolated_score("joblib", models_directory / "wr_touchdowns" / touchdown_model, frame_path),
        0.0,
        None,
    )

    totals: dict[str, dict[str, Any]] = {}
    weekly_results: dict[int, list[dict[str, Any]]] = {}
    for index, row in enumerate(features.iter_rows(named=True)):
        projected_yards = max(0.0, float(yards_per_game[index]))
        projected_receptions = float(receptions_per_game[index])
        projected_touchdowns = float(touchdowns_per_game[index])
        weekly_results.setdefault(row["projection_week"], []).append(
            {
                "player_id": row["player_id"],
                "player_name": row["player_name"],
                "team": row["team"],
                "opponent": row["opponent"],
                "week": row["projection_week"],
                "projected_yards": projected_yards,
                "projected_receptions": projected_receptions,
                "projected_touchdowns": projected_touchdowns,
                "projected_ppr_points": standard_ppr_points(
                    projected_yards, projected_receptions, projected_touchdowns
                ),
            }
        )
        result = totals.setdefault(
            row["player_id"],
            {
                "player_id": row["player_id"],
                "player_name": row["player_name"],
                "team": row["team"],
                "remaining_games": 0,
                "projected_yards": 0.0,
                "projected_receptions": 0.0,
                "projected_touchdowns": 0.0,
            },
        )
        result["remaining_games"] += 1
        result["projected_yards"] += projected_yards
        result["projected_receptions"] += projected_receptions
        result["projected_touchdowns"] += projected_touchdowns
    results = list(totals.values())
    for result in results:
        result["projected_ppr_points"] = standard_ppr_points(
            result["projected_yards"],
            result["projected_receptions"],
            result["projected_touchdowns"],
        )
    results.sort(key=lambda item: item["projected_ppr_points"], reverse=True)
    for projection_week, weekly in weekly_results.items():
        weekly.sort(key=lambda item: item["projected_ppr_points"], reverse=True)
        (cache_directory / f"weekly_v5_{season}_{week}_{projection_week}.json").write_text(
            json.dumps(weekly, indent=2)
        )
    cache_path.write_text(json.dumps(results, indent=2))
    return results


def predict_weekly_rankings(
    season: int,
    projection_week: int,
    *,
    data_cutoff_week: int,
    historical_path: Path = Path("data/processed/wr_player_games.parquet"),
    models_directory: Path = Path("models"),
    cache_directory: Path = Path("data/cache"),
    cache_max_age_seconds: int = 6 * 60 * 60,
) -> list[dict[str, Any]]:
    """Return one-game rankings using data available before the cutoff week."""
    cache_path = cache_directory / (f"weekly_v5_{season}_{data_cutoff_week}_{projection_week}.json")
    ros_cache_path = cache_directory / f"ros_game_by_game_v5_{season}_{data_cutoff_week}.json"
    if (
        cache_path.exists()
        and time.time() - cache_path.stat().st_mtime < cache_max_age_seconds
        and not (
            ros_cache_path.exists() and ros_cache_path.stat().st_mtime > cache_path.stat().st_mtime
        )
    ):
        return json.loads(cache_path.read_text())
    predict_ros_rankings(
        season,
        data_cutoff_week,
        historical_path=historical_path,
        models_directory=models_directory,
        cache_directory=cache_directory,
        cache_max_age_seconds=0,
    )
    if not cache_path.exists():
        return []
    return json.loads(cache_path.read_text())
