"""Publish immutable forecast releases; replace the discovery manifest last.

Manifest file paths are relative to the output directory. All other JSON files
retain the API's list/dict contracts (forecasts are keyed by player ID). Run this
in a dedicated process: source and candidate patches are scoped but process-wide.
"""

import argparse
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4
from zoneinfo import ZoneInfo

import polars as pl

from the_script import api, ros
from the_script.data import live
from the_script.features.injuries import apply_availability, injury_risk_tags
from the_script.models import multiposition
from the_script.predict import predict_live_wr
from the_script.week import get_current_week

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POSITIONS = ("WR", "QB", "RB", "TE")
LOGGER = logging.getLogger(__name__)


class ExportError(RuntimeError):
    """A release cannot be published safely."""


def _write_json(path: Path, payload: Any) -> None:
    # Never coerce NaN/Infinity to null or silently drop missing model outputs.
    text = json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(text + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _snapshot_path(name: str, seasons: list[int] | None, kwargs: dict) -> Path:
    key = hashlib.sha256(
        json.dumps([name, seasons, kwargs], sort_keys=True).encode()
    ).hexdigest()
    return live.CACHE_DIRECTORY / f"{name}_{key}.parquet"


def _public_metrics(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _public_metrics(item) for key, item in value.items() if key != "parameters"}
    if isinstance(value, list):
        return [_public_metrics(item) for item in value]
    return value


@contextmanager
def _fresh_sources(season: int, allow_stale: bool):
    """Force each distinct request once, detecting live._load's silent fallback.

    A successful _load atomically rewrites its snapshot. An unchanged snapshot
    timestamp means the refresh was not verified, even if it was only seconds old.
    """
    original = live._load
    frames: dict[str, pl.DataFrame] = {}
    sources: list[dict[str, Any]] = []

    def load(name, seasons=None, *, force_refresh=False, **kwargs):
        key = json.dumps([name, seasons, kwargs], sort_keys=True)
        if key in frames:
            return frames[key]
        path = _snapshot_path(name, seasons, kwargs)
        before = path.stat().st_mtime_ns if path.exists() else None
        frame = original(name, seasons, force_refresh=True, **kwargs)
        after = path.stat().st_mtime_ns if path.exists() else None
        stale = after is None or after == before
        timestamp = (
            datetime.fromtimestamp(after / 1_000_000_000, UTC).isoformat()
            if after is not None else None
        )
        through_week = None
        if "season" in frame.columns and "week" in frame.columns:
            current = frame.filter(pl.col("season").cast(pl.Int32) == season)
            through_week = current["week"].max() if current.height else None
        sources.append({
            "name": name, "seasons": seasons, "parameters": kwargs,
            "snapshot_at": timestamp, "stale": stale,
            "data_through_week": through_week,
        })
        if stale and not allow_stale:
            raise ExportError(f"Source refresh failed or was unverified: {name} {seasons}")
        frames[key] = frame
        return frame

    # Bypass both API lru caches and nflreadpy's own in-memory network cache.
    from nflreadpy.cache import clear_cache

    clear_cache()
    with ExitStack() as stack:
        stack.enter_context(patch.object(live, "_load", load))
        stack.enter_context(patch.object(
            api, "_player_stats", lambda year: live.load_player_stats([year], summary_level="week")
        ))
        stack.enter_context(patch.object(api, "_schedules", lambda year: live.load_schedules([year])))
        yield sources


def _rankings(position: str, season: int, week: int, cutoff: int, *,
              rest_of_season: bool, temporary: Path) -> list[dict]:
    if position == "WR":
        kwargs = {
            "historical_path": PROJECT_ROOT / "data/processed/wr_player_games.parquet",
            "models_directory": PROJECT_ROOT / "models",
            "cache_directory": temporary,
        }
        if rest_of_season:
            return ros.predict_ros_rankings(season, cutoff, **kwargs)
        return ros.predict_weekly_rankings(season, week, data_cutoff_week=cutoff, **kwargs)
    return multiposition.predict_position_rankings(
        position, season, cutoff if rest_of_season else week, ros=rest_of_season,
        processed_directory=PROJECT_ROOT / "data/processed",
        models_directory=PROJECT_ROOT / "models",
    )


def _forecast(player: dict, position: str, season: int, week: int,
              matchup: dict, temporary: Path) -> dict:
    kwargs = {"team": player["team"], "models_directory": PROJECT_ROOT / "models"}
    if position == "WR":
        record = predict_live_wr(
            player["name"], season, week,
            historical_path=PROJECT_ROOT / "data/processed/wr_player_games.parquet",
            output_directory=temporary / "predictions", **kwargs,
        )
        required = {
            "predictions", "features", "ensemble_prediction", "ensemble_weights",
            "volume_predictions", "touchdown_predictions", "receiving_yards_quantiles",
            "decomposition_prediction",
        }
    else:
        record = multiposition.predict_live_position(
            player["name"], position, season, week,
            processed_directory=PROJECT_ROOT / "data/processed", **kwargs,
        )
        required = {
            "predictions", "features", "selected_models", "model_predictions",
            "primary_quantiles", "projected_ppr_points", "recent_games",
        }
    if (not isinstance(record, dict) or required.difference(record)
            or any(record[key] is None for key in required)
            or not record.get("predictions")
            or record.get("player_id") != player["player_id"]
            or record.get("game_id") != matchup["game_id"]):
        raise ExportError(f"Missing or mismatched forecast: {position} {player['player_id']} week {week}")
    return api._prediction_payload(record) if position == "WR" else record


def export_release(output: Path = PROJECT_ROOT / "frontend/public/data", *,
                   season: int | None = None, week: int | None = None,
                   limit: int | None = None, allow_stale: bool = False) -> dict:
    """Export all four positions, current/next week and ROS, or fail as a unit.

    --limit selects the first N API players *per position*, including rankings.
    Such releases, and stale-source releases, are explicitly preview-only.
    """
    now = datetime.now(UTC)
    season = season if season is not None else now.year - (now.month < 3)
    week = week if week is not None else get_current_week(api.CURRENT_WEEK_PATH)
    if not 2016 <= season <= 2100 or not 1 <= week <= 18:
        raise ValueError("Season must be 2016–2100 and week must be 1–18")
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive")
    weeks = list(range(week, min(week + 1, 18) + 1))
    output = Path(output).resolve()
    releases = output / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    release_id = now.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:12]
    destination = releases / release_id
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=releases))
    pointer = output / f".manifest-{release_id}.tmp"
    files: dict[str, Any] = {"players": {}, "forecasts": {}, "matchups": {},
                             "weekly": {}, "ros": {}, "injuries": {}, "profiles": {}}

    def write(relative: str, payload: Any) -> str:
        _write_json(staging / relative, payload)
        return f"releases/{release_id}/{relative}"

    try:
        with tempfile.TemporaryDirectory(prefix="the-script-export-") as scratch:
            temporary = Path(scratch)
            with _fresh_sources(season, allow_stale) as sources, ExitStack() as stack:
                health = api.health(season=season)
                health = {**health, "current_week": week}
                if health.get("latest_completed_week") is None:
                    raise ExportError("No completed player data is available")
                schedule = live.load_schedules([season])
                for projection_week in weeks:
                    games = schedule.filter(
                        (pl.col("season") == season) & (pl.col("week") == projection_week)
                        & (pl.col("game_type") == "REG")
                    )
                    if games.is_empty():
                        raise ExportError(f"No regular-season schedule for week {projection_week}")
                cohorts = {}
                for position in POSITIONS:
                    players = api.search_players(query="", season=season, limit=10000, position=position)
                    if limit is not None:
                        players = players[:limit]
                    ids = [player["player_id"] for player in players]
                    if not ids or len(set(ids)) != len(ids):
                        raise ExportError(f"Empty or duplicate player cohort: {position}")
                    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) for identifier in ids):
                        raise ExportError("Unsafe player ID")
                    cohorts[position] = players
                    files["players"][position] = write(f"players/{position}/players.json", players)
                    for player in players:
                        identifier = player["player_id"]
                        profile = api.player_profile(identifier, season=season)
                        if profile.get("player_id") != identifier:
                            raise ExportError(f"Missing profile: {identifier}")
                        files["profiles"][identifier] = write(f"player/{identifier}.json", profile)

                # Smoke exports also restrict the expensive batch ranking candidate lists.
                if limit is not None:
                    selected = {p["player_id"] for players in cohorts.values() for p in players}
                    def restrict(builder):
                        def candidates(*args, **kwargs):
                            return builder(*args, **kwargs).filter(pl.col("player_id").is_in(selected))
                        return candidates
                    for module in (ros, multiposition):
                        stack.enter_context(patch.object(
                            module, "position_ranking_candidates", restrict(module.position_ranking_candidates)
                        ))

                tags_by_week = {}
                for projection_week in weeks:
                    tags = injury_risk_tags(live.load_injuries([season]), season, projection_week)
                    tags_by_week[projection_week] = tags
                    files["injuries"][str(projection_week)] = write(f"injuries/{projection_week}.json", tags)
                byes: dict[str, dict[str, list[str]]] = {}
                for position, players in cohorts.items():
                    ids = {player["player_id"] for player in players}
                    files["forecasts"][position] = {}
                    files["matchups"][position] = {}
                    files["weekly"][position] = {}
                    byes[position] = {}
                    for projection_week in weeks:
                        forecasts, matchups, bye_ids = {}, {}, []
                        for player in players:
                            games = schedule.filter(
                                (pl.col("season") == season) & (pl.col("week") == projection_week)
                                & (pl.col("game_type") == "REG")
                                & ((pl.col("home_team") == player["team"])
                                   | (pl.col("away_team") == player["team"]))
                            )
                            # Only a genuine scheduled team with no game is a bye.
                            team_known = schedule.filter(
                                (pl.col("home_team") == player["team"])
                                | (pl.col("away_team") == player["team"])
                            ).height > 0
                            if games.is_empty() and team_known:
                                bye_ids.append(player["player_id"])
                                continue
                            if games.height != 1:
                                raise ExportError(f"Invalid schedule for {player['player_id']} week {projection_week}")
                            matchup = api._matchup(player["player_id"], season, projection_week)
                            if matchup.get("game_id") != games[0, "game_id"]:
                                raise ExportError("Matchup does not match schedule")
                            matchups[player["player_id"]] = matchup
                            forecasts[player["player_id"]] = _forecast(
                                player, position, season, projection_week, matchup, temporary
                            )
                        key = str(projection_week)
                        byes[position][key] = bye_ids
                        files["forecasts"][position][key] = write(f"forecasts/{position}/{projection_week}.json", forecasts)
                        files["matchups"][position][key] = write(f"matchups/{position}/{projection_week}.json", matchups)
                        rows = _rankings(position, season, projection_week, week,
                                         rest_of_season=False, temporary=temporary)
                        if limit is not None:
                            rows = [row for row in rows if row["player_id"] in ids]
                        rows = apply_availability(rows, tags_by_week[projection_week])
                        files["weekly"][position][key] = write(f"weekly/{position}/{projection_week}.json", rows)
                    rows = _rankings(position, season, week, week,
                                     rest_of_season=True, temporary=temporary)
                    if limit is not None:
                        rows = [row for row in rows if row["player_id"] in ids]
                    files["ros"][position] = write(f"ros/{position}.json", rows)
                stale = any(source["stale"] for source in sources)
                health = {**health, "source_stale": stale, "status": "stale" if stale else health["status"]}
                files["health"] = write("health.json", health)
                files["metrics"] = write("metrics.json", _public_metrics(api.model_metrics()))
                manifest = {
                    "schema_version": 1, "release_id": release_id,
                    "release_path": f"releases/{release_id}",
                    "generated_at": datetime.now(UTC).isoformat(), "season": season,
                    "current_week": week, "available_weeks": weeks, "positions": list(POSITIONS),
                    "data_through_week": health["latest_completed_week"],
                    "source_freshness": {"stale": stale, "sources": sources},
                    "preview_only": stale or limit is not None, "limit_per_position": limit,
                    "player_counts": {p: len(players) for p, players in cohorts.items()},
                    "byes": byes, "files": files,
                }
                _write_json(staging / "manifest.json", manifest)
        staging.replace(destination)
        _write_json(pointer, manifest)
        pointer.replace(output / "manifest.json")
        return manifest
    except Exception:
        # No reader can discover an incomplete release through the old pointer.
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(destination, ignore_errors=True)
        raise
    finally:
        pointer.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "frontend/public/data")
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument("--limit", type=int, help="Players per position (preview/smoke release)")
    parser.add_argument("--allow-stale", action="store_true", help="Allow labelled local-preview fallback")
    parser.add_argument("--scheduled", action="store_true", help="Publish once after each 3 AM Eastern slot")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    scheduled = args.scheduled
    del args.scheduled
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".export.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if scheduled:
            now = datetime.now(ZoneInfo("America/New_York"))
            slot = now.replace(hour=3, minute=0, second=0, microsecond=0)
            if slot > now:
                slot -= timedelta(days=1)
            pointer = args.output / "manifest.json"
            if pointer.exists():
                previous = json.loads(pointer.read_text())
                if not previous.get("preview_only") and datetime.fromisoformat(previous["generated_at"]) >= slot:
                    return
        try:
            manifest = export_release(**vars(args))
        except Exception as error:
            LOGGER.error("Export aborted; previous manifest retained: %s", error)
            raise SystemExit(1) from error
        LOGGER.info("Published release %s (preview_only=%s)", manifest["release_id"], manifest["preview_only"])


if __name__ == "__main__":
    main()