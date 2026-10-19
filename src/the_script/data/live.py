"""Persist live NFL source snapshots and reuse them during network outages."""

import hashlib
import json
import logging
import time
from pathlib import Path
from uuid import uuid4

import nflreadpy as nfl
import polars as pl

CACHE_DIRECTORY = Path(__file__).resolve().parents[3] / "data" / "cache" / "live_sources"
CACHE_MAX_AGE_SECONDS = 60 * 60
LOGGER = logging.getLogger(__name__)


def _load(
    name: str, seasons: list[int] | None = None, *, force_refresh: bool = False, **kwargs
) -> pl.DataFrame:
    # Include the complete request in the key: never substitute another season.
    key = hashlib.sha256(
        json.dumps([name, seasons, kwargs], sort_keys=True).encode()
    ).hexdigest()
    path = CACHE_DIRECTORY / f"{name}_{key}.parquet"
    if not force_refresh and path.exists() and time.time() - path.stat().st_mtime < CACHE_MAX_AGE_SECONDS:
        return pl.read_parquet(path)
    try:
        loader = getattr(nfl, name)
        frame = loader(**kwargs) if seasons is None else loader(seasons, **kwargs)
    except ConnectionError:
        if not path.exists():
            raise
        LOGGER.warning("NFL source unavailable; using saved snapshot %s", path.name)
        return pl.read_parquet(path)
    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid4().hex}.tmp")
    try:
        frame.write_parquet(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return frame


def load_schedules(seasons: list[int]) -> pl.DataFrame:
    return _load("load_schedules", seasons)


def load_player_stats(seasons: list[int], *, summary_level: str = "week") -> pl.DataFrame:
    return _load("load_player_stats", seasons, summary_level=summary_level)


def load_pbp(seasons: list[int]) -> pl.DataFrame:
    return _load("load_pbp", seasons)

def load_nextgen_receiving(seasons: list[int]) -> pl.DataFrame:
    return _load("load_nextgen_stats", seasons, stat_type="receiving")


def load_nextgen_receiving_optional(seasons: list[int]) -> pl.DataFrame:
    try:
        return load_nextgen_receiving(seasons)
    except ConnectionError:
        LOGGER.warning("Next Gen receiving metrics unavailable; using null features")
        return pl.DataFrame()


def load_snap_counts(seasons: list[int]) -> pl.DataFrame:
    return _load("load_snap_counts", seasons)


def load_depth_charts(seasons: list[int]) -> pl.DataFrame:
    return _load("load_depth_charts", seasons)


def load_injuries(seasons: list[int]) -> pl.DataFrame:
    return _load("load_injuries", seasons)


def load_players() -> pl.DataFrame:
    return _load("load_players")