"""Download immutable core nflverse sources for the first WR milestone."""

import json
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from the_script import SEASONS

CORE_FILES = {
    "players": "players.parquet",
    "player_stats": "player_stats.parquet",
    "schedules": "schedules.parquet",
    "snap_counts": "snap_counts.parquet",
    "depth_charts": "depth_charts.parquet",
    "play_by_play": "play_by_play.parquet",
    "injuries": "injuries.parquet",
    "nextgen_receiving": "nextgen_receiving.parquet",
}


def _schema_payload(frame: pl.DataFrame) -> dict[str, str]:
    return {name: str(dtype) for name, dtype in frame.schema.items()}


def download_core_data(
    raw_directory: Path,
    interim_directory: Path,
    seasons: list[int] | tuple[int, ...] = SEASONS,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Download weekly player stats and schedules once, preserving raw schemas."""
    requested = sorted(set(seasons))
    unsupported = sorted(set(requested).difference(SEASONS))
    if unsupported:
        raise ValueError(f"Seasons outside the 2016-2025 contract: {unsupported}")

    raw_directory.mkdir(parents=True, exist_ok=True)
    interim_directory.mkdir(parents=True, exist_ok=True)
    paths = {name: raw_directory / filename for name, filename in CORE_FILES.items()}
    loaders = {
        "players": nfl.load_players,
        "player_stats": lambda: nfl.load_player_stats(requested, summary_level="week"),
        "schedules": lambda: nfl.load_schedules(requested),
        "snap_counts": lambda: nfl.load_snap_counts(requested),
        "depth_charts": lambda: nfl.load_depth_charts(requested),
        "play_by_play": lambda: nfl.load_pbp(requested),
        "injuries": lambda: nfl.load_injuries(requested),
        "nextgen_receiving": lambda: nfl.load_nextgen_stats(requested, stat_type="receiving"),
    }
    schemas = {}
    for name, loader in loaders.items():
        path = paths[name]
        if path.exists() and not overwrite:
            frame = pl.read_parquet(path)
        else:
            frame = loader()
            frame.write_parquet(path)
        schemas[name] = {
            "rows": frame.height,
            "columns": _schema_payload(frame),
            "seasons": requested,
        }

    schema_path = interim_directory / "source_schemas.json"
    schema_path.write_text(json.dumps(schemas, indent=2, sort_keys=True))
    paths["schemas"] = schema_path
    return paths
