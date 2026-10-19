"""Scheduled live-data refresh and manual model retraining."""

import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from the_script.data import live

LOGGER = logging.getLogger("the_script.refresh")
REFRESH_ENABLED = os.getenv("THE_SCRIPT_DAILY_REFRESH", "0") == "1"
EASTERN = ZoneInfo("America/New_York")
REFRESH_SLOTS = tuple((weekday, 3) for weekday in range(7))


def next_refresh_at(now: datetime | None = None) -> datetime:
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=EASTERN)
    current = current.astimezone(EASTERN)
    candidates = []
    for weekday, hour in REFRESH_SLOTS:
        candidate = current.replace(hour=hour, minute=0, second=0, microsecond=0)
        candidate += timedelta(days=(weekday - current.weekday()) % 7)
        if candidate <= current:
            candidate += timedelta(days=7)
        candidates.append(candidate)
    return min(candidates)


def seconds_until_next_refresh(now: datetime | None = None) -> float:
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=EASTERN)
    return max(0.0, next_refresh_at(current).timestamp() - current.timestamp())


def refresh_data(project_root: Path, on_refreshed: Callable[[], None] | None = None) -> bool:
    season = datetime.now(EASTERN).year
    if datetime.now(EASTERN).month < 3:
        season -= 1
    from nflreadpy.cache import clear_cache

    clear_cache()
    requests = [
        ("load_schedules", [season], {}),
        ("load_player_stats", [season], {"summary_level": "week"}),
        ("load_player_stats", [season - 1, season], {"summary_level": "week"}),
        ("load_pbp", [season], {}),
        ("load_pbp", [season - 1, season], {}),
        ("load_nextgen_stats", list(range(2016, season + 1)), {"stat_type": "receiving"}),
        ("load_snap_counts", [season], {}),
        ("load_depth_charts", [season], {}),
        ("load_injuries", [season], {}),
        ("load_players", None, {}),
    ]
    success = True
    for name, seasons, kwargs in requests:
        try:
            live._load(name, seasons, force_refresh=True, **kwargs)
        except Exception:
            success = False
            LOGGER.exception("Could not refresh %s; keeping existing snapshot", name)
    for pattern in ("ros_game_by_game_v*.json", "weekly_v*.json", "ros_features_*.parquet"):
        for path in (project_root / "data" / "cache").glob(pattern):
            path.unlink(missing_ok=True)
    if on_refreshed:
        on_refreshed()
    _run([sys.executable, "-m", "the_script.ranking_cache", "--project-root", str(project_root)], project_root)
    LOGGER.info("Live data refresh and ranking precompute finished")
    return success


def _run(command: list[str], project_root: Path) -> bool:
    LOGGER.info("Starting daily refresh command: %s", " ".join(command))
    result = subprocess.run(command, cwd=project_root, check=False)
    if result.returncode:
        LOGGER.error("Refresh command failed with exit code %s", result.returncode)
        return False
    return True


def refresh_models(project_root: Path) -> bool:
    """Rebuild feature data and model artifacts in dependency order."""
    cli = [sys.executable, "-m", "the_script.cli"]
    commands = [
        *[
            [*cli, "build-position-features", "--position", position]
            for position in ("QB", "RB", "TE")
        ],
        *[
            [*cli, "train-position-models", "--position", position]
            for position in ("QB", "RB", "TE")
        ],
        *[
            [*cli, "train-position-quantiles", "--position", position]
            for position in ("QB", "RB", "TE")
        ],
        [*cli, "build-player-features"],
        [*cli, "train-wr-baselines"],
        [*cli, "train-wr-neural"],
        [*cli, "train-wr-volume-models"],
        [*cli, "train-wr-touchdowns"],
        [*cli, "train-wr-quantiles"],
        [*cli, "train-wr-opponent-neural"],
        [*cli, "train-wr-opponent-experiment"],
        [*cli, "evaluate-yards-challengers"],
        [*cli, "compare-wr-models"],
    ]
    success = all(_run(command, project_root) for command in commands)
    if success:
        cache_directory = project_root / "data" / "cache"
        for pattern in ("ros_game_by_game_v*.json", "weekly_v*.json", "ros_features_*.parquet"):
            for path in cache_directory.glob(pattern):
                path.unlink(missing_ok=True)
        LOGGER.info("Daily refresh completed and ranking caches invalidated")
    return success


def _worker(project_root: Path, on_refreshed: Callable[[], None] | None = None) -> None:
    while True:
        time.sleep(seconds_until_next_refresh())
        try:
            if _run([sys.executable, "-m", "the_script.static_export", "--scheduled"], project_root) and on_refreshed:
                on_refreshed()
        except Exception:
            LOGGER.exception("Scheduled data refresh crashed")


def start_daily_refresh(
    project_root: Path, on_refreshed: Callable[[], None] | None = None
) -> threading.Thread | None:
    if not REFRESH_ENABLED:
        return None
    thread = threading.Thread(
        target=_worker,
        args=(project_root, on_refreshed),
        daemon=True,
        name="the-script-daily-refresh",
    )
    thread.start()
    LOGGER.info("Eastern data refresh enabled; next run: %s", next_refresh_at().isoformat())
    return thread