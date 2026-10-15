"""Persistent ranking snapshots, precomputed independently of browser requests."""

import argparse
import fcntl
import json
import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from the_script.models.multiposition import predict_position_rankings
from the_script.ros import predict_ros_rankings, predict_weekly_rankings
from the_script.week import get_current_week

LOGGER = logging.getLogger(__name__)


def ranking_snapshot(
    root: Path, position: str, season: int, week: int, *, ros: bool,
    cutoff: int | None = None, rebuild: bool = False,
) -> list[dict]:
    cutoff = cutoff or week
    directory = root / "data" / "cache" / "rankings"
    path = directory / f"{position}_{season}_{cutoff}_{week}_{'ros' if ros else 'weekly'}.json"
    if path.exists() and not rebuild:
        return json.loads(path.read_text())["rows"]
    directory.mkdir(parents=True, exist_ok=True)
    if position == "WR":
        with tempfile.TemporaryDirectory(dir=directory) as staging:
            kwargs = {
                "historical_path": root / "data" / "processed" / "wr_player_games.parquet",
                "models_directory": root / "models", "cache_directory": Path(staging),
            }
            rows = (predict_ros_rankings(season, week, **kwargs) if ros else
                    predict_weekly_rankings(season, week, data_cutoff_week=cutoff, **kwargs))
    else:
        rows = predict_position_rankings(
            position, season, week, ros=ros,
            processed_directory=root / "data" / "processed", models_directory=root / "models",
        )
    temporary = path.with_suffix(f".{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps({"generated_at": datetime.now().astimezone().isoformat(), "rows": rows}))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return rows


def prewarm_rankings(root: Path, *, scheduled: bool = False) -> None:
    directory = root / "data" / "cache" / "rankings"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "prewarm.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        now = datetime.now(ZoneInfo("America/New_York"))
        week = get_current_week(root / "data" / "current_week.json")
        marker = directory / "last_prewarm.json"
        if scheduled:
            from the_script.refresh import REFRESH_SLOTS

            slots = []
            for weekday, hour in REFRESH_SLOTS:
                slot = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                slot -= timedelta(days=(now.weekday() - weekday) % 7)
                if slot > now:
                    slot -= timedelta(days=7)
                slots.append(slot)
            latest = max(slots)
            if marker.exists():
                state = json.loads(marker.read_text())
                if state.get("week") == week and datetime.fromisoformat(state["completed_at"]) >= latest:
                    return
        season = now.year - (now.month < 3)
        success = True
        for position in ("WR", "QB", "RB", "TE"):
            for projection_week, ros in [(week, True), (week, False), *([(week + 1, False)] if week < 18 else [])]:
                try:
                    ranking_snapshot(root, position, season, projection_week, ros=ros, cutoff=week, rebuild=True)
                    LOGGER.info("Prepared %s %s week %s", position, "ROS" if ros else "weekly", projection_week)
                except Exception:
                    success = False
                    LOGGER.exception("Ranking precompute failed; retaining previous %s snapshot", position)
        if success:
            marker.write_text(json.dumps({"completed_at": now.isoformat(), "week": week}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--scheduled", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    prewarm_rankings(args.project_root, scheduled=args.scheduled)


if __name__ == "__main__":
    main()