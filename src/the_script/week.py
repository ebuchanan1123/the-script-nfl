"""Persistent current NFL week with a local Tuesday 3:00 AM rollover."""

import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_INITIAL_WEEK = 4
_MAX_WEEK = 18
_ROLLOVER_DAYS = 7
_STATE_LOCK = threading.Lock()


def _next_tuesday_at_three(now: datetime) -> datetime:
    days_until_tuesday = (1 - now.weekday()) % 7
    rollover = now.replace(tzinfo=None, hour=3, minute=0, second=0, microsecond=0)
    rollover += timedelta(days=days_until_tuesday)
    if time.mktime(rollover.timetuple()) <= now.timestamp():
        rollover += timedelta(days=_ROLLOVER_DAYS)
    return rollover


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(state, indent=2))
    temporary_path.replace(path)


def get_current_week(state_path: Path, now: datetime | None = None) -> int:
    """Return the persisted week, advancing once per Tuesday at 3:00 AM local time."""
    current_time = now or datetime.now().astimezone()
    if current_time.tzinfo is None:
        current_time = current_time.astimezone()

    with _STATE_LOCK:
        if state_path.exists():
            state = json.loads(state_path.read_text())
            next_rollover = datetime.fromisoformat(state["next_rollover_at"]).replace(
                tzinfo=None
            )
            week = int(state["current_week"])
        else:
            next_rollover = _next_tuesday_at_three(current_time)
            week = _INITIAL_WEEK

        current_timestamp = current_time.timestamp()
        while time.mktime(next_rollover.timetuple()) <= current_timestamp:
            week = min(_MAX_WEEK, week + 1)
            next_rollover += timedelta(days=_ROLLOVER_DAYS)

        _write_state(
            state_path,
            {
                "current_week": week,
                "next_rollover_at": next_rollover.isoformat(),
            },
        )
        return week


def start_week_rollover(state_path: Path) -> threading.Thread:
    """Start the daemon that persists each Tuesday 3:00 AM week transition."""

    def worker() -> None:
        while True:
            week = get_current_week(state_path)
            if week >= _MAX_WEEK:
                return
            state = json.loads(state_path.read_text())
            next_rollover = datetime.fromisoformat(state["next_rollover_at"])
            delay = time.mktime(next_rollover.timetuple()) - time.time()
            time.sleep(max(0.05, delay))

    thread = threading.Thread(
        target=worker,
        daemon=True,
        name="the-script-week-rollover",
    )
    thread.start()
    return thread
