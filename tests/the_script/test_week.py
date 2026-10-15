import json
import time
from datetime import datetime
from pathlib import Path

from the_script.week import get_current_week


def local_datetime(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    timestamp = time.mktime((year, month, day, hour, minute, 0, -1, -1, -1))
    return datetime.fromtimestamp(timestamp).astimezone()


def test_current_week_starts_at_week_four_and_schedules_next_tuesday(tmp_path: Path) -> None:
    state_path = tmp_path / "current_week.json"

    week = get_current_week(state_path, local_datetime(2026, 9, 29, 2, 59))

    assert week == 4
    assert state_path.exists()


def test_current_week_increments_at_tuesday_three_am(tmp_path: Path) -> None:
    state_path = tmp_path / "current_week.json"
    before_rollover = local_datetime(2026, 9, 29, 2, 59)
    at_rollover = local_datetime(2026, 9, 29, 3)

    assert get_current_week(state_path, before_rollover) == 4
    assert get_current_week(state_path, at_rollover) == 5
    assert get_current_week(state_path, local_datetime(2026, 10, 6, 3)) == 6


def test_rollover_remains_at_three_am_after_daylight_saving_change(tmp_path: Path) -> None:
	state_path = tmp_path / "current_week.json"
	assert get_current_week(state_path, local_datetime(2026, 10, 27, 2, 59)) == 4
	assert get_current_week(state_path, local_datetime(2026, 10, 27, 3)) == 5
	assert get_current_week(state_path, local_datetime(2026, 11, 3, 2, 59)) == 5
	assert get_current_week(state_path, local_datetime(2026, 11, 3, 3)) == 6


def test_current_week_caps_at_week_eighteen(tmp_path: Path) -> None:
    state_path = tmp_path / "current_week.json"
    state_path.write_text(
        json.dumps(
            {
                "current_week": 17,
                "next_rollover_at": "2027-01-19T03:00:00",
            }
        )
    )

    assert get_current_week(state_path, local_datetime(2027, 1, 19, 3)) == 18
    assert get_current_week(state_path, local_datetime(2027, 1, 26, 3)) == 18
