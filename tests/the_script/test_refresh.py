from datetime import datetime
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from the_script import refresh


@pytest.mark.parametrize(
    "now, expected",
    [
        ("2026-10-11T17:59:00-04:00", "2026-10-12T03:00:00-04:00"),
        ("2026-10-12T03:00:00-04:00", "2026-10-13T03:00:00-04:00"),
        ("2026-10-13T02:01:00-04:00", "2026-10-13T03:00:00-04:00"),
        ("2026-11-01T01:00:00-04:00", "2026-11-01T03:00:00-05:00"),
        ("2026-10-12T05:30:00+00:00", "2026-10-12T03:00:00-04:00"),
    ],
)
def test_eastern_schedule(now, expected):
    assert refresh.next_refresh_at(datetime.fromisoformat(now)).isoformat() == expected


def test_delay_across_fall_dst_transition():
    now = datetime(2026, 10, 31, 3, tzinfo=ZoneInfo("America/New_York"))
    assert refresh.seconds_until_next_refresh(now) == 25 * 3600


def test_refresh_forces_sources_and_invalidates_rankings(tmp_path, monkeypatch):
    cache = tmp_path / "data" / "cache"
    cache.mkdir(parents=True)
    stale = cache / "weekly_v5_2026_5_6.json"
    stale.write_text("[]")
    unrelated = cache / "keep.json"
    unrelated.write_text("{}")
    loader = Mock()
    monkeypatch.setattr(refresh.live, "_load", loader)
    monkeypatch.setattr(refresh, "_run", Mock(return_value=True))
    monkeypatch.setattr("nflreadpy.cache.clear_cache", Mock())
    callback = Mock()
    assert refresh.refresh_data(tmp_path, callback)
    assert loader.call_count == 9
    assert all(call.kwargs["force_refresh"] for call in loader.call_args_list)
    assert not stale.exists()
    assert unrelated.exists()
    callback.assert_called_once()


def test_source_failure_does_not_stop_other_sources(tmp_path, monkeypatch):
    loader = Mock(side_effect=[ConnectionError("offline"), *([None] * 8)])
    monkeypatch.setattr(refresh.live, "_load", loader)
    monkeypatch.setattr(refresh, "_run", Mock(return_value=True))
    monkeypatch.setattr("nflreadpy.cache.clear_cache", Mock())
    callback = Mock()
    assert not refresh.refresh_data(tmp_path, callback)
    assert loader.call_count == 9
    callback.assert_called_once()