import os
import time

import polars as pl
import pytest

from the_script.data import live


def test_persists_successful_source_and_reuses_snapshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(live, "CACHE_DIRECTORY", tmp_path)
    calls = []
    frame = pl.DataFrame({"season": [2026], "week": [4]})

    def loader(seasons):
        calls.append(seasons)
        return frame

    monkeypatch.setattr(live.nfl, "load_schedules", loader)
    assert live.load_schedules([2026]).equals(frame)
    assert live.load_schedules([2026]).equals(frame)
    assert calls == [[2026]]
    assert len(list(tmp_path.glob("*.parquet"))) == 1


def test_outage_reuses_expired_snapshot_but_not_another_season(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(live, "CACHE_DIRECTORY", tmp_path)
    frame = pl.DataFrame({"season": [2026], "week": [4]})
    monkeypatch.setattr(live.nfl, "load_schedules", lambda seasons: frame)
    live.load_schedules([2026])
    path = next(tmp_path.glob("*.parquet"))
    expired = time.time() - live.CACHE_MAX_AGE_SECONDS - 10
    os.utime(path, (expired, expired))

    def unavailable(seasons):
        raise ConnectionError("DNS lookup failed")

    monkeypatch.setattr(live.nfl, "load_schedules", unavailable)
    assert live.load_schedules([2026]).equals(frame)
    with pytest.raises(ConnectionError, match="DNS"):
        live.load_schedules([2027])


def test_optional_nextgen_source_returns_empty_when_unavailable(monkeypatch) -> None:
    def unavailable(*args, **kwargs):
        raise ConnectionError("source offline")

    monkeypatch.setattr(live, "_load", unavailable)
    assert live.load_nextgen_receiving_optional([2024]).is_empty()


def test_forced_refresh_bypasses_fresh_snapshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(live, "CACHE_DIRECTORY", tmp_path)
    old = pl.DataFrame({"week": [4]})
    new = pl.DataFrame({"week": [5]})
    monkeypatch.setattr(live.nfl, "load_schedules", lambda seasons: old)
    live.load_schedules([2026])
    monkeypatch.setattr(live.nfl, "load_schedules", lambda seasons: new)
    assert live._load("load_schedules", [2026], force_refresh=True).equals(new)
    assert live.load_schedules([2026]).equals(new)