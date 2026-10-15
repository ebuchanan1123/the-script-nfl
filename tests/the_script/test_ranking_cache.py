import json
from datetime import datetime
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from the_script import ranking_cache


def test_snapshot_reused_and_failed_refresh_preserves_previous(tmp_path, monkeypatch):
    calls = []

    def predict(*args, **kwargs):
        calls.append(kwargs)
        return [{"player_id": "qb-1", "projected_ppr_points": 20.0}]

    monkeypatch.setattr(ranking_cache, "predict_position_rankings", predict)
    first = ranking_cache.ranking_snapshot(tmp_path, "QB", 2026, 6, ros=False)
    assert ranking_cache.ranking_snapshot(tmp_path, "QB", 2026, 6, ros=False) == first
    assert len(calls) == 1

    def fail(*args, **kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(ranking_cache, "predict_position_rankings", fail)
    with pytest.raises(ConnectionError):
        ranking_cache.ranking_snapshot(tmp_path, "QB", 2026, 6, ros=False, rebuild=True)
    assert ranking_cache.ranking_snapshot(tmp_path, "QB", 2026, 6, ros=False) == first


def test_scheduled_prewarm_skips_completed_slot_and_detects_rollover(tmp_path, monkeypatch):
    directory = tmp_path / "data" / "cache" / "rankings"
    directory.mkdir(parents=True)
    (directory / "last_prewarm.json").write_text(json.dumps({
        "completed_at": datetime.now(ZoneInfo("America/New_York")).isoformat(), "week": 5,
    }))
    monkeypatch.setattr(ranking_cache, "get_current_week", lambda path: 5)
    compute = Mock(return_value=[])
    monkeypatch.setattr(ranking_cache, "ranking_snapshot", compute)
    ranking_cache.prewarm_rankings(tmp_path, scheduled=True)
    compute.assert_not_called()
    monkeypatch.setattr(ranking_cache, "get_current_week", lambda path: 6)
    ranking_cache.prewarm_rankings(tmp_path, scheduled=True)
    assert compute.call_count == 12