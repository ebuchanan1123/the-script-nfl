import json
import os
from contextlib import contextmanager
from unittest.mock import Mock

import polars as pl
import pytest

from the_script import static_export as exporter


@pytest.fixture
def backend(monkeypatch, tmp_path):
    @contextmanager
    def fresh(*args):
        yield [{"name": "load_player_stats", "stale": False, "snapshot_at": "2026-10-12T00:00:00+00:00"}]

    monkeypatch.setattr(exporter, "_fresh_sources", fresh)
    monkeypatch.setattr(exporter.api, "health", lambda **kwargs: {
        "status": "ok", "current_week": 4, "latest_completed_week": 3,
        "models_ready": True,
    })
    schedule = pl.DataFrame([
        {"season": 2026, "week": week, "game_type": "REG", "home_team": "A",
         "away_team": "B", "game_id": f"2026_{week:02}_A_B"}
        for week in (4, 5)
    ] + [{"season": 2026, "week": 6, "game_type": "REG", "home_team": "C",
          "away_team": "B", "game_id": "2026_06_C_B"}])
    monkeypatch.setattr(exporter.live, "load_schedules", lambda *args: schedule)
    monkeypatch.setattr(exporter.live, "load_injuries", lambda *args: pl.DataFrame())
    monkeypatch.setattr(exporter, "injury_risk_tags", lambda *args: {
        "WR-1": {"play_probability": 0.63, "status": "Questionable"}
    })
    search = Mock(side_effect=lambda **kwargs: [
        {"player_id": f"{kwargs['position']}-{i}", "name": f"Player {i}",
         "team": "C" if i == 2 else "A", "position": kwargs["position"]}
        for i in (1, 2)
    ])
    monkeypatch.setattr(exporter.api, "search_players", search)
    monkeypatch.setattr(exporter.api, "player_profile", lambda identifier, **kwargs: {
        "player_id": identifier, "recent_games": []
    })
    matchup = Mock(side_effect=lambda identifier, season, week: {
        "game_id": f"2026_{week:02}_A_B", "week": week, "opponent": "B"
    })
    monkeypatch.setattr(exporter.api, "_matchup", matchup)
    monkeypatch.setattr(exporter.api, "_prediction_payload", lambda record: {
        **record, "primary_model": "weighted_ensemble", "recent_games": []
    })
    wr_record = {
        "player_id": "WR-1", "game_id": "2026_04_A_B", "features": {"missing": None},
        "predictions": {"ridge": 40.0, "xgboost": 42.0, "numerical_mlp": 41.0},
        "ensemble_prediction": 41.0, "ensemble_weights": {"ridge": 0.5},
        "volume_predictions": {"targets": {"xgboost": 6.0}},
        "touchdown_predictions": {"poisson": 0.2},
        "receiving_yards_quantiles": {"q10": 10.0, "q50": 40.0, "q90": 80.0},
        "decomposition_prediction": 39.0, "custom_existing_field": {"keep": True},
    }

    def wr(name, season, week, **kwargs):
        directory = kwargs["output_directory"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "history.json").write_text("{}")
        return {**wr_record, "game_id": f"2026_{week:02}_A_B"}

    wr_predict = Mock(side_effect=wr)
    monkeypatch.setattr(exporter, "predict_live_wr", wr_predict)
    position_predict = Mock(side_effect=lambda name, position, season, week, **kwargs: {
        "player_id": f"{position}-1", "game_id": f"2026_{week:02}_A_B",
        "features": {}, "predictions": {"yards": 50.0},
        "selected_models": {"yards": "weighted_blend"}, "model_predictions": {"ridge": 48.0},
        "primary_quantiles": {"q10": 20.0, "q50": 50.0, "q90": 90.0},
        "projected_ppr_points": 5.0, "recent_games": [],
    })
    monkeypatch.setattr(exporter.multiposition, "predict_live_position", position_predict)
    rankings = Mock(side_effect=lambda position, *args, **kwargs: [
        {"player_id": f"{position}-1", "projected_yards": 41.0,
         "projected_receptions": 4.0, "projected_touchdowns": 0.2,
         "projected_ppr_points": 9.3}
    ])
    monkeypatch.setattr(exporter, "_rankings", rankings)
    monkeypatch.setattr(exporter.api, "model_metrics", lambda: {"real_metrics": {"rmse": 25.0}})
    return {"search": search, "wr": wr_predict, "position": position_predict,
            "matchup": matchup, "rankings": rankings, "wr_record": wr_record}


def read(output, path):
    return json.loads((output / path).read_text())


def test_full_contract_and_bye_exemption(tmp_path, backend):
    output = tmp_path / "public/data"
    manifest = exporter.export_release(output, season=2026, week=4)
    assert read(output, "manifest.json") == manifest
    assert read(output, manifest["release_path"] + "/manifest.json") == manifest
    assert manifest["available_weeks"] == [4, 5]
    assert manifest["data_through_week"] == 3
    assert manifest["generated_at"]
    assert not manifest["preview_only"]
    assert not manifest["source_freshness"]["stale"]
    files = manifest["files"]
    for position in exporter.POSITIONS:
        assert len(read(output, files["players"][position])) == 2
        assert read(output, files["profiles"][position + "-1"])["recent_games"] == []
        for week in ("4", "5"):
            forecasts = read(output, files["forecasts"][position][week])
            assert set(forecasts) == {position + "-1"}
            assert manifest["byes"][position][week] == [position + "-2"]
            assert read(output, files["matchups"][position][week])[position + "-1"]["opponent"] == "B"
        assert read(output, files["ros"][position])[0]["projected_ppr_points"] == 9.3
    wr = read(output, files["forecasts"]["WR"]["4"])["WR-1"]
    for key, value in backend["wr_record"].items():
        assert wr[key] == value
    assert wr["primary_model"] == "weighted_ensemble"
    assert read(output, files["weekly"]["WR"]["4"])[0]["projected_yards"] == pytest.approx(41 * 0.63)
    assert read(output, files["injuries"]["4"])["WR-1"]["play_probability"] == 0.63
    assert read(output, files["metrics"]) == {"real_metrics": {"rmse": 25.0}}
    assert read(output, files["health"])["status"] == "ok"
    assert backend["wr"].call_count == 2
    assert backend["position"].call_count == 6
    assert backend["matchup"].call_count == 8
    assert all(c.kwargs["query"] == "" and c.kwargs["limit"] == 10000
               for c in backend["search"].call_args_list)
    assert not any(p.name.startswith(".staging") for p in (output / "releases").iterdir())
    assert all(not c.kwargs["output_directory"].exists() for c in backend["wr"].call_args_list)


def test_limit_selects_subset_and_labels_preview(tmp_path, backend):
    manifest = exporter.export_release(tmp_path, season=2026, week=4, limit=1)
    assert manifest["preview_only"]
    assert manifest["limit_per_position"] == 1
    assert manifest["player_counts"] == dict.fromkeys(exporter.POSITIONS, 1)
    assert len(manifest["files"]["profiles"]) == 4


@pytest.mark.parametrize("failure", ["exception", "missing", "null", "identity", "nan", "inf", "metrics", "matchup"])
def test_failure_retains_previous_release_and_cleans_staging(tmp_path, backend, monkeypatch, failure):
    previous = exporter.export_release(tmp_path, season=2026, week=4)
    before = (tmp_path / "manifest.json").read_bytes()
    if failure == "exception":
        backend["wr"].side_effect = ConnectionError("offline")
    elif failure == "matchup":
        backend["matchup"].side_effect = RuntimeError("missing matchup is not a bye")
    elif failure == "metrics":
        monkeypatch.setattr(exporter.api, "model_metrics", lambda: {"rmse": float("nan")})
    else:
        record = dict(backend["wr_record"])
        if failure == "missing":
            record.pop("receiving_yards_quantiles")
        elif failure == "null":
            record["receiving_yards_quantiles"] = None
        elif failure == "identity":
            record["player_id"] = "wrong"
        else:
            record["features"] = {"bad": float(failure)}
        backend["wr"].side_effect = None
        backend["wr"].return_value = record
    with pytest.raises((exporter.ExportError, ValueError, ConnectionError, RuntimeError)):
        exporter.export_release(tmp_path, season=2026, week=4)
    assert (tmp_path / "manifest.json").read_bytes() == before
    assert [p.name for p in (tmp_path / "releases").iterdir()] == [previous["release_id"]]
    assert not list(tmp_path.glob(".manifest-*"))


def test_pointer_is_last_and_pointer_failure_rolls_back(tmp_path, backend, monkeypatch):
    previous = exporter.export_release(tmp_path, season=2026, week=4)
    original = exporter._write_json

    def fail_pointer(path, payload):
        if path.name.startswith(".manifest-"):
            assert (tmp_path / payload["release_path"] / "manifest.json").exists()
            assert read(tmp_path, "manifest.json") == previous
            raise OSError("disk full")
        original(path, payload)

    monkeypatch.setattr(exporter, "_write_json", fail_pointer)
    with pytest.raises(OSError):
        exporter.export_release(tmp_path, season=2026, week=4)
    assert read(tmp_path, "manifest.json") == previous
    assert len(list((tmp_path / "releases").iterdir())) == 1


def test_versioned_releases_are_retained(tmp_path, backend):
    first = exporter.export_release(tmp_path, season=2026, week=4)
    second = exporter.export_release(tmp_path, season=2026, week=4)
    assert first["release_id"] != second["release_id"]
    assert read(tmp_path, "manifest.json") == second
    assert read(tmp_path, first["files"]["metrics"])
    assert len(list((tmp_path / "releases").iterdir())) == 2


@pytest.mark.parametrize("allow_stale", [False, True])
def test_source_refresh_fallback_detection(tmp_path, monkeypatch, allow_stale):
    monkeypatch.setattr(exporter.live, "CACHE_DIRECTORY", tmp_path)
    monkeypatch.setattr("nflreadpy.cache.clear_cache", Mock())
    kwargs = {"summary_level": "week"}
    path = exporter._snapshot_path("load_player_stats", [2026], kwargs)
    frame = pl.DataFrame({"season": [2026], "week": [3]})
    frame.write_parquet(path)
    loader = Mock(return_value=frame)
    monkeypatch.setattr(exporter.live, "_load", loader)
    original_stats = exporter.api._player_stats
    with exporter._fresh_sources(2026, allow_stale) as sources:
        if not allow_stale:
            with pytest.raises(exporter.ExportError, match="refresh failed"):
                exporter.live.load_player_stats([2026])
        else:
            assert exporter.api._player_stats(2026).equals(frame)
            assert exporter.live.load_player_stats([2026]).equals(frame)
            assert loader.call_count == 1
        assert sources[0]["stale"]
        assert sources[0]["snapshot_at"]
        assert sources[0]["data_through_week"] == 3
    assert exporter.api._player_stats is original_stats
    assert exporter.live._load is loader
    assert loader.call_args.kwargs["force_refresh"] is True


def test_fresh_sources_cached_by_complete_request(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter.live, "CACHE_DIRECTORY", tmp_path)
    monkeypatch.setattr("nflreadpy.cache.clear_cache", Mock())
    frame = pl.DataFrame({"season": [2026], "week": [3]})

    def refresh(name, seasons=None, **kwargs):
        assert kwargs.pop("force_refresh") is True
        path = exporter._snapshot_path(name, seasons, kwargs)
        frame.write_parquet(path)
        os.utime(path, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
        return frame

    loader = Mock(side_effect=refresh)
    monkeypatch.setattr(exporter.live, "_load", loader)
    with exporter._fresh_sources(2026, False) as sources:
        exporter.live.load_player_stats([2026])
        exporter.live.load_player_stats([2026])
        exporter.live.load_player_stats([2025, 2026])
        exporter.live.load_player_stats([2026], summary_level="reg")
        assert loader.call_count == 3
        assert not any(s["stale"] for s in sources)
    assert exporter.live._load is loader


def test_actual_live_loader_outage_is_rejected_even_with_recent_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter.live, "CACHE_DIRECTORY", tmp_path)
    monkeypatch.setattr("nflreadpy.cache.clear_cache", Mock())
    path = exporter._snapshot_path("load_schedules", [2026], {})
    pl.DataFrame({"season": [2026], "week": [4]}).write_parquet(path)
    upstream = Mock(side_effect=ConnectionError("offline"))
    monkeypatch.setattr(exporter.live.nfl, "load_schedules", upstream)
    with exporter._fresh_sources(2026, False), pytest.raises(exporter.ExportError, match="load_schedules"):
        exporter.live.load_schedules([2026])
    upstream.assert_called_once_with([2026])


def test_upstream_failure_without_snapshot_is_not_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter.live, "CACHE_DIRECTORY", tmp_path)
    monkeypatch.setattr("nflreadpy.cache.clear_cache", Mock())
    monkeypatch.setattr(exporter.live.nfl, "load_schedules", Mock(side_effect=ConnectionError("offline")))
    with exporter._fresh_sources(2026, True), pytest.raises(ConnectionError):
        exporter.live.load_schedules([2026])


def test_ranking_routes_rebuild_in_scratch(monkeypatch, tmp_path):
    weekly = Mock(return_value=[{"player_id": "WR-1"}])
    ros = Mock(return_value=[{"player_id": "WR-1"}])
    position = Mock(return_value=[{"player_id": "QB-1"}])
    monkeypatch.setattr(exporter.ros, "predict_weekly_rankings", weekly)
    monkeypatch.setattr(exporter.ros, "predict_ros_rankings", ros)
    monkeypatch.setattr(exporter.multiposition, "predict_position_rankings", position)
    exporter._rankings("WR", 2026, 5, 4, rest_of_season=False, temporary=tmp_path)
    exporter._rankings("WR", 2026, 5, 4, rest_of_season=True, temporary=tmp_path)
    exporter._rankings("QB", 2026, 5, 4, rest_of_season=False, temporary=tmp_path)
    exporter._rankings("QB", 2026, 5, 4, rest_of_season=True, temporary=tmp_path)
    assert weekly.call_args.args == (2026, 5)
    assert weekly.call_args.kwargs["data_cutoff_week"] == 4
    assert weekly.call_args.kwargs["cache_directory"] == tmp_path
    assert ros.call_args.args == (2026, 4)
    assert ros.call_args.kwargs["cache_directory"] == tmp_path
    assert position.call_args_list[0].args == ("QB", 2026, 5)
    assert position.call_args_list[1].args == ("QB", 2026, 4)
    assert position.call_args_list[1].kwargs["ros"] is True


def test_stale_preview_labels_manifest_and_health(tmp_path, backend, monkeypatch):
    @contextmanager
    def stale(*args):
        yield [{"name": "load_schedules", "stale": True, "snapshot_at": "2026-10-01T00:00:00+00:00"}]
    monkeypatch.setattr(exporter, "_fresh_sources", stale)
    manifest = exporter.export_release(tmp_path, season=2026, week=4, allow_stale=True)
    assert manifest["source_freshness"]["stale"]
    assert manifest["preview_only"]
    assert read(tmp_path, manifest["files"]["health"])["status"] == "stale"


def test_cli_arguments_and_failure_exit(monkeypatch, tmp_path):
    run = Mock(return_value={"release_id": "example", "preview_only": True})
    monkeypatch.setattr(exporter, "export_release", run)
    exporter.main(["--output", str(tmp_path), "--season", "2026", "--week", "4",
                   "--limit", "1", "--allow-stale"])
    run.assert_called_once_with(output=tmp_path, season=2026, week=4, limit=1, allow_stale=True)
    run.side_effect = exporter.ExportError("offline")
    with pytest.raises(SystemExit) as error:
        exporter.main(["--output", str(tmp_path)])
    assert error.value.code == 1


@pytest.mark.parametrize("kwargs", [{"week": 0}, {"week": 19}, {"season": 2015}, {"limit": 0}])
def test_invalid_arguments_do_not_stage(tmp_path, kwargs):
    with pytest.raises(ValueError):
        exporter.export_release(tmp_path, season=kwargs.get("season", 2026),
                                week=kwargs.get("week", 4), limit=kwargs.get("limit"))
    assert not (tmp_path / "releases").exists()