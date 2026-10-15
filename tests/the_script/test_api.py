from fastapi.testclient import TestClient

from the_script.api import app


def test_health_contract(monkeypatch) -> None:
	import polars as pl

	monkeypatch.setattr("the_script.api.get_current_week", lambda path: 4)
	monkeypatch.setattr(
		"the_script.api._player_stats", lambda season: pl.DataFrame({"week": [1, 2]})
	)
	response = TestClient(app).get("/api/health?season=2026")

	assert response.status_code == 200
	assert response.json()["current_week"] == 4
	assert response.json()["latest_completed_week"] == 2
	assert response.json()["models_ready"]


def test_prediction_endpoint_returns_primary_summary(monkeypatch) -> None:
	record = {
		"player_name": "Test Receiver",
		"player_id": "wr-1",
		"game_id": "2026_03_AAA_BBB",
		"game_date": "2026-09-20",
		"team": "AAA",
		"opponent": "BBB",
		"features": {
			"yards_last_game": 80.0,
			"opponent_wr_yards_allowed_last_game": 100.0,
			"opponent_wr_yards_allowed_avg_last_5": 120.0,
			"opponent_wr_ppr_points_allowed_avg_last_5": 25.0,
		},
		"predictions": {"ridge": 60.0, "xgboost": 65.0, "numerical_mlp": 70.0},
		"volume_predictions": {
			"targets": {"ridge": 8.0, "xgboost": 8.5},
			"receptions": {"ridge": 5.0, "xgboost": 5.5},
		},
		"expected_targets": 8.5,
		"expected_receptions": 5.5,
		"touchdown_predictions": {"poisson": 0.35, "xgboost_poisson": 0.4},
		"touchdown_model": "xgboost_poisson",
		"expected_receiving_touchdowns": 0.4,
		"anytime_touchdown_probability": 0.3296799539643607,
		"receiving_yards_quantiles": {"q10": 25.0, "q50": 61.0, "q90": 105.0},
		"decomposition_prediction": 64.0,
		"ensemble_weights": {
			"ridge": 0.0,
			"xgboost": 0.1,
			"numerical_mlp": 0.78,
			"decomposition": 0.12,
		},
		"ensemble_prediction": 68.0,
	}
	monkeypatch.setattr("the_script.api.predict_live_wr", lambda *args, **kwargs: record)
	monkeypatch.setattr(
		"the_script.api._player_rows",
		lambda *args: __import__("polars").DataFrame(
			{
				"week": [1],
				"game_id": ["g1"],
				"opponent_team": ["C"],
				"receiving_yards": [80],
				"targets": [9],
				"receptions": [6],
				"receiving_tds": [1],
			}
		),
	)
	response = TestClient(app).post(
		"/api/predict",
		json={"player_name": "Test Receiver", "team": "AAA", "season": 2026, "week": 3},
	)

	assert response.status_code == 200
	body = response.json()
	assert body["predicted_yards"] == 68.0
	assert body["primary_model"] == "weighted_ensemble"
	assert body["expected_targets"] == 8.5
	assert body["expected_receptions"] == 5.5
	assert body["expected_receiving_touchdowns"] == 0.4
	assert body["anytime_touchdown_probability"] == 0.3296799539643607
	assert body["model_range"] == {"low": 25.0, "high": 105.0}
	assert body["recent_games"][0]["yards"] == 80


def test_ros_rankings_endpoint_returns_cached_projection_rows(monkeypatch) -> None:
	rows = [
		{
			"player_id": "wr-1",
			"player_name": "Test Receiver",
			"team": "AAA",
			"remaining_games": 15,
			"projected_yards": 900.0,
			"projected_receptions": 75.0,
			"projected_touchdowns": 6.0,
			"projected_ppr_points": 201.0,
		}
	]
	monkeypatch.setattr("the_script.api._ros_rankings", lambda season, week: rows)

	response = TestClient(app).get("/api/ros-rankings?season=2026&week=3")

	assert response.status_code == 200
	assert response.json() == rows


def test_weekly_rankings_rejects_week_before_data_cutoff() -> None:
	response = TestClient(app).get(
		"/api/weekly-rankings?season=2026&week=2&data_cutoff_week=3"
	)

	assert response.status_code == 422


def test_position_prediction_rejects_unsupported_position() -> None:
	response = TestClient(app).post(
		"/api/predict-position",
		json={"player_name": "Test Player", "position": "K", "season": 2026, "week": 3},
	)

	assert response.status_code == 422


def test_rb_rankings_source_outage_returns_readable_cors_error(monkeypatch) -> None:
	monkeypatch.setattr("the_script.api.get_current_week", lambda path: 4)

	def unavailable(*args, **kwargs):
		raise ConnectionError("Failed to resolve github.com")

	monkeypatch.setattr("the_script.api.ranking_snapshot", unavailable)
	response = TestClient(app).get(
		"/api/weekly-rankings?season=2026&week=4&data_cutoff_week=4&position=RB",
		headers={"Origin": "http://127.0.0.1:5173"},
	)

	assert response.status_code == 503
	assert "github.com" in response.json()["detail"]
	assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"