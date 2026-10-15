from datetime import UTC, date, datetime

import polars as pl
import pytest

from the_script.features.injuries import add_injury_features, apply_availability, injury_risk_tags


def test_teammate_injury_uses_only_prior_target_share() -> None:
    games = pl.DataFrame(
        {
            "game_id": ["g3"],
            "game_date": [date(2026, 9, 27)],
            "season": [2026],
            "week": [3],
            "team": ["LAR"],
            "player_id": ["adams"],
        }
    )
    stats = pl.DataFrame(
        {
            "season": [2026, 2026, 2026, 2026],
            "week": [1, 2, 1, 2],
            "season_type": ["REG"] * 4,
            "player_id": ["nacua", "nacua", "adams", "adams"],
            "target_share": [0.3, 0.4, 0.2, 0.25],
        }
    )
    injuries = pl.DataFrame(
        {
            "season": [2026],
            "week": [3],
            "game_type": ["REG"],
            "team": ["LA"],
            "gsis_id": ["nacua"],
            "report_status": ["Questionable"],
            "practice_status": ["Did Not Participate In Practice"],
        }
    )

    result = add_injury_features(games, stats, injuries)

    assert result[0, "player_questionable"] == 0
    assert result[0, "teammate_questionable_target_share"] == pytest.approx(0.35)
    assert result[0, "teammate_out_or_doubtful_target_share"] == 0
    assert result[0, "injury_report_available"] == 1


def test_player_status_is_not_counted_as_teammate_opportunity() -> None:
    games = pl.DataFrame(
        {
            "game_id": ["g3"],
            "game_date": [date(2026, 9, 27)],
            "season": [2026],
            "week": [3],
            "team": ["LAR"],
            "player_id": ["adams"],
        }
    )
    stats = pl.DataFrame(
        {
            "season": [2026],
            "week": [2],
            "season_type": ["REG"],
            "player_id": ["adams"],
            "target_share": [0.25],
        }
    )
    injuries = pl.DataFrame(
        {
            "season": [2026],
            "week": [3],
            "game_type": ["REG"],
            "team": ["LA"],
            "gsis_id": ["adams"],
            "report_status": ["Questionable"],
            "practice_status": ["Limited Participation in Practice"],
        }
    )

    result = add_injury_features(games, stats, injuries)

    assert result[0, "player_questionable"] == 1
    assert result[0, "player_practice_limited"] == 1
    assert result[0, "teammate_questionable_target_share"] == 0


def test_unpublished_team_report_is_unknown_not_healthy() -> None:
    games = pl.DataFrame(
        {
            "game_id": ["g3"],
            "game_date": [date(2026, 9, 27)],
            "season": [2026],
            "week": [3],
            "team": ["LAR"],
            "player_id": ["adams"],
        }
    )

    result = add_injury_features(games, pl.DataFrame(), pl.DataFrame())

    assert result[0, "injury_report_available"] == 0
    assert result[0, "teammate_questionable_target_share"] is None

def test_latest_weekly_report_overrides_earlier_status() -> None:
    games = pl.DataFrame(
        {
            "game_id": ["g3"],
            "game_date": [date(2026, 9, 27)],
            "season": [2026],
            "week": [3],
            "team": ["LAR"],
            "player_id": ["nacua"],
        }
    )
    stats = pl.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "season_type": ["REG"],
            "player_id": ["nacua"],
            "target_share": [0.3],
        }
    )
    injuries = pl.DataFrame(
        {
            "season": [2026, 2026],
            "week": [3, 3],
            "game_type": ["REG", "REG"],
            "team": ["LA", "LA"],
            "gsis_id": ["nacua", "nacua"],
            "report_status": ["Out", "Questionable"],
            "practice_status": ["Did Not Participate In Practice", "Full Participation in Practice"],
            "date_modified": [
                datetime(2026, 9, 23, tzinfo=UTC),
                datetime(2026, 9, 26, tzinfo=UTC),
            ],
        }
    )

    result = add_injury_features(games, stats, injuries)

    assert result[0, "player_out_or_doubtful"] == 0
    assert result[0, "player_questionable"] == 1
    assert result[0, "player_practice_dnp"] == 0


def test_injury_risk_tags_use_latest_report_and_skip_healthy() -> None:
    injuries = pl.DataFrame(
        {
            "season": [2026] * 5,
            "week": [5] * 5,
            "game_type": ["REG"] * 5,
            "gsis_id": ["a", "a", "b", "c", "d"],
            "report_status": ["Out", "Questionable", "Doubtful", None, None],
            "practice_status": [None, None, None, "Limited Participation in Practice", "Full Participation in Practice"],
            "report_primary_injury": ["Knee", "Knee", "Hamstring", None, None],
            "practice_primary_injury": [None] * 5,
            "date_modified": [
                datetime(2026, 10, 7, tzinfo=UTC),
                datetime(2026, 10, 9, tzinfo=UTC),
                datetime(2026, 10, 8, tzinfo=UTC),
                datetime(2026, 10, 8, tzinfo=UTC),
                datetime(2026, 10, 8, tzinfo=UTC),
            ],
        }
    )

    tags = injury_risk_tags(injuries, 2026, 5)

    assert tags["a"]["label"] == "Questionable" and tags["a"]["risk"] == "medium"
    assert tags["b"]["risk"] == "high" and tags["b"]["injury"] == "Hamstring"
    assert tags["c"]["label"] == "Limited practice"
    assert "d" not in tags


def test_apply_availability_scales_projections_and_reranks() -> None:
    rows = [
        {"player_id": "a", "projected_yards": 100.0, "projected_receptions": 8.0, "projected_touchdowns": 1.0,
         "projected_ppr_points": 24.0, "predictions": {"receiving_yards": 100.0}},
        {"player_id": "b", "projected_yards": 60.0, "projected_receptions": 5.0, "projected_touchdowns": 0.5,
         "projected_ppr_points": 14.0},
    ]
    tags = {"a": {"play_probability": 0.5}}

    result = apply_availability(rows, tags)

    assert [r["player_id"] for r in result] == ["b", "a"]
    assert result[1]["projected_ppr_points"] == pytest.approx(12.0)
    assert result[1]["unadjusted_ppr_points"] == 24.0
    assert result[1]["predictions"]["receiving_yards"] == pytest.approx(50.0)
    assert result[0]["play_probability"] == 1.0
    assert rows[0]["projected_ppr_points"] == 24.0
