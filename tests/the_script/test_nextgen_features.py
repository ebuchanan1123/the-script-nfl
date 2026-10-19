from datetime import date, timedelta

import polars as pl

from the_script.features.advanced import add_advanced_features
from the_script.features.nextgen import NGS_METRICS, add_nextgen_receiving_features


def _games() -> pl.DataFrame:
    return pl.DataFrame({
        "game_id": ["g1", "g2", "g3"],
        "game_date": [date(2024, 9, 1) + timedelta(days=7 * index) for index in range(3)],
        "season": [2024] * 3,
        "week": [1, 2, 3],
        "player_id": ["p1"] * 3,
    })


def _ngs(current_week_three: float = 99.0) -> pl.DataFrame:
    rows = []
    for week, value in [(0, 500.0), (1, 5.0), (2, 3.0), (3, current_week_three)]:
        row = {
            "season": 2024,
            "season_type": "REG",
            "week": week,
            "player_gsis_id": "p1",
        }
        row.update({metric: value for metric in NGS_METRICS})
        rows.append(row)
    return pl.DataFrame(rows)


def test_nextgen_features_shift_week_zero_summary_and_current_game():
    result = add_nextgen_receiving_features(_games(), _ngs()).sort("week")
    assert result[0, "ngs_avg_separation_avg_last_3"] is None

    def test_nextgen_features_survive_missing_optional_pbp_context():
        row = _games().head(1)
        result = add_advanced_features(row, pl.DataFrame(), pl.DataFrame(), _ngs())
        assert "ngs_avg_separation_avg_last_3" in result.columns
        assert result[0, "ngs_avg_separation_avg_last_3"] is None
