"""Pregame injury status and teammate target-opportunity features."""

import polars as pl

from the_script.data.clean import canonical_team

INJURY_FEATURE_COLUMNS = [
    "player_questionable",
    "player_out_or_doubtful",
    "player_practice_limited",
    "player_practice_dnp",
    "teammate_questionable_target_share",
    "teammate_out_or_doubtful_target_share",
    "injury_report_available",
]


def _empty_injury_features(player_games: pl.DataFrame) -> pl.DataFrame:
    return player_games.with_columns(
        *[
            pl.lit(0.0).alias(column)
            for column in INJURY_FEATURE_COLUMNS
            if column
            not in {
                "injury_report_available",
                "teammate_questionable_target_share",
                "teammate_out_or_doubtful_target_share",
            }
        ],
        pl.lit(None, dtype=pl.Float64).alias("teammate_questionable_target_share"),
        pl.lit(None, dtype=pl.Float64).alias(
            "teammate_out_or_doubtful_target_share"
        ),
        pl.lit(0, dtype=pl.Int8).alias("injury_report_available"),
    )


def _prior_player_usage(player_stats: pl.DataFrame) -> pl.DataFrame:
    required = {"season", "week", "season_type", "player_id", "target_share"}
    missing = sorted(required.difference(player_stats.columns))
    if missing:
        raise ValueError(f"player_stats is missing injury usage inputs: {', '.join(missing)}")
    return (
        player_stats.filter(
            (pl.col("season_type") == "REG") & pl.col("player_id").is_not_null()
        )
        .sort(["player_id", "season", "week"])
        .with_columns(
            pl.col("target_share")
            .rolling_mean(window_size=5, min_samples=1)
            .over("player_id")
            .alias("prior_target_share"),
            (pl.col("season") * 100 + pl.col("week")).alias("game_order"),
        )
        .select("player_id", "game_order", "prior_target_share")
    )


def _normalized_reports(
    injuries: pl.DataFrame, player_stats: pl.DataFrame
) -> pl.DataFrame:
    required = {
        "season",
        "week",
        "team",
        "gsis_id",
        "report_status",
        "practice_status",
    }
    missing = sorted(required.difference(injuries.columns))
    if missing:
        raise ValueError(f"injuries is missing columns: {', '.join(missing)}")
    ordered = injuries.filter(
        pl.col("gsis_id").is_not_null()
        & (pl.col("game_type") == "REG")
        & pl.col("week").is_not_null()
    )
    if "date_modified" in ordered.columns:
        # Keep each player's most recent report of the week, not the worst one.
        ordered = ordered.sort("date_modified", nulls_last=False).unique(
            ["season", "week", "gsis_id"], keep="last", maintain_order=True
        )
    reports = (
        ordered.with_columns(
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            canonical_team("team").alias("team"),
            pl.col("gsis_id").alias("player_id"),
            (pl.col("report_status") == "Questionable").cast(pl.Float64).alias(
                "player_questionable"
            ),
            pl.col("report_status")
            .is_in(["Out", "Doubtful"])
            .cast(pl.Float64)
            .alias("player_out_or_doubtful"),
            pl.col("practice_status")
            .str.contains("Limited Participation", literal=True)
            .fill_null(False)
            .cast(pl.Float64)
            .alias("player_practice_limited"),
            pl.col("practice_status")
            .str.contains("Did Not Participate", literal=True)
            .fill_null(False)
            .cast(pl.Float64)
            .alias("player_practice_dnp"),
            (
                pl.col("season").cast(pl.Int32) * 100
                + pl.col("week").cast(pl.Int32)
            ).alias("report_order"),
        )
        .group_by("season", "week", "team", "player_id", "report_order")
        .agg(
            pl.col("player_questionable").max(),
            pl.col("player_out_or_doubtful").max(),
            pl.col("player_practice_limited").max(),
            pl.col("player_practice_dnp").max(),
        )
    )
    usage = _prior_player_usage(player_stats)
    return reports.sort("report_order").join_asof(
        usage.sort("game_order"),
        left_on="report_order",
        right_on="game_order",
        by="player_id",
        strategy="backward",
        allow_exact_matches=False,
        check_sortedness=False,
    ).with_columns(pl.col("prior_target_share").fill_null(0.0))


def add_injury_features(
    player_games: pl.DataFrame,
    player_stats: pl.DataFrame,
    injuries: pl.DataFrame,
) -> pl.DataFrame:
    """Attach current report status and injured-teammate prior target shares."""
    if injuries.is_empty():
        return _empty_injury_features(player_games)
    reports = _normalized_reports(injuries, player_stats)
    team_totals = reports.group_by("season", "week", "team").agg(
        (
            pl.col("prior_target_share") * pl.col("player_questionable")
        ).sum().alias("team_questionable_target_share"),
        (
            pl.col("prior_target_share") * pl.col("player_out_or_doubtful")
        ).sum().alias("team_out_target_share"),
        pl.lit(1, dtype=pl.Int8).alias("injury_report_available"),
    )
    own = reports.select(
        "season",
        "week",
        "team",
        "player_id",
        "player_questionable",
        "player_out_or_doubtful",
        "player_practice_limited",
        "player_practice_dnp",
        "prior_target_share",
    )
    result = player_games.join(
        team_totals,
        on=["season", "week", "team"],
        how="left",
        validate="m:1",
    ).join(
        own,
        on=["season", "week", "team", "player_id"],
        how="left",
        validate="m:1",
    )
    return result.with_columns(
        pl.col("player_questionable").fill_null(0.0),
        pl.col("player_out_or_doubtful").fill_null(0.0),
        pl.col("player_practice_limited").fill_null(0.0),
        pl.col("player_practice_dnp").fill_null(0.0),
        pl.when(pl.col("injury_report_available") == 1)
        .then(
            (
                pl.col("team_questionable_target_share").fill_null(0.0)
                - pl.col("prior_target_share").fill_null(0.0)
                * pl.col("player_questionable").fill_null(0.0)
            ).clip(0.0, 1.0)
        )
        .alias("teammate_questionable_target_share"),
        pl.when(pl.col("injury_report_available") == 1)
        .then(
            (
                pl.col("team_out_target_share").fill_null(0.0)
                - pl.col("prior_target_share").fill_null(0.0)
                * pl.col("player_out_or_doubtful").fill_null(0.0)
            ).clip(0.0, 1.0)
        )
        .alias("teammate_out_or_doubtful_target_share"),
        pl.col("injury_report_available").fill_null(0),
    ).drop(
        "team_questionable_target_share",
        "team_out_target_share",
        "prior_target_share",
    )

# Share of 2016-2025 QB/RB/WR/TE report entries that logged offensive snaps, by final status.
PLAY_PROBABILITY = {
    "OUT": 0.0,
    "DOUBTFUL": 0.01,
    "Questionable": 0.63,
    "Missed practice": 0.66,
    "Limited practice": 1.0,
}


def apply_availability(rows: list[dict], tags: dict[str, dict]) -> list[dict]:
    """Scale projections by play probability and re-rank; returns new rows."""
    adjusted = []
    for row in rows:
        probability = tags.get(row["player_id"], {}).get("play_probability", 1.0)
        row = dict(row)
        row["unadjusted_ppr_points"] = row["projected_ppr_points"]
        row["play_probability"] = probability
        if probability < 1.0:
            for key in ("projected_yards", "projected_receptions", "projected_touchdowns", "projected_ppr_points"):
                row[key] = row[key] * probability
            if row.get("predictions"):
                row["predictions"] = {k: v * probability for k, v in row["predictions"].items()}
        adjusted.append(row)
    return sorted(adjusted, key=lambda r: r["projected_ppr_points"], reverse=True)


def injury_risk_tags(injuries: pl.DataFrame, season: int, week: int) -> dict[str, dict]:
    """Latest report per player for one week, keeping only players with injury risk."""
    reports = injuries.filter(
        pl.col("gsis_id").is_not_null()
        & (pl.col("game_type") == "REG")
        & (pl.col("season").cast(pl.Int32) == season)
        & (pl.col("week").cast(pl.Int32) == week)
    )
    if reports.is_empty():
        return {}
    if "date_modified" in reports.columns:
        reports = reports.sort("date_modified", nulls_last=False)
    latest = reports.unique("gsis_id", keep="last", maintain_order=True)
    tags = {}
    for row in latest.iter_rows(named=True):
        status = row["report_status"]
        practice = row["practice_status"] or ""
        if status in ("Out", "Doubtful"):
            risk, label = "high", status.upper()
        elif status == "Questionable":
            risk, label = "medium", "Questionable"
        elif "Did Not Participate" in practice:
            risk, label = "medium", "Missed practice"
        elif "Limited Participation" in practice:
            risk, label = "medium", "Limited practice"
        else:
            continue
        tags[row["gsis_id"]] = {
            "risk": risk,
            "label": label,
            "play_probability": PLAY_PROBABILITY[label],
            "injury": row.get("report_primary_injury") or row.get("practice_primary_injury"),
        }
    return tags
