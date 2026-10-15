"""Pregame depth-chart and team target-opportunity features."""

import polars as pl

from the_script.data.clean import canonical_team

ROLE_FEATURE_COLUMNS = [
    "depth_chart_rank",
    "depth_chart_rank_healthy_baseline",
    "depth_chart_rank_change",
    "depth_chart_available",
    "returning_team_target_share",
    "vacated_team_target_share",
    "active_teammate_target_share",
    "team_role_available",
    "games_with_current_team",
    "new_team_flag",
    "target_share_delta_last_game",
    "target_share_delta_last_3",
    "snap_share_delta_last_game",
    "snap_share_delta_last_3",
    "role_rank_target_share_delta",
]

SKILL_POSITIONS = ["QB", "WR", "TE", "RB", "FB"]


def _empty_assignments() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "season": pl.Int32,
            "week": pl.Int32,
            "team": pl.String,
            "player_id": pl.String,
            "position": pl.String,
            "depth_chart_rank": pl.Float64,
            "depth_chart_rank_healthy_baseline": pl.Float64,
        }
    )


def normalize_depth_chart_assignments(
    depth_charts: pl.DataFrame, player_games: pl.DataFrame
) -> pl.DataFrame:
    """Normalize legacy weekly and current timestamped depth-chart feeds."""
    assignments = []
    legacy_columns = {"season", "week", "club_code", "gsis_id", "position", "depth_team"}
    if legacy_columns.issubset(depth_charts.columns):
        legacy = (
            depth_charts.filter(
                pl.col("week").is_not_null()
                & pl.col("gsis_id").is_not_null()
                & pl.col("position").is_in(SKILL_POSITIONS)
                & (pl.col("game_type") == "REG")
            )
            .select(
                pl.col("season").cast(pl.Int32),
                pl.col("week").cast(pl.Int32),
                canonical_team("club_code").alias("team"),
                pl.col("gsis_id").alias("player_id"),
                "position",
                pl.col("depth_team").cast(pl.Float64, strict=False).alias("depth_chart_rank"),
                pl.col("depth_team").cast(pl.Float64, strict=False).alias(
                    "depth_chart_rank_healthy_baseline"
                ),
            )
        )
        assignments.append(legacy)

    current_columns = {"dt", "team", "gsis_id", "pos_abb", "pos_rank"}
    if current_columns.issubset(depth_charts.columns):
        current = (
            depth_charts.filter(
                pl.col("dt").is_not_null()
                & pl.col("gsis_id").is_not_null()
                & pl.col("pos_abb").is_in(SKILL_POSITIONS)
            )
            .select(
                canonical_team("team").alias("team"),
                pl.col("gsis_id").alias("player_id"),
                pl.col("pos_abb").alias("position"),
                pl.col("pos_rank").cast(pl.Float64).alias("depth_chart_rank"),
                pl.col("dt")
                .str.to_datetime("%Y-%m-%dT%H:%M:%SZ", strict=False)
                .alias("snapshot_at"),
            )
            .drop_nulls("snapshot_at")
            .sort(["team", "player_id", "position", "snapshot_at"])
            .with_columns(
                pl.col("depth_chart_rank")
                .cum_min()
                .over(["player_id", "position"])
                .alias("depth_chart_rank_healthy_baseline")
            )
        )
        games = (
            player_games.select("season", "week", "team", "game_date")
            .unique()
            .with_columns(pl.col("game_date").cast(pl.Datetime).alias("cutoff"))
        )
        snapshot_dates = current.select("team", "snapshot_at").unique()
        matched_games = games.sort("cutoff").join_asof(
            snapshot_dates.sort("snapshot_at"),
            left_on="cutoff",
            right_on="snapshot_at",
            by="team",
            strategy="backward",
            check_sortedness=False,
        )
        timestamped = (
            matched_games.drop_nulls("snapshot_at")
            .join(current, on=["team", "snapshot_at"], how="inner")
            .select(
                "season",
                "week",
                "team",
                "player_id",
                "position",
                "depth_chart_rank",
                "depth_chart_rank_healthy_baseline",
            )
        )
        assignments.append(timestamped)

    if not assignments:
        return _empty_assignments()
    return (
        pl.concat(assignments, how="vertical_relaxed")
        .group_by("season", "week", "team", "player_id", "position")
        .agg(
            pl.col("depth_chart_rank").min(),
            pl.col("depth_chart_rank_healthy_baseline").min(),
        )
    )


def _prior_target_shares(player_stats: pl.DataFrame) -> pl.DataFrame:
    required = {"season", "season_type", "team", "player_id", "targets"}
    if not required.issubset(player_stats.columns):
        return pl.DataFrame(
            schema={
                "season": pl.Int32,
                "prior_team": pl.String,
                "player_id": pl.String,
                "prior_team_target_share": pl.Float64,
            }
        )
    totals = (
        player_stats.filter(
            (pl.col("season_type") == "REG") & pl.col("player_id").is_not_null()
        )
        .with_columns(canonical_team("team").alias("team"))
        .group_by("season", "team", "player_id")
        .agg(pl.col("targets").fill_null(0).sum().alias("player_targets"))
        .with_columns(
            pl.col("player_targets").sum().over("season", "team").alias("team_targets")
        )
        .filter(pl.col("team_targets") > 0)
    )
    return totals.select(
        (pl.col("season") + 1).alias("season"),
        pl.col("team").alias("prior_team"),
        "player_id",
        (pl.col("player_targets") / pl.col("team_targets")).alias(
            "prior_team_target_share"
        ),
    )


def _with_empty_role_features(player_games: pl.DataFrame) -> pl.DataFrame:
    return player_games.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("depth_chart_rank"),
        pl.lit(None, dtype=pl.Float64).alias("depth_chart_rank_healthy_baseline"),
        pl.lit(None, dtype=pl.Float64).alias("depth_chart_rank_change"),
        pl.lit(0, dtype=pl.Int8).alias("depth_chart_available"),
        pl.lit(None, dtype=pl.Float64).alias("returning_team_target_share"),
        pl.lit(None, dtype=pl.Float64).alias("vacated_team_target_share"),
        pl.lit(None, dtype=pl.Float64).alias("active_teammate_target_share"),
        pl.lit(0, dtype=pl.Int8).alias("team_role_available"),
        pl.int_range(0, pl.len()).over("player_id", "team").alias("games_with_current_team"),
        (pl.int_range(0, pl.len()).over("player_id", "team") == 0).cast(pl.Int8).alias("new_team_flag"),
        pl.lit(None, dtype=pl.Float64).alias("target_share_delta_last_game"),
        pl.lit(None, dtype=pl.Float64).alias("target_share_delta_last_3"),
        pl.lit(None, dtype=pl.Float64).alias("snap_share_delta_last_game"),
        pl.lit(None, dtype=pl.Float64).alias("snap_share_delta_last_3"),
        pl.lit(None, dtype=pl.Float64).alias("role_rank_target_share_delta"),
    )


def add_role_features(
    player_games: pl.DataFrame,
    player_stats: pl.DataFrame,
    depth_charts: pl.DataFrame,
) -> pl.DataFrame:
    """Attach role known before each game and prior-season roster opportunity."""
    assignments = normalize_depth_chart_assignments(depth_charts, player_games)
    if assignments.is_empty():
        return _with_empty_role_features(player_games)

    wr_ranks = (
        assignments.filter(pl.col("position") == "WR")
        .select(
            "season",
            "week",
            "team",
            "player_id",
            "depth_chart_rank",
            "depth_chart_rank_healthy_baseline",
        )
        .unique(["season", "week", "team", "player_id"])
    )
    result = (
        player_games.join(
            wr_ranks,
            on=["season", "week", "team", "player_id"],
            how="left",
            validate="m:1",
        )
        .sort(["player_id", "game_date", "game_id"])
        .with_columns(
            (
                pl.col("depth_chart_rank").shift(1).over("player_id")
                - pl.col("depth_chart_rank")
            ).alias("depth_chart_rank_change"),
            pl.col("depth_chart_rank").is_not_null().cast(pl.Int8).alias(
                "depth_chart_available"
            ),
        )
    )

    result = result.with_columns(
        pl.int_range(0, pl.len()).over("player_id", "team").alias("games_with_current_team"),
        (pl.int_range(0, pl.len()).over("player_id", "team") == 0).cast(pl.Int8).alias("new_team_flag"),
    )

    if {"target_share", "snap_share"}.issubset(result.columns):
        result = result.with_columns(
            (
                pl.col("target_share").shift(1).over("player_id")
                - pl.col("target_share").shift(2).over("player_id")
            ).alias("target_share_delta_last_game"),
            (
                pl.col("target_share").shift(1).over("player_id")
                - pl.col("target_share").shift(1).rolling_mean(3, min_samples=1).over("player_id")
            ).alias("target_share_delta_last_3"),
            (
                pl.col("snap_share").shift(1).over("player_id")
                - pl.col("snap_share").shift(2).over("player_id")
            ).alias("snap_share_delta_last_game"),
            (
                pl.col("snap_share").shift(1).over("player_id")
                - pl.col("snap_share").shift(1).rolling_mean(3, min_samples=1).over("player_id")
            ).alias("snap_share_delta_last_3"),
        )
    else:
        result = result.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("target_share_delta_last_game"),
            pl.lit(None, dtype=pl.Float64).alias("target_share_delta_last_3"),
            pl.lit(None, dtype=pl.Float64).alias("snap_share_delta_last_game"),
            pl.lit(None, dtype=pl.Float64).alias("snap_share_delta_last_3"),
        )

    if "target_share" in result.columns:
        rank_baselines = (
            result.filter(pl.col("depth_chart_rank").is_not_null() & pl.col("target_share").is_not_null())
            .group_by("team", "depth_chart_rank", "game_date")
            .agg(pl.col("target_share").mean().alias("observed_role_rank_target_share"))
            .sort(["team", "depth_chart_rank", "game_date"])
            .with_columns(
                pl.col("observed_role_rank_target_share")
                .shift(1)
                .rolling_mean(window_size=16, min_samples=1)
                .over(["team", "depth_chart_rank"])
                .alias("role_rank_target_share")
            )
            .select("team", "depth_chart_rank", "game_date", "role_rank_target_share")
        )
        result = result.join(
            rank_baselines,
            on=["team", "depth_chart_rank", "game_date"],
            how="left",
            validate="m:1",
        ).with_columns(
            (pl.col("target_share").shift(1).over("player_id") - pl.col("role_rank_target_share"))
            .alias("role_rank_target_share_delta")
        ).drop("role_rank_target_share")
    else:
        result = result.with_columns(pl.lit(None, dtype=pl.Float64).alias("role_rank_target_share_delta"))

    prior_shares = _prior_target_shares(player_stats)
    roster = assignments.select("season", "week", "team", "player_id").unique()
    team_availability = prior_shares.select(
        "season", pl.col("prior_team").alias("team")
    ).unique().with_columns(pl.lit(1, dtype=pl.Int8).alias("team_role_available"))
    origin_shares = prior_shares.group_by("season", "player_id").agg(
        pl.col("prior_team_target_share").sum().alias("prior_player_target_share")
    )
    roster_opportunity = (
        roster.join(origin_shares, on=["season", "player_id"], how="left")
        .join(
            prior_shares,
            left_on=["season", "team", "player_id"],
            right_on=["season", "prior_team", "player_id"],
            how="left",
        )
        .with_columns(
            pl.col("prior_player_target_share").fill_null(0),
            pl.col("prior_team_target_share").fill_null(0),
        )
        .group_by("season", "week", "team")
        .agg(
            pl.col("prior_player_target_share").sum().alias("active_roster_target_share"),
            pl.col("prior_team_target_share").sum().alias("returning_team_target_share"),
        )
        .join(team_availability, on=["season", "team"], how="left", validate="m:1")
        .with_columns(
            pl.when(pl.col("team_role_available") == 1)
            .then((1.0 - pl.col("returning_team_target_share")).clip(0.0, 1.0))
            .alias("vacated_team_target_share"),
        )
    )
    result = result.join(
        roster_opportunity,
        on=["season", "week", "team"],
        how="left",
        validate="m:1",
    ).join(
        origin_shares,
        on=["season", "player_id"],
        how="left",
        validate="m:1",
    )
    result = result.with_columns(
        (
            pl.col("active_roster_target_share")
            - pl.col("prior_player_target_share").fill_null(0)
        )
        .clip(0.0, 1.0)
        .alias("active_teammate_target_share"),
        pl.col("team_role_available").fill_null(0),
        pl.when(pl.col("team_role_available") != 1)
        .then(None)
        .otherwise(pl.col("returning_team_target_share"))
        .alias("returning_team_target_share"),
    )
    return result.with_columns(
        pl.when(pl.col("team_role_available") != 1)
        .then(None)
        .otherwise(pl.col("active_teammate_target_share"))
        .alias("active_teammate_target_share"),
    ).drop("active_roster_target_share", "prior_player_target_share")