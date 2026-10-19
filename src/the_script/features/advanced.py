"""Advanced pregame environment, quarterback, and receiver usage features."""

import polars as pl

from the_script.data.clean import canonical_team
from the_script.features.matchup import MATCHUP_FEATURE_COLUMNS, add_matchup_history_features
from the_script.features.nextgen import (
    NGS_RECEIVING_FEATURE_COLUMNS,
    add_nextgen_receiving_features,
)
from the_script.features.recency import prior_recency_mean

GAME_CONTEXT_COLUMNS = [
    "team_implied_margin",
    "game_total_line",
    "team_implied_points",
    "team_is_favorite",
    "game_is_outdoors",
    "game_temperature",
    "game_wind",
]
QB_CONTEXT_COLUMNS = [
    "qb_epa_avg_last_3",
    "qb_cpoe_avg_last_3",
    "qb_completion_rate_avg_last_3",
    "primary_qb_changed_last_game",
    "qb_scramble_rate_avg_last_5",
    "qb_pressure_rate_avg_last_5",
    "qb_checkdown_rate_avg_last_5",
    "qb_receiver_target_share_avg_last_5",
    "qb_receiver_catch_rate_avg_last_5",
    "qb_receiver_yards_per_target_avg_last_5",
]
USAGE_QUALITY_COLUMNS = [
    "pbp_target_share_avg_last_3",
    "pbp_target_share_avg_last_5",
    "deep_target_rate_avg_last_3",
    "deep_target_rate_avg_last_5",
    "red_zone_target_share_avg_last_3",
    "red_zone_target_share_avg_last_5",
    "target_share_recency_5",
    "red_zone_target_share_recency_5",
    "end_zone_targets_avg_last_5",
    "adot_avg_last_3",
    "adot_avg_last_5",
]
ADVANCED_FEATURE_COLUMNS = [
    *GAME_CONTEXT_COLUMNS,
    *QB_CONTEXT_COLUMNS,
    *USAGE_QUALITY_COLUMNS,
        *NGS_RECEIVING_FEATURE_COLUMNS,
    *MATCHUP_FEATURE_COLUMNS,
]


def add_game_context_features(
    player_games: pl.DataFrame, schedules: pl.DataFrame
) -> pl.DataFrame:
    """Attach archived market and weather context from the game schedule."""
    required = {
        "game_id",
        "home_team",
        "away_team",
        "spread_line",
        "total_line",
        "roof",
        "temp",
        "wind",
    }
    missing = sorted(required.difference(schedules.columns))
    if missing:
        raise ValueError(f"schedules is missing game context: {', '.join(missing)}")
    games = schedules.select(
        "game_id",
        canonical_team("home_team").alias("home_team"),
        canonical_team("away_team").alias("away_team"),
        "spread_line",
        "total_line",
        "roof",
        "temp",
        "wind",
    ).unique("game_id")
    result = player_games.join(games, on="game_id", how="left", validate="m:1")
    is_home = pl.col("team") == pl.col("home_team")
    implied_margin = pl.when(is_home).then(pl.col("spread_line")).otherwise(
        -pl.col("spread_line")
    )
    return result.with_columns(
        implied_margin.alias("team_implied_margin"),
        pl.col("total_line").alias("game_total_line"),
        (pl.col("total_line") / 2 + implied_margin / 2).alias("team_implied_points"),
        (implied_margin > 0).cast(pl.Int8).alias("team_is_favorite"),
        (pl.col("roof") == "outdoors").cast(pl.Int8).alias("game_is_outdoors"),
        pl.col("temp").cast(pl.Float64).alias("game_temperature"),
        pl.col("wind").cast(pl.Float64).alias("game_wind"),
    ).drop(
        "home_team",
        "away_team",
        "spread_line",
        "total_line",
        "roof",
        "temp",
        "wind",
    )


def _primary_qb_games(play_by_play: pl.DataFrame) -> pl.DataFrame:
    quarterbacks = (
        play_by_play.filter(
            pl.col("posteam").is_not_null()
            & pl.col("passer_player_id").is_not_null()
            & (pl.col("qb_dropback") == 1)
        )
        .with_columns(canonical_team("posteam").alias("team"))
        .group_by("game_id", "team", "passer_player_id")
        .agg(
            pl.len().alias("dropbacks"),
            pl.col("qb_epa").mean().alias("qb_epa"),
            pl.col("cpoe").mean().alias("qb_cpoe"),
            pl.col("complete_pass").mean().alias("qb_completion_rate"),
        )
        .sort(["game_id", "team", "dropbacks"], descending=[False, False, True])
        .unique(["game_id", "team"], keep="first")
    )
    return quarterbacks


def add_qb_context_features(
    player_games: pl.DataFrame, play_by_play: pl.DataFrame
) -> pl.DataFrame:
    """Attach efficiency and stability from completed primary-QB games only."""
    game_grid = player_games.select(
        "game_id", "game_date", "season", "week", "team"
    ).unique()
    history = (
        game_grid.join(
            _primary_qb_games(play_by_play),
            on=["game_id", "team"],
            how="left",
            validate="1:1",
        )
        .sort(["team", "game_date", "game_id"])
        .with_columns(
            pl.col("qb_epa")
            .shift(1)
            .rolling_mean(3, min_samples=1)
            .over("team")
            .alias("qb_epa_avg_last_3"),
            pl.col("qb_cpoe")
            .shift(1)
            .rolling_mean(3, min_samples=1)
            .over("team")
            .alias("qb_cpoe_avg_last_3"),
            pl.col("qb_completion_rate")
            .shift(1)
            .rolling_mean(3, min_samples=1)
            .over("team")
            .alias("qb_completion_rate_avg_last_3"),
            (
                pl.col("passer_player_id").shift(1).over("team")
                != pl.col("passer_player_id").shift(2).over("team")
            )
            .fill_null(False)
            .cast(pl.Int8)
            .alias("primary_qb_changed_last_game"),
        )
            .select("game_id", "team", *QB_CONTEXT_COLUMNS[:4])
    )
    result = player_games.join(
        history, on=["game_id", "team"], how="left", validate="m:1"
    )
    return _add_qb_receiver_context(result, play_by_play)


def _add_qb_receiver_context(
    player_games: pl.DataFrame, play_by_play: pl.DataFrame
) -> pl.DataFrame:
    """Attach shifted QB tendencies and QB-specific receiver preferences."""
    required = {
        "game_id",
        "posteam",
        "passer_player_id",
        "receiver_player_id",
        "qb_dropback",
        "qb_scramble",
        "qb_hit",
        "sack",
        "air_yards",
        "complete_pass",
        "yards_gained",
    }
    if not required.issubset(play_by_play.columns):
        return player_games.with_columns(
            *[pl.lit(None, dtype=pl.Float64).alias(column) for column in QB_CONTEXT_COLUMNS[4:]]
        )
    if "player_id" not in player_games.columns:
        return player_games.with_columns(
            *[pl.lit(None, dtype=pl.Float64).alias(column) for column in QB_CONTEXT_COLUMNS[7:]]
        )

    plays = play_by_play.filter(
        pl.col("posteam").is_not_null() & pl.col("passer_player_id").is_not_null()
    ).with_columns(canonical_team("posteam").alias("team"))
    primary = _primary_qb_games(play_by_play).select(
        "game_id", "team", pl.col("passer_player_id").alias("primary_qb_id")
    )
    game_grid = (
        player_games.select("game_id", "game_date", "season", "week", "team")
        .unique()
        .join(primary, on=["game_id", "team"], how="left", validate="1:1")
        .sort(["team", "game_date", "game_id"])
        .with_columns(
            pl.col("primary_qb_id").fill_null(strategy="forward").over("team")
        )
    )
    dropbacks = plays.group_by("game_id", "team", "passer_player_id").agg(
        pl.col("qb_dropback").sum().alias("dropbacks"),
        (pl.col("qb_scramble").fill_null(0).sum()).alias("scrambles"),
        ((pl.col("qb_hit").fill_null(0) + pl.col("sack").fill_null(0)) > 0)
        .sum()
        .alias("pressured_dropbacks"),
    )
    player_throws = (
        plays.filter(pl.col("receiver_player_id").is_not_null() & (pl.col("pass_attempt") == 1))
        .group_by("game_id", "team", "passer_player_id", "receiver_player_id")
        .agg(
            pl.len().alias("receiver_targets"),
            pl.col("complete_pass").fill_null(0).sum().alias("receiver_receptions"),
            pl.col("yards_gained").fill_null(0).sum().alias("receiver_yards"),
            (pl.col("air_yards").fill_null(0) <= 2).sum().alias("checkdowns"),
        )
        .join(dropbacks, on=["game_id", "team", "passer_player_id"], how="left")
        .with_columns(
            (pl.col("receiver_targets") / pl.col("dropbacks")).alias("receiver_target_share"),
            (pl.col("receiver_receptions") / pl.col("receiver_targets")).alias("receiver_catch_rate"),
            (pl.col("receiver_yards") / pl.col("receiver_targets")).alias("receiver_yards_per_target"),
        )
    )
    team_qb = (
        game_grid.select("game_id", "game_date", "team", "primary_qb_id")
        .join(dropbacks, left_on=["game_id", "team", "primary_qb_id"], right_on=["game_id", "team", "passer_player_id"], how="left")
        .with_columns(
            (pl.col("scrambles") / pl.col("dropbacks")).alias("qb_scramble_rate"),
            (pl.col("pressured_dropbacks") / pl.col("dropbacks")).alias("qb_pressure_rate"),
        )
        .join(
            player_throws.group_by("game_id", "team", "passer_player_id")
            .agg(
                pl.col("checkdowns").sum().alias("checkdowns"),
                pl.col("receiver_targets").sum().alias("team_receiver_targets"),
            ),
            left_on=["game_id", "team", "primary_qb_id"],
            right_on=["game_id", "team", "passer_player_id"],
            how="left",
        )
        .with_columns(
            (pl.col("checkdowns") / pl.col("team_receiver_targets")).alias("qb_checkdown_rate")
        )
        .sort(["team", "game_date", "game_id"])
        .with_columns(
            *[
                pl.col(source).shift(1).rolling_mean(5, min_samples=1).over("team").alias(name)
                for source, name in [
                    ("qb_scramble_rate", "qb_scramble_rate_avg_last_5"),
                    ("qb_pressure_rate", "qb_pressure_rate_avg_last_5"),
                    ("qb_checkdown_rate", "qb_checkdown_rate_avg_last_5"),
                ]
            ]
        )
        .select("game_id", "team", *QB_CONTEXT_COLUMNS[4:7])
    )
    receiver_history = (
        player_games.select("game_id", "game_date", "player_id", "team")
        .unique(subset=["game_id", "player_id"], keep="last")
        .join(game_grid.select("game_id", "team", "primary_qb_id"), on=["game_id", "team"], how="left")
        .join(
            player_throws,
            left_on=["game_id", "team", "primary_qb_id", "player_id"],
            right_on=["game_id", "team", "passer_player_id", "receiver_player_id"],
            how="left",
        )
        .sort(["player_id", "game_date", "game_id"])
        .with_columns(
            *[
                pl.col(source).shift(1).rolling_mean(5, min_samples=1).over("player_id").alias(name)
                for source, name in [
                    ("receiver_target_share", "qb_receiver_target_share_avg_last_5"),
                    ("receiver_catch_rate", "qb_receiver_catch_rate_avg_last_5"),
                    ("receiver_yards_per_target", "qb_receiver_yards_per_target_avg_last_5"),
                ]
            ]
        )
        .select("game_id", "player_id", *QB_CONTEXT_COLUMNS[7:])
        .unique(subset=["game_id", "player_id"], keep="last")
    )
    return (
        player_games.join(team_qb, on=["game_id", "team"], how="left", validate="m:1")
        .join(receiver_history, on=["game_id", "player_id"], how="left", validate="m:1")
    )


def add_usage_quality_features(
    player_games: pl.DataFrame, play_by_play: pl.DataFrame
) -> pl.DataFrame:
    """Attach shifted deep, red-zone, end-zone, and air-yard usage."""
    targets = play_by_play.filter(
        pl.col("receiver_player_id").is_not_null()
        & pl.col("posteam").is_not_null()
    ).with_columns(canonical_team("posteam").alias("team"))
    team_red_zone = (
        targets.filter(pl.col("yardline_100") <= 20)
        .group_by("game_id", "team")
        .agg(pl.len().alias("team_red_zone_targets"))
    )
    team_targets = targets.group_by("game_id", "team").agg(
        pl.len().alias("team_pass_targets")
    )
    player_targets = (
        targets.group_by("game_id", "team", "receiver_player_id")
        .agg(
            pl.len().alias("pbp_targets"),
            (pl.col("air_yards") >= 20).sum().alias("deep_targets"),
            (pl.col("yardline_100") <= 20).sum().alias("red_zone_targets"),
            (pl.col("air_yards") >= pl.col("yardline_100"))
            .sum()
            .alias("end_zone_targets"),
            pl.col("air_yards").mean().alias("adot"),
        )
        .join(team_red_zone, on=["game_id", "team"], how="left")
        .join(team_targets, on=["game_id", "team"], how="left")
        .with_columns(
            (pl.col("deep_targets") / pl.col("pbp_targets")).alias("deep_target_rate"),
            (pl.col("pbp_targets") / pl.col("team_pass_targets")).alias("target_share"),
            pl.when(pl.col("team_red_zone_targets") > 0)
            .then(pl.col("red_zone_targets") / pl.col("team_red_zone_targets"))
            .otherwise(0.0)
            .alias("red_zone_target_share"),
        )
    )
    history = (
        player_games.select("game_id", "game_date", "player_id", "team")
        .join(
            player_targets,
            left_on=["game_id", "team", "player_id"],
            right_on=["game_id", "team", "receiver_player_id"],
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.col("deep_target_rate").fill_null(0.0),
            pl.col("red_zone_target_share").fill_null(0.0),
            pl.col("target_share").fill_null(0.0),
            pl.col("end_zone_targets").fill_null(0),
        )
        .sort(["player_id", "game_date", "game_id"])
        .with_columns(
            *[
                pl.col(source)
                .shift(1)
                .rolling_mean(window, min_samples=1)
                .over("player_id")
                .alias(name)
                for source, window, name in [
                    ("target_share", 3, "pbp_target_share_avg_last_3"),
                    ("target_share", 5, "pbp_target_share_avg_last_5"),
                    ("deep_target_rate", 3, "deep_target_rate_avg_last_3"),
                    ("deep_target_rate", 5, "deep_target_rate_avg_last_5"),
                    ("red_zone_target_share", 3, "red_zone_target_share_avg_last_3"),
                    ("red_zone_target_share", 5, "red_zone_target_share_avg_last_5"),
                    ("end_zone_targets", 5, "end_zone_targets_avg_last_5"),
                    ("adot", 3, "adot_avg_last_3"),
                    ("adot", 5, "adot_avg_last_5"),
                ]
            ],
            prior_recency_mean("target_share", "player_id").alias("target_share_recency_5"),
            prior_recency_mean("red_zone_target_share", "player_id").alias("red_zone_target_share_recency_5"),
        )
        .select("game_id", "player_id", *USAGE_QUALITY_COLUMNS)
        .unique(subset=["game_id", "player_id"], keep="last")
    )
    return player_games.join(
        history, on=["game_id", "player_id"], how="left", validate="m:1"
    )


def add_advanced_features(
    player_games: pl.DataFrame,
    schedules: pl.DataFrame,
    play_by_play: pl.DataFrame,
    nextgen_receiving: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Attach all advanced features using one historical/live contract."""
    if schedules.is_empty() or play_by_play.is_empty():
        base = player_games.with_columns(
            *[
                pl.lit(None, dtype=pl.Float64).alias(column)
                for column in ADVANCED_FEATURE_COLUMNS
                if column not in player_games.columns
            ]
        )
        return add_nextgen_receiving_features(
            base,
            nextgen_receiving if nextgen_receiving is not None else pl.DataFrame(),
        )
    with_matchup = add_matchup_history_features(player_games, play_by_play)
    with_game = add_game_context_features(with_matchup, schedules)
    with_qb = add_qb_context_features(with_game, play_by_play)
    with_usage = add_usage_quality_features(with_qb, play_by_play)
    with_nextgen = add_nextgen_receiving_features(
        with_usage,
        nextgen_receiving if nextgen_receiving is not None else pl.DataFrame(),
    )
    return with_nextgen.with_columns(
        *[
            pl.lit(None, dtype=pl.Float64).alias(column)
            for column in [*MATCHUP_FEATURE_COLUMNS, *NGS_RECEIVING_FEATURE_COLUMNS]
            if column not in with_nextgen.columns
        ]
    )