"""Leakage-safe offensive play-calling and WR allocation tendencies."""

import polars as pl

from the_script.data.clean import canonical_team

TENDENCY_FEATURE_COLUMNS = [
    "offense_pass_rate_avg_last_3",
    "offense_pass_rate_avg_last_5",
    "offense_pass_rate_season_to_date",
    "neutral_pass_rate_avg_last_3",
    "neutral_pass_rate_avg_last_5",
    "team_wr_target_share_avg_last_3",
    "team_wr_target_share_avg_last_5",
    "team_wr_target_share_season_to_date",
    "team_wr_target_concentration_avg_last_5",
]


def _prior_average(column: str, window: int, group: str | list[str]) -> pl.Expr:
    return (
        pl.col(column)
        .shift(1)
        .rolling_mean(window_size=window, min_samples=1)
        .over(group)
    )


def _team_target_allocation(player_stats: pl.DataFrame) -> pl.DataFrame:
    required = {"game_id", "season_type", "team", "position", "targets"}
    missing = sorted(required.difference(player_stats.columns))
    if missing:
        raise ValueError(f"player_stats is missing tendency inputs: {', '.join(missing)}")
    return (
        player_stats.filter(pl.col("season_type") == "REG")
        .with_columns(
            canonical_team("team").alias("team"),
            pl.col("targets").fill_null(0).cast(pl.Float64).alias("targets"),
        )
        .with_columns(
            pl.when(pl.col("position") == "WR")
            .then(pl.col("targets"))
            .otherwise(0.0)
            .alias("wr_targets"),
            pl.when(pl.col("position") == "WR")
            .then(pl.col("targets") ** 2)
            .otherwise(0.0)
            .alias("wr_targets_squared"),
        )
        .group_by("game_id", "team")
        .agg(
            pl.col("targets").sum().alias("team_targets"),
            pl.col("wr_targets").sum().alias("team_wr_targets"),
            pl.col("wr_targets_squared").sum().alias("wr_targets_squared"),
        )
        .with_columns(
            pl.when(pl.col("team_targets") > 0)
            .then(pl.col("team_wr_targets") / pl.col("team_targets"))
            .alias("team_wr_target_share"),
            pl.when(pl.col("team_wr_targets") > 0)
            .then(pl.col("wr_targets_squared") / pl.col("team_wr_targets") ** 2)
            .alias("team_wr_target_concentration"),
        )
        .select(
            "game_id",
            "team",
            "team_wr_target_share",
            "team_wr_target_concentration",
        )
    )


def _team_pass_rates(play_by_play: pl.DataFrame) -> pl.DataFrame:
    required = {
        "game_id",
        "posteam",
        "qb_dropback",
        "rush_attempt",
        "qb_kneel",
        "qb_spike",
        "score_differential",
        "wp",
    }
    missing = sorted(required.difference(play_by_play.columns))
    if missing:
        raise ValueError(f"play_by_play is missing tendency inputs: {', '.join(missing)}")
    valid_play = (
        ((pl.col("qb_dropback") == 1) | (pl.col("rush_attempt") == 1))
        & (pl.col("qb_kneel").fill_null(0) != 1)
        & (pl.col("qb_spike").fill_null(0) != 1)
    )
    neutral_play = (
        pl.col("score_differential").is_between(-8, 8)
        & pl.col("wp").is_between(0.2, 0.8)
    )
    return (
        play_by_play.filter(pl.col("posteam").is_not_null() & valid_play)
        .with_columns(canonical_team("posteam").alias("team"))
        .group_by("game_id", "team")
        .agg(
            pl.col("qb_dropback").mean().alias("offense_pass_rate"),
            pl.col("qb_dropback")
            .filter(neutral_play)
            .mean()
            .alias("neutral_pass_rate"),
        )
    )


def add_team_tendency_features(
    player_games: pl.DataFrame,
    player_stats: pl.DataFrame,
    play_by_play: pl.DataFrame,
) -> pl.DataFrame:
    """Attach team tendencies known strictly before each player-game."""
    if player_stats.is_empty() or play_by_play.is_empty():
        return player_games.with_columns(
            *[
                pl.lit(None, dtype=pl.Float64).alias(column)
                for column in TENDENCY_FEATURE_COLUMNS
            ]
        )
    game_grid = player_games.select(
        "game_id", "game_date", "season", "week", "team"
    ).unique()
    history = (
        game_grid.join(
            _team_target_allocation(player_stats),
            on=["game_id", "team"],
            how="left",
            validate="1:1",
        )
        .join(
            _team_pass_rates(play_by_play),
            on=["game_id", "team"],
            how="left",
            validate="1:1",
        )
        .sort(["team", "game_date", "game_id"])
    )
    season_group = ["team", "season"]
    history = history.with_columns(
        _prior_average("offense_pass_rate", 3, "team").alias(
            "offense_pass_rate_avg_last_3"
        ),
        _prior_average("offense_pass_rate", 5, "team").alias(
            "offense_pass_rate_avg_last_5"
        ),
        _prior_average("offense_pass_rate", 100, season_group).alias(
            "offense_pass_rate_season_to_date"
        ),
        _prior_average("neutral_pass_rate", 3, "team").alias(
            "neutral_pass_rate_avg_last_3"
        ),
        _prior_average("neutral_pass_rate", 5, "team").alias(
            "neutral_pass_rate_avg_last_5"
        ),
        _prior_average("team_wr_target_share", 3, "team").alias(
            "team_wr_target_share_avg_last_3"
        ),
        _prior_average("team_wr_target_share", 5, "team").alias(
            "team_wr_target_share_avg_last_5"
        ),
        _prior_average("team_wr_target_share", 100, season_group).alias(
            "team_wr_target_share_season_to_date"
        ),
        _prior_average("team_wr_target_concentration", 5, "team").alias(
            "team_wr_target_concentration_avg_last_5"
        ),
    ).select("game_id", "team", *TENDENCY_FEATURE_COLUMNS)
    return player_games.join(
        history, on=["game_id", "team"], how="left", validate="m:1"
    )