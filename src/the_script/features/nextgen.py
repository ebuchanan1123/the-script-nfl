"""Leakage-safe historical receiving traits from NFL Next Gen Stats."""

import polars as pl

NGS_METRICS = (
    "avg_separation",
    "avg_cushion",
    "avg_intended_air_yards",
    "percent_share_of_intended_air_yards",
    "catch_percentage",
    "avg_yac",
    "avg_expected_yac",
    "avg_yac_above_expectation",
)
NGS_RECEIVING_FEATURE_COLUMNS = [
    *(f"ngs_{metric}_avg_last_{window}" for metric in NGS_METRICS for window in (3, 5)),
    "ngs_receiving_games_observed_last_5",
]


def add_nextgen_receiving_features(
    player_games: pl.DataFrame, nextgen_receiving: pl.DataFrame
) -> pl.DataFrame:
    """Attach shifted player NGS means over prior participated games only.

    NGS publishes regular-season summary rows at week 0 and player-week rows for
    players meeting minimum attempt thresholds. Summary rows are excluded; absent
    low-volume weeks remain null and are not treated as zero performance.
    """
    if nextgen_receiving.is_empty():
        return player_games.with_columns(*[
            pl.lit(None, dtype=pl.Float64).alias(column)
            for column in NGS_RECEIVING_FEATURE_COLUMNS
            if column not in player_games.columns
        ])
    required = {"season", "season_type", "week", "player_gsis_id", *NGS_METRICS}
    missing = sorted(required.difference(nextgen_receiving.columns))
    if missing:
        raise ValueError(f"nextgen_receiving is missing columns: {', '.join(missing)}")
    ngs = (
        nextgen_receiving.filter(
            (pl.col("season_type") == "REG")
            & (pl.col("week") > 0)
            & pl.col("player_gsis_id").is_not_null()
        )
        .select(
            "season", "week", pl.col("player_gsis_id").cast(pl.String).alias("player_id"),
            *NGS_METRICS,
        )
        .group_by("season", "week", "player_id")
        .agg(*[pl.col(metric).mean().alias(metric) for metric in NGS_METRICS])
    )
    history = (
        player_games.select("game_id", "game_date", "season", "week", "player_id")
        .join(ngs, on=["season", "week", "player_id"], how="left", validate="m:1")
        .sort(["player_id", "game_date", "game_id"])
        .with_columns(*[
            pl.col(metric)
            .shift(1)
            .rolling_mean(window, min_samples=1)
            .over("player_id")
            .alias(f"ngs_{metric}_avg_last_{window}")
            for metric in NGS_METRICS
            for window in (3, 5)
        ],
        pl.col("avg_separation").is_not_null().cast(pl.Float64)
        .shift(1).rolling_sum(5, min_samples=1).over("player_id")
        .alias("ngs_receiving_games_observed_last_5"))
        .select("game_id", "player_id", *NGS_RECEIVING_FEATURE_COLUMNS)
    )
    base = player_games.drop(*[
        column for column in NGS_RECEIVING_FEATURE_COLUMNS if column in player_games.columns
    ])
    return base.join(history, on=["game_id", "player_id"], how="left", validate="1:1")
