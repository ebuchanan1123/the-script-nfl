"""Small, strictly prior-game recency-weighted feature helpers."""

import polars as pl

RECENCY_DECAY = 0.75
RECENCY_GAMES = 5
EXPERIMENTAL_RECENCY_FEATURE_COLUMNS = (
    "targets_recency_5",
    "target_share_recency_5",
    "red_zone_target_share_recency_5",
    "pbp_target_share_avg_last_3",
    "pbp_target_share_avg_last_5",
)


def prior_recency_mean(column: str, group: str | list[str]) -> pl.Expr:
    """Exponential average over the five prior observations, newest weighted most."""
    weights = [RECENCY_DECAY**lag for lag in range(RECENCY_GAMES)]
    values = [pl.col(column).shift(lag + 1).over(group) for lag in range(RECENCY_GAMES)]
    numerator = pl.sum_horizontal([
        value.fill_null(0.0) * weight for value, weight in zip(values, weights, strict=True)
    ])
    denominator = pl.sum_horizontal([
        value.is_not_null().cast(pl.Float64) * weight
        for value, weight in zip(values, weights, strict=True)
    ])
    return pl.when(denominator > 0).then(numerator / denominator).otherwise(None)