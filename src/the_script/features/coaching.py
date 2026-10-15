"""Pregame QB volume context from observed team and head-coach tendencies."""

import polars as pl

from the_script.data.clean import canonical_team

_METRICS = [
    "neutral_pass_rate", "trailing_pass_rate", "leading_pass_rate",
    "late_trailing_pass_rate", "plays", "no_huddle_rate", "seconds_per_play",
    "pass_over_expected",
]
QB_PLAYCALL_FEATURE_COLUMNS = [
    *[f"playcall_{metric}_avg_last_5" for metric in _METRICS],
    "coach_neutral_pass_rate_avg_last_16",
    "coach_games_before_game",
    "opponent_plays_avg_last_5",
    "underdog_neutral_pass_interaction",
]


def add_qb_playcall_features(
    frame: pl.DataFrame, play_by_play: pl.DataFrame, schedules: pl.DataFrame
) -> pl.DataFrame:
    required = {"game_id", "season", "week", "posteam", "qb_dropback",
                "rush_attempt", "score_differential"}
    if not required.issubset(play_by_play.columns) or play_by_play.is_empty():
        return frame.with_columns(*[
            pl.lit(None, dtype=pl.Float64).alias(c) for c in QB_PLAYCALL_FEATURE_COLUMNS
        ])
    pbp = play_by_play
    defaults = {"qb_kneel": 0, "qb_spike": 0, "no_huddle": None,
                "game_seconds_remaining": None, "pass_oe": None, "play_id": None}
    pbp = pbp.with_columns(*[
        pl.lit(value, dtype=pl.Float64).alias(c)
        for c, value in defaults.items() if c not in pbp.columns
    ])
    plays = pbp.filter(
        pl.col("posteam").is_not_null()
        & ((pl.col("qb_dropback") == 1) | (pl.col("rush_attempt") == 1))
        & (pl.col("qb_kneel").fill_null(0) != 1)
        & (pl.col("qb_spike").fill_null(0) != 1)
    ).with_columns(canonical_team("posteam").alias("team"))
    neutral = (pl.col("score_differential").abs() <= 8) & (pl.col("game_seconds_remaining") > 120)
    trailing = pl.col("score_differential") < 0
    leading = pl.col("score_differential") > 0
    plays = plays.sort(["game_id", "team", "play_id"]).with_columns(
        (-pl.col("game_seconds_remaining").diff().over(["game_id", "team"]))
        .alias("elapsed")
    )
    metrics = plays.group_by("game_id", "season", "week", "team").agg(
        pl.col("qb_dropback").filter(neutral).mean().alias("neutral_pass_rate"),
        pl.col("qb_dropback").filter(trailing).mean().alias("trailing_pass_rate"),
        pl.col("qb_dropback").filter(leading).mean().alias("leading_pass_rate"),
        pl.col("qb_dropback").filter(trailing & (pl.col("game_seconds_remaining") <= 900))
        .mean().alias("late_trailing_pass_rate"),
        pl.len().cast(pl.Float64).alias("plays"),
        pl.col("no_huddle").mean().alias("no_huddle_rate"),
        pl.col("elapsed").filter(pl.col("elapsed").is_between(1, 60)).mean()
        .alias("seconds_per_play"),
        pl.col("pass_oe").mean().alias("pass_over_expected"),
    ).with_columns((pl.col("season") * 100 + pl.col("week")).cast(pl.Int64).alias("_order"))
    coach_rows = []
    if {"home_coach", "away_coach", "home_team", "away_team", "game_id"}.issubset(schedules.columns):
        for side in ("home", "away"):
            coach_rows.append(schedules.select(
                "game_id", canonical_team(f"{side}_team").alias("team"),
                pl.col(f"{side}_coach").alias("_coach"),
            ))
    coaches = (pl.concat(coach_rows).unique(["game_id", "team"])
               if coach_rows else pl.DataFrame(schema={"game_id": pl.String, "team": pl.String, "_coach": pl.String}))
    metrics = metrics.join(coaches, on=["game_id", "team"], how="left", validate="1:1")
    history = metrics.sort(["team", "_order"]).with_columns(*[
        pl.col(metric).rolling_mean(5, min_samples=1).over("team")
        .alias(f"playcall_{metric}_avg_last_5") for metric in _METRICS
    ])
    coach_history = metrics.filter(pl.col("_coach").is_not_null()).sort(["team", "_coach", "_order"]).with_columns(
        pl.col("neutral_pass_rate").rolling_mean(16, min_samples=1).over(["team", "_coach"])
        .alias("coach_neutral_pass_rate_avg_last_16"),
        pl.col("game_id").cum_count().over(["team", "_coach"]).cast(pl.Float64)
        .alias("coach_games_before_game"),
    )
    targets = frame.with_columns(
        (pl.col("season") * 100 + pl.col("week")).cast(pl.Int64).alias("_order")
    ).join(coaches, on=["game_id", "team"], how="left", validate="m:1").sort("_order")
    # Strictly earlier games exclude current outcomes, including historical backtests.
    result = targets.join_asof(
        history.select("team", "_order", *QB_PLAYCALL_FEATURE_COLUMNS[:8]).sort("_order"),
        on="_order", by="team", strategy="backward", allow_exact_matches=False, check_sortedness=False,
    ).join_asof(
        coach_history.select("team", "_coach", "_order", "coach_neutral_pass_rate_avg_last_16", "coach_games_before_game").sort("_order"),
        on="_order", by=["team", "_coach"], strategy="backward", allow_exact_matches=False, check_sortedness=False,
    ).join_asof(
        history.select(pl.col("team").alias("opponent"), "_order", pl.col("playcall_plays_avg_last_5").alias("opponent_plays_avg_last_5")).sort("_order"),
        on="_order", by="opponent", strategy="backward", allow_exact_matches=False, check_sortedness=False,
    )
    margin = pl.col("team_implied_margin") if "team_implied_margin" in result.columns else pl.lit(None, dtype=pl.Float64)
    return result.with_columns(
        ((-margin).clip(0, None) * pl.col("playcall_neutral_pass_rate_avg_last_5"))
        .alias("underdog_neutral_pass_interaction")
    ).drop("_order", "_coach")
