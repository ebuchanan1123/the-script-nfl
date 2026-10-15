"""Leakage-safe team, opponent, and QB tendencies from play-by-play."""

import polars as pl

from the_script.data.clean import canonical_team

MATCHUP_FEATURE_COLUMNS = [
    "team_trailing_play_rate_avg_last_3",
    "team_trailing_play_rate_avg_last_5",
    "team_td_drive_rate_avg_last_3",
    "team_td_drive_rate_avg_last_5",
    "team_dropbacks_avg_last_3",
    "team_dropbacks_avg_last_5",
    "opponent_td_drive_rate_allowed_avg_last_3",
    "opponent_td_drive_rate_allowed_avg_last_5",
    "qb_dropbacks_avg_last_3",
    "qb_dropbacks_avg_last_5",
    "qb_fantasy_points_per_dropback_avg_last_3",
    "qb_fantasy_points_per_dropback_avg_last_5",
]


def _prior_average(column: str, window: int, group: str) -> pl.Expr:
    return (
        pl.col(column)
        .shift(1)
        .rolling_mean(window_size=window, min_samples=1)
        .over(group)
    )


def _team_game_metrics(play_by_play: pl.DataFrame) -> pl.DataFrame:
    required = {
        "game_id", "posteam", "qb_dropback", "rush_attempt", "qb_kneel", "qb_spike",
        "score_differential", "series", "series_result", "home_team", "away_team",
        "season", "week",
    }
    missing = sorted(required.difference(play_by_play.columns))
    if missing:
        raise ValueError(f"play_by_play is missing matchup inputs: {', '.join(missing)}")

    time_order = (
        play_by_play.select("game_id", "season", "week")
        .unique("game_id")
        .with_columns((pl.col("season") * 100 + pl.col("week")).alias("game_order"))
        .select("game_id", "game_order")
    )
    valid_play = (
        ((pl.col("qb_dropback") == 1) | (pl.col("rush_attempt") == 1))
        & (pl.col("qb_kneel").fill_null(0) != 1)
        & (pl.col("qb_spike").fill_null(0) != 1)
    )
    plays = (
        play_by_play.filter(pl.col("posteam").is_not_null() & valid_play)
        .with_columns(
            canonical_team("posteam").alias("team"),
            canonical_team("home_team").alias("canonical_home_team"),
            canonical_team("away_team").alias("canonical_away_team"),
        )
        .with_columns(
            pl.when(pl.col("team") == pl.col("canonical_home_team"))
            .then(pl.col("canonical_away_team"))
            .otherwise(pl.col("canonical_home_team"))
            .alias("opponent"),
            (pl.col("score_differential") < 0).cast(pl.Float64).alias("trailing_play"),
        )
    )
    play_metrics = plays.group_by("game_id", "team", "opponent").agg(
        pl.col("trailing_play").mean().alias("trailing_play_rate"),
        pl.col("qb_dropback").sum().alias("dropbacks"),
    )
    drives = (
        play_by_play.filter(
            pl.col("posteam").is_not_null()
            & pl.col("series").is_not_null()
            & pl.col("series_result").is_not_null()
        )
        .with_columns(canonical_team("posteam").alias("team"))
        .group_by("game_id", "team", "series")
        .agg(pl.col("series_result").last().alias("series_result"))
        .group_by("game_id", "team")
        .agg(
            pl.len().alias("drive_count"),
            (pl.col("series_result") == "Touchdown").sum().alias("touchdown_drive_count"),
        )
        .with_columns(
            (pl.col("touchdown_drive_count") / pl.col("drive_count")).alias("td_drive_rate")
        )
        .select("game_id", "team", "td_drive_rate")
    )
    return (
        play_metrics.join(drives, on=["game_id", "team"], how="left", validate="1:1")
        .join(time_order, on="game_id", how="left", validate="m:1")
    )


def _qb_game_metrics(play_by_play: pl.DataFrame) -> pl.DataFrame:
    required = {
        "game_id", "passer_player_id", "qb_dropback", "passing_yards", "pass_touchdown",
        "interception", "rusher_player_id", "rush_attempt", "rushing_yards", "rush_touchdown",
    }
    if not required.issubset(play_by_play.columns):
        return pl.DataFrame(
            schema={
                "game_id": pl.String,
                "player_id": pl.String,
                "qb_dropbacks": pl.Float64,
                "qb_fantasy_points_per_dropback": pl.Float64,
            }
        )
    passing = (
        play_by_play.filter(pl.col("passer_player_id").is_not_null())
        .with_columns(pl.col("passer_player_id").cast(pl.String))
        .group_by("game_id", "passer_player_id")
        .agg(
            pl.col("qb_dropback").fill_null(0).sum().alias("qb_dropbacks"),
            pl.col("passing_yards").fill_null(0).sum().alias("passing_yards"),
            pl.col("pass_touchdown").fill_null(0).sum().alias("passing_tds"),
            pl.col("interception").fill_null(0).sum().alias("interceptions"),
        )
        .rename({"passer_player_id": "player_id"})
    )
    rushing = (
        play_by_play.filter(
            pl.col("rusher_player_id").is_not_null() & (pl.col("rush_attempt") == 1)
        )
        .with_columns(pl.col("rusher_player_id").cast(pl.String))
        .group_by("game_id", "rusher_player_id")
        .agg(
            pl.col("rushing_yards").fill_null(0).sum().alias("rushing_yards"),
            pl.col("rush_touchdown").fill_null(0).sum().alias("rushing_tds"),
        )
        .rename({"rusher_player_id": "player_id"})
    )
    return (
        passing.join(rushing, on=["game_id", "player_id"], how="left", validate="1:1")
        .with_columns(pl.col("rushing_yards").fill_null(0), pl.col("rushing_tds").fill_null(0))
        .with_columns(
            (
                pl.col("passing_yards") / 25.0
                + pl.col("passing_tds") * 4.0
                - pl.col("interceptions") * 2.0
                + pl.col("rushing_yards") / 10.0
                + pl.col("rushing_tds") * 6.0
            ).alias("qb_fantasy_points")
        )
        .with_columns(
            pl.when(pl.col("qb_dropbacks") > 0)
            .then(pl.col("qb_fantasy_points") / pl.col("qb_dropbacks"))
            .otherwise(None)
            .alias("qb_fantasy_points_per_dropback")
        )
        .select(
            "game_id", "player_id", pl.col("qb_dropbacks").cast(pl.Float64),
            pl.col("qb_fantasy_points_per_dropback").cast(pl.Float64),
        )
    )


def add_matchup_history_features(
    player_games: pl.DataFrame,
    play_by_play: pl.DataFrame,
) -> pl.DataFrame:
    """Attach prior-game offense/opponent metrics and QB trends."""
    if play_by_play.is_empty():
        return player_games.with_columns(
            *[pl.lit(None, dtype=pl.Float64).alias(column) for column in MATCHUP_FEATURE_COLUMNS]
        )

    metrics = _team_game_metrics(play_by_play)
    game_order = metrics.select("game_id", "game_order").unique("game_id")
    target_team_games = pl.concat(
        [
            player_games.select(
                "game_id", pl.col("team").alias("history_team")
            ),
            player_games.select(
                "game_id", pl.col("opponent").alias("history_team")
            ),
        ],
        how="vertical_relaxed",
    ).unique()
    requested_team_rows = (
        target_team_games.join(game_order, on="game_id", how="left", validate="m:1")
        .with_columns(
            pl.lit(None, dtype=pl.Float64).alias("trailing_play_rate"),
            pl.lit(None, dtype=pl.Float64).alias("td_drive_rate"),
            pl.lit(None, dtype=pl.Float64).alias("dropbacks"),
        )
        .rename({"history_team": "team"})
    )
    team_history = (
        pl.concat(
            [
                metrics.select(
                    "game_id", "game_order", "team", "trailing_play_rate", "td_drive_rate", "dropbacks"
                ),
                requested_team_rows.select(
                    "game_id", "game_order", "team", "trailing_play_rate", "td_drive_rate", "dropbacks"
                ),
            ],
            how="vertical_relaxed",
        )
        .unique(subset=["game_id", "team"], keep="first")
        .sort(["team", "game_order", "game_id"])
        .with_columns(
            _prior_average("trailing_play_rate", 3, "team").alias("team_trailing_play_rate_avg_last_3"),
            _prior_average("trailing_play_rate", 5, "team").alias("team_trailing_play_rate_avg_last_5"),
            _prior_average("td_drive_rate", 3, "team").alias("team_td_drive_rate_avg_last_3"),
            _prior_average("td_drive_rate", 5, "team").alias("team_td_drive_rate_avg_last_5"),
            _prior_average("dropbacks", 3, "team").alias("team_dropbacks_avg_last_3"),
            _prior_average("dropbacks", 5, "team").alias("team_dropbacks_avg_last_5"),
            _prior_average("td_drive_rate", 3, "team").alias("opponent_td_drive_rate_allowed_avg_last_3"),
            _prior_average("td_drive_rate", 5, "team").alias("opponent_td_drive_rate_allowed_avg_last_5"),
        )
    )
    offense = team_history.select("game_id", pl.col("team").alias("team"), *MATCHUP_FEATURE_COLUMNS[:6])
    defense = team_history.select("game_id", pl.col("team").alias("opponent"), *MATCHUP_FEATURE_COLUMNS[6:8])

    qb_metrics = _qb_game_metrics(play_by_play)
    requested_qbs = (
        player_games.filter(pl.col("position") == "QB")
        .select("game_id", "player_id")
        .unique()
        .join(game_order, on="game_id", how="left", validate="m:1")
        .with_columns(
            pl.lit(None, dtype=pl.Float64).alias("qb_dropbacks"),
            pl.lit(None, dtype=pl.Float64).alias("qb_fantasy_points_per_dropback"),
        )
    )
    qb_history = (
        pl.concat(
            [
                qb_metrics.join(game_order, on="game_id", how="left", validate="m:1").select(
                    "game_id", "player_id", "game_order", "qb_dropbacks",
                    "qb_fantasy_points_per_dropback",
                ),
                requested_qbs.select(
                    "game_id", "player_id", "game_order", "qb_dropbacks",
                    "qb_fantasy_points_per_dropback",
                ),
            ],
            how="vertical_relaxed",
        )
        .unique(subset=["game_id", "player_id"], keep="first")
        .sort(["player_id", "game_order", "game_id"])
        .with_columns(
            _prior_average("qb_dropbacks", 3, "player_id").alias("qb_dropbacks_avg_last_3"),
            _prior_average("qb_dropbacks", 5, "player_id").alias("qb_dropbacks_avg_last_5"),
            _prior_average("qb_fantasy_points_per_dropback", 3, "player_id").alias("qb_fantasy_points_per_dropback_avg_last_3"),
            _prior_average("qb_fantasy_points_per_dropback", 5, "player_id").alias("qb_fantasy_points_per_dropback_avg_last_5"),
        )
        .select("game_id", "player_id", *MATCHUP_FEATURE_COLUMNS[8:])
    )
    return (
        player_games.join(
            offense.select(
                "game_id",
                "team",
                *[
                    column
                    for column in MATCHUP_FEATURE_COLUMNS[:6]
                    if column in offense.columns
                ],
            ),
            on=["game_id", "team"], how="left", validate="m:1",
        )
        .join(
            defense.select(
                "game_id",
                "opponent",
                *[
                    column
                    for column in MATCHUP_FEATURE_COLUMNS[6:8]
                    if column in defense.columns
                ],
            ),
            on=["game_id", "opponent"], how="left", validate="m:1",
        )
        .join(
            qb_history.select("game_id", "player_id", *MATCHUP_FEATURE_COLUMNS[8:]),
            on=["game_id", "player_id"], how="left", validate="m:1",
        )
    )
