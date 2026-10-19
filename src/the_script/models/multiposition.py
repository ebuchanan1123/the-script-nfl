"""Leakage-safe QB, RB, and TE multi-output training pipeline."""

import json
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from the_script.data import live as nfl
from the_script.data.clean import (
    canonical_team,
    canonical_team_code,
    clean_schedules,
    clean_snap_counts,
)
from the_script.evaluation.backtest import chronological_splits
from the_script.evaluation.metrics import regression_metrics
from the_script.features.advanced import (
    ADVANCED_FEATURE_COLUMNS,
    add_game_context_features,
    add_qb_context_features,
    add_usage_quality_features,
)
from the_script.features.coaching import QB_PLAYCALL_FEATURE_COLUMNS, add_qb_playcall_features
from the_script.features.injuries import INJURY_FEATURE_COLUMNS, add_injury_features
from the_script.features.matchup import add_matchup_history_features
from the_script.features.nextgen import (
    NGS_RECEIVING_FEATURE_COLUMNS,
    add_nextgen_receiving_features,
)
from the_script.features.recency import (
    EXPERIMENTAL_RECENCY_FEATURE_COLUMNS,
    prior_recency_mean,
)
from the_script.features.role import normalize_depth_chart_assignments
from the_script.models.preprocessing import NumericPreprocessor

IDENTIFIER_COLUMNS = [
    "game_id",
    "game_date",
    "season",
    "week",
    "player_id",
    "team",
    "opponent",
    "home_game",
    "days_rest",
    "position",
]


@dataclass(frozen=True)
class PositionModelConfig:
    position: str
    outcomes: tuple[str, ...]
    opportunity: pl.Expr


POSITION_CONFIGS = {
    "QB": PositionModelConfig(
        position="QB",
        outcomes=(
            "attempts",
            "passing_yards",
            "passing_tds",
            "passing_interceptions",
            "carries",
            "rushing_yards",
            "rushing_tds",
        ),
        opportunity=(pl.col("attempts") + pl.col("carries")) > 0,
    ),
    "RB": PositionModelConfig(
        position="RB",
        outcomes=(
            "carries",
            "rushing_yards",
            "rushing_tds",
            "targets",
            "receptions",
            "receiving_yards",
            "receiving_tds",
        ),
        opportunity=(pl.col("carries") + pl.col("targets")) > 0,
    ),
    "TE": PositionModelConfig(
        position="TE",
        outcomes=("targets", "receptions", "receiving_yards", "receiving_tds"),
        opportunity=pl.col("targets") > 0,
    ),
}
PRIMARY_TARGETS = {
    "QB": "passing_yards",
    "RB": "rushing_yards",
    "TE": "receiving_yards",
}
QUANTILES = {"q10": 0.1, "q50": 0.5, "q90": 0.9}

TEAM_TENDENCY_COLUMNS = [
    "team_pass_rate_avg_last_3",
    "team_pass_rate_avg_last_5",
    "team_rush_rate_avg_last_3",
    "team_rush_rate_avg_last_5",
    "team_red_zone_pass_rate_avg_last_3",
    "team_red_zone_pass_rate_avg_last_5",
    "team_red_zone_rush_rate_avg_last_3",
    "team_red_zone_rush_rate_avg_last_5",
]
RB_SHARE_COLUMNS = [
    "rb_carry_share_last_game",
    "rb_carry_share_avg_last_3",
    "rb_carry_share_avg_last_5",
    "rb_carry_share_avg_last_16",
]
ROLE_RAMP_COLUMNS = [
    "games_with_current_team",
    "new_team_flag",
    "snap_share_delta_last_game",
    "snap_share_delta_last_3",
]
RB_RAMP_COLUMNS = [
    "carry_share_delta_last_game",
    "carry_share_delta_last_3",
    "target_share_delta_last_game",
    "target_share_delta_last_3",
]


def _rolling_feature_names(config: PositionModelConfig) -> list[str]:
    return [
        f"{outcome}_avg_last_{window}"
        for outcome in config.outcomes
        for window in (3, 5, 16)
    ]


def position_feature_columns(position: str) -> list[str]:
    config = POSITION_CONFIGS[position]
    advanced_columns = [
        column
        for column in ADVANCED_FEATURE_COLUMNS
        if column not in set(NGS_RECEIVING_FEATURE_COLUMNS)
        and column not in EXPERIMENTAL_RECENCY_FEATURE_COLUMNS
    ]
    opponent_columns = [
        f"opponent_{outcome}_allowed_avg_last_{window}"
        for outcome in config.outcomes
        for window in (3, 5)
    ]
    return [
        *_rolling_feature_names(config),
        "snap_share_last_game",
        "snap_share_avg_last_3",
        "depth_chart_rank",
            "depth_chart_rank_healthy_baseline",
        *ROLE_RAMP_COLUMNS,
        "games_played_before_game",
        "home_game",
        "days_rest",
        *advanced_columns,
        *INJURY_FEATURE_COLUMNS,
        *TEAM_TENDENCY_COLUMNS,
        *opponent_columns,
        *(RB_SHARE_COLUMNS if position == "RB" else []),
        *(RB_RAMP_COLUMNS if position == "RB" else []),
        *(QB_PLAYCALL_FEATURE_COLUMNS if position == "QB" else []),
    ]


def build_position_player_games(
    player_stats: pl.DataFrame,
    schedules: pl.DataFrame,
    snap_counts: pl.DataFrame,
    player_registry: pl.DataFrame,
    position: str,
    min_season: int = 2016,
    max_season: int = 2025,
) -> pl.DataFrame:
    """Create one opportunity-bearing, snap-verified row per player-game."""
    config = POSITION_CONFIGS[position]
    required = {
        "game_id",
        "player_id",
        "position",
        "season",
        "week",
        "season_type",
        "team",
        "opponent_team",
        *config.outcomes,
    }
    missing = sorted(required.difference(player_stats.columns))
    if missing:
        raise ValueError(f"player_stats is missing {position} columns: {', '.join(missing)}")
    stats = (
        player_stats.filter(
            (pl.col("season_type") == "REG")
            & (pl.col("position") == position)
            & pl.col("season").is_between(min_season, max_season)
            & config.opportunity
            & pl.col("player_id").is_not_null()
        )
        .with_columns(
            canonical_team("team").alias("team"),
            canonical_team("opponent_team").alias("source_opponent"),
        )
        .unique(["game_id", "player_id"], keep="none")
    )
    games = clean_schedules(schedules, min_season, max_season).select(
        "game_id",
        "game_date",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_rest",
        "away_rest",
    )
    participation = clean_snap_counts(snap_counts, player_registry)
    joined = stats.join(
        games, on=["game_id", "season", "week"], how="inner", validate="m:1"
    ).join(participation, on=["game_id", "player_id"], how="inner", validate="1:1")
    is_home = pl.col("team") == pl.col("home_team")
    result = joined.with_columns(
        is_home.alias("home_game"),
        pl.when(is_home)
        .then(pl.col("away_team"))
        .otherwise(pl.col("home_team"))
        .alias("opponent"),
        pl.when(is_home)
        .then(pl.col("home_rest"))
        .otherwise(pl.col("away_rest"))
        .alias("days_rest"),
    ).select(*IDENTIFIER_COLUMNS, *config.outcomes, "snap_share")
    if result.select(pl.struct(["game_id", "player_id"]).n_unique()).item() != result.height:
        raise ValueError(f"Duplicate {position} player-game rows")
    return result.sort(["game_date", "game_id", "player_id"])


def add_position_form_features(frame: pl.DataFrame, position: str) -> pl.DataFrame:
    """Add shifted 3/5/16-game form without current-game outcomes."""
    config = POSITION_CONFIGS[position]
    ordered = frame.sort(["player_id", "game_date", "game_id"])
    rolling = [
        pl.col(outcome)
        .shift(1)
        .rolling_mean(window_size=window, min_samples=1)
        .over("player_id")
        .alias(f"{outcome}_avg_last_{window}")
        for outcome in config.outcomes
        for window in (3, 5, 16)
    ]
    return ordered.with_columns(
        *rolling,
        *(
            [prior_recency_mean("targets", "player_id").alias("targets_recency_5")]
            if position != "QB"
            else []
        ),
        pl.col("snap_share").shift(1).over("player_id").alias("snap_share_last_game"),
        pl.col("snap_share")
        .shift(1)
        .rolling_mean(3, min_samples=1)
        .over("player_id")
        .alias("snap_share_avg_last_3"),
        pl.int_range(0, pl.len()).over("player_id").alias("games_played_before_game"),
        pl.int_range(0, pl.len()).over("player_id", "team").alias("games_with_current_team"),
        (pl.int_range(0, pl.len()).over("player_id", "team") == 0).cast(pl.Int8).alias("new_team_flag"),
        (pl.col("snap_share").shift(1).over("player_id") - pl.col("snap_share").shift(2).over("player_id")).alias("snap_share_delta_last_game"),
        (pl.col("snap_share").shift(1).over("player_id") - pl.col("snap_share").shift(1).rolling_mean(3, min_samples=1).over("player_id")).alias("snap_share_delta_last_3"),
    )


def _prior_average(column: str, window: int, group: str | list[str]) -> pl.Expr:
    return (
        pl.col(column)
        .shift(1)
        .rolling_mean(window_size=window, min_samples=1)
        .over(group)
    )


def _add_depth_rank(frame: pl.DataFrame, depth_charts: pl.DataFrame, position: str) -> pl.DataFrame:
    if depth_charts.is_empty():
        return frame.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("depth_chart_rank"),
            pl.lit(None, dtype=pl.Float64).alias("depth_chart_rank_healthy_baseline"),
        )
    assignments = normalize_depth_chart_assignments(depth_charts, frame)
    if assignments.is_empty():
        return frame.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("depth_chart_rank"),
            pl.lit(None, dtype=pl.Float64).alias("depth_chart_rank_healthy_baseline"),
        )
    ranks = (
        assignments.filter(pl.col("position") == position)
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
    return frame.join(
        ranks, on=["season", "week", "team", "player_id"], how="left", validate="m:1"
    )


def _add_team_tendencies(frame: pl.DataFrame, play_by_play: pl.DataFrame) -> pl.DataFrame:
    if play_by_play.is_empty():
        return frame.with_columns(
            *[pl.lit(None, dtype=pl.Float64).alias(column) for column in TEAM_TENDENCY_COLUMNS]
        )
    valid_play = (
        ((pl.col("qb_dropback") == 1) | (pl.col("rush_attempt") == 1))
        & (pl.col("qb_kneel").fill_null(0) != 1)
        & (pl.col("qb_spike").fill_null(0) != 1)
    )
    red_zone = pl.col("yardline_100") <= 20
    game_rates = (
        play_by_play.filter(pl.col("posteam").is_not_null() & valid_play)
        .with_columns(canonical_team("posteam").alias("team"))
        .group_by("game_id", "team")
        .agg(
            pl.col("qb_dropback").mean().alias("team_pass_rate"),
            pl.col("rush_attempt").mean().alias("team_rush_rate"),
            pl.col("qb_dropback").filter(red_zone).mean().alias("team_red_zone_pass_rate"),
            pl.col("rush_attempt").filter(red_zone).mean().alias("team_red_zone_rush_rate"),
        )
    )
    history = (
        frame.select("game_id", "game_date", "season", "week", "team").unique()
        .join(game_rates, on=["game_id", "team"], how="left", validate="1:1")
        .sort(["team", "game_date", "game_id"])
        .with_columns(
            _prior_average("team_pass_rate", 3, "team").alias("team_pass_rate_avg_last_3"),
            _prior_average("team_pass_rate", 5, "team").alias("team_pass_rate_avg_last_5"),
            _prior_average("team_rush_rate", 3, "team").alias("team_rush_rate_avg_last_3"),
            _prior_average("team_rush_rate", 5, "team").alias("team_rush_rate_avg_last_5"),
            _prior_average("team_red_zone_pass_rate", 3, "team").alias("team_red_zone_pass_rate_avg_last_3"),
            _prior_average("team_red_zone_pass_rate", 5, "team").alias("team_red_zone_pass_rate_avg_last_5"),
            _prior_average("team_red_zone_rush_rate", 3, "team").alias("team_red_zone_rush_rate_avg_last_3"),
            _prior_average("team_red_zone_rush_rate", 5, "team").alias("team_red_zone_rush_rate_avg_last_5"),
        )
        .select("game_id", "team", *TEAM_TENDENCY_COLUMNS)
    )
    return frame.join(history, on=["game_id", "team"], how="left", validate="m:1")


def _add_opponent_allowed(frame: pl.DataFrame, position: str) -> pl.DataFrame:
    config = POSITION_CONFIGS[position]
    game_totals = (
        frame.group_by("opponent", "game_id", "game_date", "season", "week")
        .agg(*[pl.col(outcome).sum().alias(f"{outcome}_allowed") for outcome in config.outcomes])
        .sort(["opponent", "game_date", "game_id"])
    )
    history = game_totals.with_columns(
        *[
            _prior_average(f"{outcome}_allowed", window, "opponent").alias(
                f"opponent_{outcome}_allowed_avg_last_{window}"
            )
            for outcome in config.outcomes
            for window in (3, 5)
        ]
    ).select(
        "game_id",
        "opponent",
        *[
            f"opponent_{outcome}_allowed_avg_last_{window}"
            for outcome in config.outcomes
            for window in (3, 5)
        ],
    )
    return frame.join(history, on=["game_id", "opponent"], how="left", validate="m:1")


def _add_rb_carry_share(frame: pl.DataFrame, position: str) -> pl.DataFrame:
    if position != "RB":
        return frame
    team_totals = frame.group_by("game_id", "team").agg(
        pl.col("carries").sum().alias("team_rb_carries"),
        pl.col("targets").sum().alias("team_rb_targets"),
    )
    return (
        frame.join(team_totals, on=["game_id", "team"], how="left", validate="m:1")
        .with_columns(
            pl.when(pl.col("team_rb_carries") > 0)
            .then(pl.col("carries") / pl.col("team_rb_carries"))
            .alias("rb_carry_share"),
            pl.when(pl.col("team_rb_targets") > 0)
            .then(pl.col("targets") / pl.col("team_rb_targets"))
            .alias("rb_target_share"),
        )
        .sort(["player_id", "game_date", "game_id"])
        .with_columns(
            pl.col("rb_carry_share").shift(1).over("player_id").alias("rb_carry_share_last_game"),
            _prior_average("rb_carry_share", 3, "player_id").alias("rb_carry_share_avg_last_3"),
            _prior_average("rb_carry_share", 5, "player_id").alias("rb_carry_share_avg_last_5"),
            _prior_average("rb_carry_share", 16, "player_id").alias("rb_carry_share_avg_last_16"),
            (pl.col("rb_carry_share").shift(1).over("player_id") - pl.col("rb_carry_share").shift(2).over("player_id")).alias("carry_share_delta_last_game"),
            (pl.col("rb_carry_share").shift(1).over("player_id") - pl.col("rb_carry_share").shift(1).rolling_mean(3, min_samples=1).over("player_id")).alias("carry_share_delta_last_3"),
            (pl.col("rb_target_share").shift(1).over("player_id") - pl.col("rb_target_share").shift(2).over("player_id")).alias("target_share_delta_last_game"),
            (pl.col("rb_target_share").shift(1).over("player_id") - pl.col("rb_target_share").shift(1).rolling_mean(3, min_samples=1).over("player_id")).alias("target_share_delta_last_3"),
        )
        .drop("team_rb_carries", "team_rb_targets", "rb_carry_share", "rb_target_share")
    )


def add_position_context_features(
    frame: pl.DataFrame,
    position: str,
    schedules: pl.DataFrame,
    play_by_play: pl.DataFrame,
    depth_charts: pl.DataFrame,
    player_stats: pl.DataFrame | None = None,
    injuries: pl.DataFrame | None = None,
    nextgen_receiving: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Attach pregame role, environment, usage, injury, and opponent context."""
    with_game = add_game_context_features(frame, schedules) if not schedules.is_empty() else frame
    with_depth = _add_depth_rank(with_game, depth_charts, position)
    with_team = _add_team_tendencies(with_depth, play_by_play)
    with_matchup = add_matchup_history_features(with_team, play_by_play)
    with_advanced = with_matchup
    advanced_columns = {
        "qb_dropback",
        "passer_player_id",
        "qb_epa",
        "cpoe",
        "complete_pass",
        "receiver_player_id",
        "air_yards",
        "yardline_100",
    }
    if advanced_columns.issubset(play_by_play.columns):
        with_advanced = add_qb_context_features(with_advanced, play_by_play)
        with_advanced = add_usage_quality_features(with_advanced, play_by_play)
    else:
        with_advanced = with_advanced.with_columns(
            *[
                pl.lit(None, dtype=pl.Float64).alias(column)
                for column in ADVANCED_FEATURE_COLUMNS
                if column not in with_advanced.columns
            ]
        )
    if position in {"WR", "TE"}:
        with_advanced = add_nextgen_receiving_features(
            with_advanced,
            nextgen_receiving if nextgen_receiving is not None else pl.DataFrame(),
        )
    with_opp = _add_opponent_allowed(with_advanced, position)
    with_role = _add_rb_carry_share(with_opp, position)
    if position == "QB":
        with_role = add_qb_playcall_features(with_role, play_by_play, schedules)
    if player_stats is not None and injuries is not None:
        injury_columns = {"season", "week", "team", "player_id"}
        if injury_columns.issubset(player_stats.columns) and not injuries.is_empty():
            return add_injury_features(with_role, player_stats, injuries)
    return with_role.with_columns(
        *[
            pl.lit(None, dtype=pl.Float64).alias(column)
            for column in INJURY_FEATURE_COLUMNS
            if column not in with_role.columns
        ]
    )


def build_position_features_from_parquet(
    raw_directory: Path, output_directory: Path, position: str
) -> pl.DataFrame:
    schedules = pl.read_parquet(raw_directory / "schedules.parquet")
    frame = add_position_context_features(
        add_position_form_features(
        build_position_player_games(
            pl.read_parquet(raw_directory / "player_stats.parquet"),
            schedules,
            pl.read_parquet(raw_directory / "snap_counts.parquet"),
            pl.read_parquet(raw_directory / "players.parquet"),
            position,
        ),
        position,
        ),
        position,
        schedules,
        pl.read_parquet(raw_directory / "play_by_play.parquet"),
        pl.read_parquet(raw_directory / "depth_charts.parquet"),
        pl.read_parquet(raw_directory / "player_stats.parquet"),
        pl.read_parquet(raw_directory / "injuries.parquet"),
        pl.read_parquet(raw_directory / "nextgen_receiving.parquet")
        if (raw_directory / "nextgen_receiving.parquet").exists()
        else pl.DataFrame(),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_directory / f"{position.lower()}_player_features.parquet")
    return frame


def _holdout_metrics(
    split: pl.DataFrame, target: str, predictions: np.ndarray
) -> dict[str, float | int]:
    return {
        "rows": split.height,
        **regression_metrics(split[target], pl.Series(predictions)),
    }


def _train_ridge(
    splits: dict[str, pl.DataFrame], features: list[str], target: str, directory: Path
) -> dict[str, object]:
    preprocessor = NumericPreprocessor.fit(splits["train"], features)
    model = Ridge(alpha=10.0)
    model.fit(preprocessor.transform(splits["train"]), splits["train"][target].to_numpy())
    result: dict[str, object] = {"model": "ridge", "target": target, "features": features}
    for split_name in ("validation", "test"):
        predictions = np.clip(model.predict(preprocessor.transform(splits[split_name])), 0.0, None)
        result[split_name] = _holdout_metrics(splits[split_name], target, predictions)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "model.joblib")
    joblib.dump(preprocessor, directory / "preprocessor.joblib")
    (directory / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def _train_xgboost(
    splits: dict[str, pl.DataFrame], features: list[str], target: str, directory: Path
) -> dict[str, object]:
    from xgboost import XGBRegressor

    preprocessor = NumericPreprocessor.fit(splits["train"], features)
    model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(preprocessor.transform(splits["train"]), splits["train"][target].to_numpy())
    result: dict[str, object] = {"model": "xgboost", "target": target, "features": features}
    for split_name in ("validation", "test"):
        predictions = np.clip(model.predict(preprocessor.transform(splits[split_name])), 0.0, None)
        result[split_name] = _holdout_metrics(splits[split_name], target, predictions)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "model.joblib")
    joblib.dump(preprocessor, directory / "preprocessor.joblib")
    (directory / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def _train_neural(
    splits: dict[str, pl.DataFrame], features: list[str], target: str, directory: Path
) -> dict[str, object]:
    from sklearn.neural_network import MLPRegressor

    preprocessor = NumericPreprocessor.fit(splits["train"], features)
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        random_state=42,
    )
    model.fit(preprocessor.transform(splits["train"]), splits["train"][target].to_numpy())
    result: dict[str, object] = {"model": "neural_mlp", "target": target, "features": features}
    for split_name in ("validation", "test"):
        predictions = np.clip(model.predict(preprocessor.transform(splits[split_name])), 0.0, None)
        result[split_name] = _holdout_metrics(splits[split_name], target, predictions)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "model.joblib")
    joblib.dump(preprocessor, directory / "preprocessor.joblib")
    (directory / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def train_position_models(
    frame: pl.DataFrame, models_directory: Path, position: str
) -> dict[str, object]:
    """Train the three model families and validation-weight their predictions."""
    config = POSITION_CONFIGS[position]
    features = position_feature_columns(position)
    splits = chronological_splits(frame)
    results: dict[str, object] = {"position": position, "outcomes": {}}
    for target in config.outcomes:
        target_directory = models_directory / position.lower() / target
        ridge = _train_ridge(splits, features, target, target_directory / "ridge")
        xgboost = _train_xgboost(splits, features, target, target_directory / "xgboost")
        neural = _train_neural(splits, features, target, target_directory / "neural_mlp")
        models = {"ridge": ridge, "xgboost": xgboost, "neural_mlp": neural}
        inverse_errors = {
            name: 1.0 / max(float(result["validation"]["mae"]), 0.001)
            for name, result in models.items()
        }
        total_weight = sum(inverse_errors.values())
        weights = {name: value / total_weight for name, value in inverse_errors.items()}
        blend_metrics = {}
        for split_name in ("validation", "test"):
            blend_predictions = np.zeros(splits[split_name].height)
            for family, weight in weights.items():
                directory = target_directory / family
                preprocessor = joblib.load(directory / "preprocessor.joblib")
                model = joblib.load(directory / "model.joblib")
                blend_predictions += weight * np.clip(
                    model.predict(preprocessor.transform(splits[split_name])), 0.0, None
                )
            blend_metrics[split_name] = _holdout_metrics(
                splits[split_name], target, blend_predictions
            )
        results["outcomes"][target] = {
            "selected_model": "weighted_blend",
            "weights": weights,
            "weighted_blend": blend_metrics,
            "models": models,
        }
    output = models_directory / position.lower() / "comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    return results


def train_position_quantiles(
    frame: pl.DataFrame, models_directory: Path, position: str
) -> dict[str, object]:
    """Train 10th/50th/90th percentile models for each position's primary stat."""
    position = position.upper()
    target = PRIMARY_TARGETS[position]
    features = position_feature_columns(position)
    splits = chronological_splits(frame)
    output_directory = models_directory / position.lower() / "quantiles"
    results: dict[str, object] = {"position": position, "target": target, "models": {}}
    predictions: dict[str, dict[str, np.ndarray]] = {"validation": {}, "test": {}}
    for name, alpha in QUANTILES.items():
        preprocessor = NumericPreprocessor.fit(splits["train"], features)
        model = XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:quantileerror",
            quantile_alpha=alpha,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(preprocessor.transform(splits["train"]), splits["train"][target].to_numpy())
        directory = output_directory / name
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, directory / "model.joblib")
        joblib.dump(preprocessor, directory / "preprocessor.joblib")
        results["models"][name] = {"alpha": alpha, "features": features}
        for split_name, split_predictions in predictions.items():
            split_predictions[name] = np.clip(
                model.predict(preprocessor.transform(splits[split_name])), 0.0, None
            )
    for split_name, split_predictions in predictions.items():
        ordered = np.sort(np.column_stack([split_predictions[name] for name in QUANTILES]), axis=1)
        actual = splits[split_name][target].to_numpy()
        results[split_name] = {
            "rows": splits[split_name].height,
            "interval_coverage": float(((actual >= ordered[:, 0]) & (actual <= ordered[:, 2])).mean()),
            "mean_interval_width": float(np.mean(ordered[:, 2] - ordered[:, 0])),
        }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


def train_position_quantiles_from_parquet(
    input_directory: Path, models_directory: Path, position: str
) -> dict[str, object]:
    return train_position_quantiles(
        pl.read_parquet(input_directory / f"{position.lower()}_player_features.parquet"),
        models_directory,
        position,
    )


def _predict_position_quantiles(
    features: pl.DataFrame, models_directory: Path, position: str
) -> dict[str, float]:
    directory = models_directory / position.lower() / "quantiles"
    values = []
    for name in QUANTILES:
        model_directory = directory / name
        preprocessor = joblib.load(model_directory / "preprocessor.joblib")
        model = joblib.load(model_directory / "model.joblib")
        values.append(max(0.0, float(model.predict(preprocessor.transform(features))[0])))
    ordered = sorted(values)
    return {name: ordered[index] for index, name in enumerate(QUANTILES)}


def _upcoming_position_row(
    schedules: pl.DataFrame,
    *,
    season: int,
    week: int,
    team: str,
    player_id: str,
    position: str,
) -> pl.DataFrame:
    config = POSITION_CONFIGS[position]
    game = schedules.filter(
        (pl.col("season") == season)
        & (pl.col("week") == week)
        & (pl.col("game_type") == "REG")
        & ((pl.col("home_team") == team) | (pl.col("away_team") == team))
    )
    if game.height != 1:
        raise ValueError(f"Expected one {team} game in {season} week {week}")
    record = game.row(0, named=True)
    home_game = record["home_team"] == team
    canonical_team_value = canonical_team_code(team)
    values: dict[str, list[object]] = {
        "game_id": [record["game_id"]],
        "game_date": [record["gameday"]],
        "season": [season],
        "week": [week],
        "player_id": [player_id],
        "team": [canonical_team_value],
        "opponent": [
            canonical_team_code(
                record["away_team"] if home_game else record["home_team"]
            )
        ],
        "home_game": [home_game],
        "days_rest": [record["home_rest"] if home_game else record["away_rest"]],
        "position": [position],
        "snap_share": [None],
    }
    values.update({outcome: [None] for outcome in config.outcomes})
    return pl.DataFrame(values).with_columns(
        pl.col("game_date").str.to_date(strict=False)
    )


def predict_live_position(
    player_name: str,
    position: str,
    season: int,
    week: int,
    *,
    team: str | None = None,
    processed_directory: Path = Path("data/processed"),
    models_directory: Path = Path("models"),
) -> dict[str, object]:
    """Predict every configured outcome for one upcoming QB/RB/TE game."""
    position = position.upper()
    config = POSITION_CONFIGS[position]
    schedules = nfl.load_schedules([season])
    stats = nfl.load_player_stats([season], summary_level="week")
    matches = stats.filter(
        (pl.col("player_display_name") == player_name)
        & (pl.col("position") == position)
    )
    player_ids = matches["player_id"].drop_nulls().unique().to_list()
    latest = matches.sort("week").tail(1)
    resolved_team = team or (latest[0, "team"] if latest.height else None)
    if len(player_ids) != 1 or resolved_team is None:
        raise ValueError(
            f"Could not uniquely resolve {player_name}: IDs={player_ids}, team={resolved_team}"
        )
    ngs_loader = getattr(nfl, "load_nextgen_receiving_optional", None)
    nextgen_receiving = ngs_loader(list(range(2016, season + 1))) if ngs_loader else pl.DataFrame()
    completed = build_position_player_games(
        stats,
        schedules,
        nfl.load_snap_counts([season]),
        nfl.load_players(),
        position,
        season,
        season,
    )
    historical_features = pl.read_parquet(
        processed_directory / f"{position.lower()}_player_features.parquet"
    )
    base_columns = [*IDENTIFIER_COLUMNS, *config.outcomes, "snap_share"]
    historical = historical_features.select(base_columns)
    upcoming = _upcoming_position_row(
        schedules,
        season=season,
        week=week,
        team=resolved_team,
        player_id=player_ids[0],
        position=position,
    )
    features = add_position_context_features(
        add_position_form_features(
            pl.concat(
                [
                    historical.select(base_columns),
                    completed.select(base_columns),
                    upcoming.select(base_columns),
                ],
                how="vertical_relaxed",
            ),
            position,
        ),
        position,
        schedules,
        nfl.load_pbp([season - 1, season]),
        nfl.load_depth_charts([season]),
        stats,
        nfl.load_injuries([season]),
        nextgen_receiving,
    ).filter(pl.col("game_id") == upcoming[0, "game_id"])
    comparison = json.loads(
        (models_directory / position.lower() / "comparison.json").read_text()
    )
    predictions = {}
    model_families = {}
    model_predictions: dict[str, dict[str, float]] = {
        "ridge": {}, "xgboost": {}, "neural_mlp": {}, "weighted_blend": {}
    }
    for outcome in config.outcomes:
        outcome_config = comparison["outcomes"][outcome]
        family_values = {}
        for family in ("ridge", "xgboost", "neural_mlp"):
            directory = models_directory / position.lower() / outcome / family
            preprocessor = joblib.load(directory / "preprocessor.joblib")
            model = joblib.load(directory / "model.joblib")
            family_values[family] = max(0.0, float(model.predict(preprocessor.transform(features))[0]))
            model_predictions[family][outcome] = family_values[family]
        weights = outcome_config.get("weights", {})
        predictions[outcome] = sum(weights.get(family, 0.0) * value for family, value in family_values.items())
        model_predictions["weighted_blend"][outcome] = predictions[outcome]
        model_families[outcome] = "weighted_blend"
    primary_quantiles = _predict_position_quantiles(features, models_directory, position)
    feature_values = {
        name: (None if value is None else float(value))
        for name, value in features.select(position_feature_columns(position)).row(0, named=True).items()
    }
    recent_rows = matches.filter(pl.col("week") < week).sort("week").tail(5)
    if position == "QB":
        recent_metric = "passing_yards"
        secondary_metric = "attempts"
    elif position == "RB":
        recent_rows = recent_rows.with_columns(
            (pl.col("rushing_yards") + pl.col("receiving_yards")).alias("total_yards")
        )
        recent_metric = "total_yards"
        secondary_metric = "carries"
    else:
        recent_metric = "receiving_yards"
        secondary_metric = None
    recent_games = [
        {
            "week": int(row["week"]),
            "game_id": row["game_id"],
            "opponent": row["opponent_team"],
            "value": float(row[recent_metric]),
            "secondary_value": (
                None if secondary_metric is None else float(row[secondary_metric])
            ),
        }
        for row in recent_rows.iter_rows(named=True)
    ]
    return {
        "player_name": player_name,
        "player_id": player_ids[0],
        "position": position,
        "team": canonical_team_code(resolved_team),
        "opponent": upcoming[0, "opponent"],
        "game_id": upcoming[0, "game_id"],
        "game_date": str(upcoming[0, "game_date"]),
        "predictions": predictions,
        "selected_models": model_families,
        "model_predictions": {
            family: primary_model_value(position, values)
            for family, values in model_predictions.items()
        },
        "features": feature_values,
        "recent_games": recent_games,
        "secondary_recent_label": (
            "Pass attempts" if secondary_metric == "attempts"
            else "Carries" if secondary_metric == "carries" else None
        ),
        "primary_target": PRIMARY_TARGETS[position],
        "primary_quantiles": primary_quantiles,
        "projected_ppr_points": fantasy_points(position, predictions),
    }


def _score_position_features(
    features: pl.DataFrame, models_directory: Path, position: str
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    config = POSITION_CONFIGS[position]
    comparison = json.loads(
        (models_directory / position.lower() / "comparison.json").read_text()
    )
    predictions = {}
    selected_models = {}
    for outcome in config.outcomes:
        outcome_config = comparison["outcomes"][outcome]
        selected = outcome_config["selected_model"]
        if selected == "weighted_blend":
            prediction = np.zeros(features.height)
            for family, weight in outcome_config["weights"].items():
                directory = models_directory / position.lower() / outcome / family
                preprocessor = joblib.load(directory / "preprocessor.joblib")
                model = joblib.load(directory / "model.joblib")
                prediction += weight * np.clip(
                    model.predict(preprocessor.transform(features)), 0.0, None
                )
            predictions[outcome] = prediction
        else:
            directory = models_directory / position.lower() / outcome / selected
            preprocessor = joblib.load(directory / "preprocessor.joblib")
            model = joblib.load(directory / "model.joblib")
            predictions[outcome] = np.clip(
                model.predict(preprocessor.transform(features)), 0.0, None
            )
        selected_models[outcome] = selected
    return predictions, selected_models


def position_ranking_candidates(
    stats: pl.DataFrame,
    depth_charts: pl.DataFrame,
    schedules: pl.DataFrame,
    position: str,
    season: int,
    week: int,
) -> pl.DataFrame:
    """Include pre-cutoff depth-chart players without prior game stats."""
    position = position.upper()
    current = (
        stats.filter(
            (pl.col("season") == season)
            & (pl.col("season_type") == "REG")
            & (pl.col("position") == position)
            & (pl.col("week") < week)
        )
        .sort(["player_id", "week"])
        .group_by("player_id", maintain_order=True)
        .agg(
            pl.col("player_display_name").last().alias("player_name"),
            pl.col("team").last().alias("team"),
            pl.lit(None, dtype=pl.Float64).alias("candidate_depth_rank"),
        )
    )
    depth_candidates = pl.DataFrame(
        schema={"player_id": pl.String, "player_name": pl.String, "team": pl.String}
    )
    current_columns = {"dt", "team", "gsis_id", "pos_abb", "pos_rank"}
    if current_columns.issubset(depth_charts.columns):
        game = schedules.filter(
            (pl.col("season") == season)
            & (pl.col("week") == week)
            & (pl.col("game_type") == "REG")
        ).sort("gameday").head(1)
        if game.height:
            cutoff = datetime.fromisoformat(str(game[0, "gameday"])).replace(
                hour=time.max.hour,
                minute=time.max.minute,
                second=time.max.second,
                microsecond=time.max.microsecond,
            )
            depth_candidates = (
                depth_charts.filter(
                    pl.col("gsis_id").is_not_null()
                    & (pl.col("pos_abb") == position)
                    & pl.col("dt").is_not_null()
                )
                .with_columns(
                    pl.col("dt")
                    .str.to_datetime("%Y-%m-%dT%H:%M:%SZ", strict=False)
                    .alias("snapshot_at"),
                    canonical_team("team").alias("canonical_team"),
                )
                .filter(
                    pl.col("snapshot_at").is_not_null()
                    & (pl.col("snapshot_at") <= cutoff)
                )
                .sort("snapshot_at", descending=True)
                .unique(["gsis_id", "canonical_team"], keep="first")
                .select(
                    pl.col("gsis_id").alias("player_id"),
                    pl.col("player_name"),
                    pl.col("canonical_team").alias("team"),
                    pl.col("pos_rank").cast(pl.Float64).alias("candidate_depth_rank"),
                )
            )
    return pl.concat([depth_candidates, current], how="vertical_relaxed").unique(
        "player_id", keep="first"
    )


def fantasy_points(position: str, predictions: dict[str, float]) -> float:
    if position == "QB":
        return (
            predictions.get("passing_yards", 0.0) / 25
            + predictions.get("passing_tds", 0.0) * 4
            - predictions.get("passing_interceptions", 0.0) * 2
            + predictions.get("rushing_yards", 0.0) / 10
            + predictions.get("rushing_tds", 0.0) * 6
        )
    if position == "RB":
        return (
            predictions.get("rushing_yards", 0.0) / 10
            + predictions.get("rushing_tds", 0.0) * 6
            + predictions.get("receptions", 0.0)
            + predictions.get("receiving_yards", 0.0) / 10
            + predictions.get("receiving_tds", 0.0) * 6
        )
    return (
        predictions.get("receptions", 0.0)
        + predictions.get("receiving_yards", 0.0) / 10
        + predictions.get("receiving_tds", 0.0) * 6
    )


def primary_model_value(position: str, predictions: dict[str, float]) -> float:
    if position == "QB":
        return predictions.get("passing_yards", 0.0)
    if position == "RB":
        return predictions.get("rushing_yards", 0.0) + predictions.get("receiving_yards", 0.0)
    return predictions.get("receiving_yards", 0.0)


def predict_position_rankings(
    position: str,
    season: int,
    week: int,
    *,
    ros: bool,
    processed_directory: Path = Path("data/processed"),
    models_directory: Path = Path("models"),
) -> list[dict[str, object]]:
    """Batch-score weekly or rest-of-season rankings for QB/RB/TE."""
    position = position.upper()
    config = POSITION_CONFIGS[position]
    schedules = nfl.load_schedules([season])
    play_by_play = nfl.load_pbp([season - 1, season])
    ngs_loader = getattr(nfl, "load_nextgen_receiving_optional", None)
    nextgen_receiving = ngs_loader(list(range(2016, season + 1))) if ngs_loader else pl.DataFrame()
    depth_charts = nfl.load_depth_charts([season])
    stats = nfl.load_player_stats([season], summary_level="week")
    players = position_ranking_candidates(
        stats, depth_charts, schedules, position, season, week
    )
    candidate_depth_ranks = {
        row["player_id"]: row["candidate_depth_rank"]
        for row in players.iter_rows(named=True)
    }
    historical_candidate_ids = set(
        stats.filter(
            (pl.col("season") == season)
            & (pl.col("season_type") == "REG")
            & (pl.col("position") == position)
            & (pl.col("week") < week)
        )["player_id"].drop_nulls().to_list()
    )
    completed = build_position_player_games(
        stats,
        schedules,
        nfl.load_snap_counts([season]),
        nfl.load_players(),
        position,
        season,
        season,
    )
    historical_features = pl.read_parquet(
        processed_directory / f"{position.lower()}_player_features.parquet"
    )
    base_columns = [*IDENTIFIER_COLUMNS, *config.outcomes, "snap_share"]
    historical = historical_features.select(base_columns)
    upcoming_by_week: dict[int, list[pl.DataFrame]] = {}
    metadata = []
    for player in players.iter_rows(named=True):
        team = player["team"]
        future_games = schedules.filter(
            (pl.col("season") == season)
            & (pl.col("game_type") == "REG")
            & (pl.col("week") >= week if ros else pl.col("week") == week)
            & ((pl.col("home_team") == team) | (pl.col("away_team") == team))
        ).sort("week")
        for projection_week in future_games["week"].to_list():
            upcoming_by_week.setdefault(int(projection_week), []).append(
                _upcoming_position_row(
                    schedules,
                    season=season,
                    week=int(projection_week),
                    team=team,
                    player_id=player["player_id"],
                    position=position,
                )
            )
            metadata.append(
                {
                    **player,
                    "team": canonical_team_code(team),
                    "projection_week": int(projection_week),
                }
            )
    feature_groups = []
    for projection_week, rows in upcoming_by_week.items():
        upcoming = pl.concat(rows, how="vertical_relaxed")
        feature_groups.append(
            add_position_context_features(
                add_position_form_features(
                pl.concat(
                    [historical.select(base_columns), completed.select(base_columns), upcoming.select(base_columns)],
                    how="vertical_relaxed",
                ),
                position,
                ),
                position,
                schedules,
                play_by_play,
                depth_charts,
                nextgen_receiving=nextgen_receiving,
            )
            .filter(pl.col("game_id").is_in(upcoming["game_id"]))
            .join(
                pl.DataFrame(metadata).filter(pl.col("projection_week") == projection_week),
                on="player_id",
                how="inner",
                validate="m:1",
            )
        )
    if not feature_groups:
        return []
    features = pl.concat(feature_groups, how="vertical_relaxed")
    scored, _ = _score_position_features(features, models_directory, position)
    totals: dict[str, dict[str, object]] = {}
    for index, row in enumerate(features.iter_rows(named=True)):
        values = {outcome: float(scored[outcome][index]) for outcome in config.outcomes}
        depth_rank = float(candidate_depth_ranks.get(row["player_id"]) or 0.0)
        if row["player_id"] not in historical_candidate_ids and depth_rank > 0.0:
            # Keep depth-chart-only returners visible without treating an RB4/WR5
            # as a starter when the model has no player usage history to anchor on.
            cold_start_factor = 1.0 / (depth_rank * depth_rank)
            values = {outcome: value * cold_start_factor for outcome, value in values.items()}
        result = totals.setdefault(
            row["player_id"],
            {
                "player_id": row["player_id"],
                "player_name": row["player_name"],
                "team": row["team_right"],
                "position": position,
                "remaining_games": 0,
                "week": row["projection_week"],
                "opponent": row["opponent"],
                "predictions": {outcome: 0.0 for outcome in config.outcomes},
            },
        )
        result["remaining_games"] += 1
        for outcome, value in values.items():
            result["predictions"][outcome] += value
    results = []
    for result in totals.values():
        predictions = result["predictions"]
        if position == "QB":
            result["projected_yards"] = predictions.get("passing_yards", 0.0)
            result["projected_receptions"] = 0.0
            result["projected_touchdowns"] = (
                predictions.get("passing_tds", 0.0) + predictions.get("rushing_tds", 0.0)
            )
        elif position == "RB":
            result["projected_yards"] = (
                predictions.get("rushing_yards", 0.0) + predictions.get("receiving_yards", 0.0)
            )
            result["projected_receptions"] = predictions.get("receptions", 0.0)
            result["projected_touchdowns"] = (
                predictions.get("rushing_tds", 0.0) + predictions.get("receiving_tds", 0.0)
            )
        else:
            result["projected_yards"] = predictions.get("receiving_yards", 0.0)
            result["projected_receptions"] = predictions.get("receptions", 0.0)
            result["projected_touchdowns"] = predictions.get("receiving_tds", 0.0)
        result["projected_ppr_points"] = fantasy_points(position, predictions)
        results.append(result)
    results.sort(key=lambda item: item["projected_ppr_points"], reverse=True)
    return results