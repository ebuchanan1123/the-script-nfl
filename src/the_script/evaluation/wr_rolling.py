"""Expanding-window diagnostics for WR models using the 2024 validation season."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from the_script.models.ensemble import DIRECT_MODEL_PATHS, _convex_weights

MODEL_COLUMNS = [*DIRECT_MODEL_PATHS, "decomposition"]
FOLDS = ((6, 10), (10, 14), (14, 18))


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    errors = predicted - actual
    return {
        "rows": len(actual),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "bias": float(np.mean(errors)),
        "median_absolute_error": float(np.median(np.abs(errors))),
    }


def evaluate_rolling_validation(
    predictions_path: Path,
    features_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Fit convex blend weights on early 2024 weeks and score only later weeks.

    Rows are snap-verified WR game participants, matching the model target contract.
    The 2025 test season is never read by this evaluation.
    """
    predictions = pl.read_parquet(predictions_path)
    features = pl.read_parquet(features_path).select(
        "game_id", "player_id", "targets_avg_last_5", "yards_avg_last_5"
    )
    required = {"game_id", "player_id", "receiving_yards", *MODEL_COLUMNS}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction file is missing columns: {', '.join(missing)}")
    rows = predictions.join(
        features, on=["game_id", "player_id"], how="left", validate="1:1"
    ).with_columns(
        pl.when(pl.col("targets_avg_last_5").is_null())
        .then(pl.lit("unknown"))
        .when(pl.col("targets_avg_last_5") < 4)
        .then(pl.lit("low (<4 targets/game)"))
        .when(pl.col("targets_avg_last_5") < 8)
        .then(pl.lit("mid (4-8 targets/game)"))
        .otherwise(pl.lit("high (8+ targets/game)"))
        .alias("pregame_usage_tier")
    )
    if rows.height == 0 or rows.filter(pl.col("game_id").str.starts_with("2024_")).height != rows.height:
        raise ValueError("Rolling validation expects non-empty 2024-only prediction rows")
    if rows.select(pl.struct("game_id", "player_id").n_unique()).item() != rows.height:
        raise ValueError("Prediction rows must be unique by (game_id, player_id)")
    if rows.select(pl.any_horizontal([pl.col(c).is_null() for c in MODEL_COLUMNS]).any()).item():
        raise ValueError("Model predictions contain null values")

    fold_results = []
    scored_frames = []
    common_frames = []
    prior_blend_errors: list[np.ndarray] = []
    calibrated_frames = []
    for training_end, scoring_end in FOLDS:
        train = rows.filter(pl.col("game_id").str.slice(5, 2).cast(pl.Int32) <= training_end)
        scored = rows.filter(
            (pl.col("game_id").str.slice(5, 2).cast(pl.Int32) > training_end)
            & (pl.col("game_id").str.slice(5, 2).cast(pl.Int32) <= scoring_end)
        )
        if train.is_empty() or scored.is_empty():
            continue
        weights = _convex_weights(
            train.select(MODEL_COLUMNS).to_numpy(),
            train["receiving_yards"].to_numpy(),
        )
        blend = scored.select(MODEL_COLUMNS).to_numpy() @ weights
        scored = scored.with_columns(pl.Series("rolling_blend", blend))
        scored_frames.append(scored)
        comparison = scored.filter(pl.col("yards_avg_last_5").is_not_null())
        if comparison.is_empty():
            raise ValueError(f"No common baseline cohort in score weeks {training_end + 1}-{scoring_end}")
        actual = comparison["receiving_yards"].to_numpy().astype(float)
        metrics = {name: _metrics(actual, comparison[name].to_numpy().astype(float))
                   for name in (*MODEL_COLUMNS, "rolling_blend")}
        metrics["recent_average"] = _metrics(
            actual,
            comparison["yards_avg_last_5"].to_numpy().astype(float),
        )
        common_frames.append(comparison)
        fold_results.append({
            "fit_weeks": f"1-{training_end}",
            "score_weeks": f"{training_end + 1}-{scoring_end}",
            "scored_rows": scored.height,
            "common_cohort_rows": comparison.height,
            "weights": dict(zip(MODEL_COLUMNS, map(float, weights), strict=True)),
            "metrics": metrics,
        })
        blend_errors = comparison["rolling_blend"].to_numpy().astype(float) - actual
        if prior_blend_errors:
            correction = -float(np.mean(np.concatenate(prior_blend_errors)))
            corrected = comparison.with_columns(
                (pl.col("rolling_blend") + correction).alias("bias_calibrated_blend")
            )
            calibrated_metrics = _metrics(
                actual, corrected["bias_calibrated_blend"].to_numpy().astype(float)
            )
            fold_results[-1]["bias_calibration"] = {
                "fit_on_prior_scored_weeks": True,
                "correction_yards": correction,
                "metrics": calibrated_metrics,
            }
            calibrated_frames.append(corrected)
        else:
            fold_results[-1]["bias_calibration"] = None
        prior_blend_errors.append(blend_errors)

    if not scored_frames:
        raise ValueError("No rolling validation folds could be evaluated")
    scored = pl.concat(scored_frames)
    common = pl.concat(common_frames)
    actual = common["receiving_yards"].to_numpy().astype(float)
    overall = {name: _metrics(actual, common[name].to_numpy().astype(float))
               for name in (*MODEL_COLUMNS, "rolling_blend")}
    overall["recent_average"] = _metrics(
        actual,
        common["yards_avg_last_5"].to_numpy().astype(float),
    )
    if calibrated_frames:
        calibrated = pl.concat(calibrated_frames)
        calibrated_actual = calibrated["receiving_yards"].to_numpy().astype(float)
        calibrated_metrics = _metrics(
            calibrated_actual,
            calibrated["bias_calibrated_blend"].to_numpy().astype(float),
        )
        uncalibrated_metrics = _metrics(
            calibrated_actual,
            calibrated["rolling_blend"].to_numpy().astype(float),
        )
        calibrated_baseline = _metrics(
            calibrated_actual,
            calibrated["yards_avg_last_5"].to_numpy().astype(float),
        )
    else:
        calibrated_metrics = uncalibrated_metrics = calibrated_baseline = None
    by_usage = {}
    for tier in common["pregame_usage_tier"].unique().sort().to_list():
        subset = common.filter(pl.col("pregame_usage_tier") == tier)
        actual = subset["receiving_yards"].to_numpy().astype(float)
        by_usage[tier] = {
            name: _metrics(actual, subset[name].to_numpy().astype(float))
            for name in ("numerical_mlp", "decomposition", "rolling_blend", "yards_avg_last_5")
        }
    result = {
        "protocol": "Expanding-window 2024 validation; fit blend weights only on earlier weeks; score weeks 7-18.",
        "population": "Snap-verified WR game participants; bye/inactive players are outside this target contract.",
        "scored_weeks": "7-18",
        "scored_rows": scored.height,
        "common_cohort_rows": common.height,
        "common_cohort": "Rows with a pregame yards_avg_last_5 baseline; every model and baseline metric uses this same cohort.",
        "folds": fold_results,
        "overall": overall,
        "bias_calibration_on_later_folds": {
            "protocol": "Fit one additive residual correction on prior out-of-fold validation weeks only; apply it to the next fold.",
            "score_weeks": "11-18",
            "rows": sum(frame.height for frame in calibrated_frames),
            "uncalibrated_blend": uncalibrated_metrics,
            "calibrated_blend": calibrated_metrics,
            "recent_average": calibrated_baseline,
        },
        "by_pregame_usage_tier": by_usage,
        "interpretation": "Use this report to select experiments. Do not tune on the 2025 test set.",
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result
