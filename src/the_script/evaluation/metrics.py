"""WR receiving-yards regression metrics."""

import math

import numpy as np
import polars as pl
from sklearn.metrics import (
	brier_score_loss,
	log_loss,
	mean_pinball_loss,
	mean_poisson_deviance,
)


def regression_metrics(actual: pl.Series, predicted: pl.Series) -> dict[str, float]:
	"""Return interpretable error metrics for aligned non-null predictions."""
	errors = actual.cast(pl.Float64) - predicted.cast(pl.Float64)
	absolute_errors = errors.abs()
	return {
		"mae": float(absolute_errors.mean()),
		"rmse": math.sqrt(float((errors**2).mean())),
		"median_absolute_error": float(absolute_errors.median()),
	}


def holdout_metrics(
	frame: pl.DataFrame,
	predicted: pl.Series | np.ndarray,
	*,
	target_column: str = "receiving_yards",
	common_cohort_column: str = "yards_avg_last_5",
) -> dict[str, object]:
	"""Report full holdout and the cohort scorable by the recent-average baseline."""
	prediction_series = predicted if isinstance(predicted, pl.Series) else pl.Series(predicted)
	result: dict[str, object] = {
		"rows": frame.height,
		**regression_metrics(frame[target_column], prediction_series),
	}
	mask = frame[common_cohort_column].is_not_null()
	result["common_history_cohort"] = {
		"rows": int(mask.sum()),
		**regression_metrics(frame.filter(mask)[target_column], prediction_series.filter(mask)),
	}
	return result


def touchdown_metrics(
	actual: pl.Series, expected_touchdowns: pl.Series | np.ndarray
) -> dict[str, float]:
	"""Evaluate expected TD counts and calibrated anytime-TD probabilities."""
	actual_values = actual.cast(pl.Float64).to_numpy()
	expected = np.clip(np.asarray(expected_touchdowns, dtype=np.float64), 1e-6, None)
	anytime_actual = (actual_values > 0).astype(np.int8)
	anytime_probability = np.clip(1.0 - np.exp(-expected), 1e-6, 1.0 - 1e-6)
	return {
		"mae": float(np.mean(np.abs(actual_values - expected))),
		"poisson_deviance": float(mean_poisson_deviance(actual_values, expected)),
		"brier_score": float(brier_score_loss(anytime_actual, anytime_probability)),
		"log_loss": float(log_loss(anytime_actual, anytime_probability, labels=[0, 1])),
		"actual_anytime_rate": float(anytime_actual.mean()),
		"mean_predicted_anytime_probability": float(anytime_probability.mean()),
	}


def quantile_metrics(
	actual: pl.Series,
	predictions: dict[str, np.ndarray],
) -> dict[str, float]:
	"""Evaluate marginal quantiles and the central 80% prediction interval."""
	actual_values = actual.cast(pl.Float64).to_numpy()
	q10 = np.asarray(predictions["q10"], dtype=np.float64)
	q50 = np.asarray(predictions["q50"], dtype=np.float64)
	q90 = np.asarray(predictions["q90"], dtype=np.float64)
	return {
		"q10_pinball_loss": float(mean_pinball_loss(actual_values, q10, alpha=0.1)),
		"q50_pinball_loss": float(mean_pinball_loss(actual_values, q50, alpha=0.5)),
		"q90_pinball_loss": float(mean_pinball_loss(actual_values, q90, alpha=0.9)),
		"interval_coverage": float(((actual_values >= q10) & (actual_values <= q90)).mean()),
		"mean_interval_width": float(np.mean(q90 - q10)),
	}
