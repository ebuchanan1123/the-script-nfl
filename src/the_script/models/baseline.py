"""Recent-average receiving-yards baseline."""

import json
from pathlib import Path

import joblib
import polars as pl
from sklearn.linear_model import Ridge

from the_script.evaluation.backtest import chronological_splits
from the_script.evaluation.metrics import holdout_metrics, regression_metrics
from the_script.models.preprocessing import TARGET_COLUMN, NumericPreprocessor


def evaluate_recent_average(frame: pl.DataFrame) -> dict[str, object]:
	"""Score `yards_avg_last_5` on the fixed 2024 and 2025 holdouts."""
	required = {"season", "receiving_yards", "yards_avg_last_5"}
	missing = sorted(required.difference(frame.columns))
	if missing:
		raise ValueError(f"player features are missing columns: {', '.join(missing)}")

	results: dict[str, object] = {
		"model": "recent_five_game_average",
		"prediction_column": "yards_avg_last_5",
		"target_column": "receiving_yards",
	}
	splits = chronological_splits(frame)
	results["train_rows"] = splits["train"].height
	for split_name in ("validation", "test"):
		scored = splits[split_name].drop_nulls(["yards_avg_last_5", "receiving_yards"])
		results[split_name] = {
			"rows": scored.height,
			**regression_metrics(scored["receiving_yards"], scored["yards_avg_last_5"]),
		}
	return results


def evaluate_recent_average_from_parquet(input_path: Path, output_path: Path) -> dict[str, object]:
	results = evaluate_recent_average(pl.read_parquet(input_path))
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(results, indent=2))
	return results


def train_ridge(
	frame: pl.DataFrame,
	output_directory: Path,
	*,
	alpha: float = 10.0,
	features: list[str] | None = None,
	target_column: str = TARGET_COLUMN,
) -> dict[str, object]:
	"""Fit Ridge on training seasons and score untouched 2024/2025 holdouts."""
	if target_column not in frame.columns:
		raise ValueError(f"Training data is missing target column: {target_column}")
	splits = chronological_splits(frame)
	preprocessor = NumericPreprocessor.fit(splits["train"], features)
	model = Ridge(alpha=alpha)
	model.fit(
		preprocessor.transform(splits["train"]),
		splits["train"][target_column].to_numpy(),
	)

	results: dict[str, object] = {
		"model": "ridge",
		"alpha": alpha,
		"target_column": target_column,
		"features": preprocessor.features,
	}
	for split_name in ("validation", "test"):
		split = splits[split_name]
		predictions = model.predict(preprocessor.transform(split))
		results[split_name] = holdout_metrics(
			split, predictions, target_column=target_column
		)

	output_directory.mkdir(parents=True, exist_ok=True)
	joblib.dump(model, output_directory / "model.joblib")
	joblib.dump(preprocessor, output_directory / "preprocessor.joblib")
	(output_directory / "metrics.json").write_text(json.dumps(results, indent=2))
	return results
