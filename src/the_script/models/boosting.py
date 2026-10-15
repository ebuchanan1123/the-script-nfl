"""XGBoost WR receiving-yards baseline."""

import json
from pathlib import Path

import joblib
import polars as pl
from xgboost import XGBRegressor

from the_script.evaluation.backtest import chronological_splits
from the_script.evaluation.metrics import holdout_metrics
from the_script.models.preprocessing import TARGET_COLUMN, NumericPreprocessor


def train_xgboost(
	frame: pl.DataFrame,
	output_directory: Path,
	*,
	features: list[str] | None = None,
	target_column: str = TARGET_COLUMN,
) -> dict[str, object]:
	"""Fit the fixed boosting baseline on training seasons only."""
	if target_column not in frame.columns:
		raise ValueError(f"Training data is missing target column: {target_column}")
	splits = chronological_splits(frame)
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
	model.fit(
		preprocessor.transform(splits["train"]),
		splits["train"][target_column].to_numpy(),
	)

	results: dict[str, object] = {
		"model": "xgboost",
		"target_column": target_column,
		"features": preprocessor.features,
		"parameters": model.get_params(),
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
