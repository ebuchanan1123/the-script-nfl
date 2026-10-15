"""Chronological WR model training orchestration."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from the_script.evaluation.backtest import chronological_splits
from the_script.models.baseline import evaluate_recent_average, train_ridge
from the_script.models.preprocessing import (
	BASE_NUMERIC_FEATURES,
	NUMERIC_FEATURES,
	TARGET_COLUMN,
	YARDS_NEURAL_FEATURES,
	YARDS_RIDGE_FEATURES,
	YARDS_XGBOOST_FEATURES,
	NumericPreprocessor,
)


def train_wr_baselines(frame: pl.DataFrame, models_directory: Path) -> dict[str, object]:
	"""Train and compare recent average, Ridge, and XGBoost under one split contract."""
	from the_script.models.boosting import train_xgboost

	recent = evaluate_recent_average(frame)
	recent_directory = models_directory / "wr_recent_average"
	recent_directory.mkdir(parents=True, exist_ok=True)
	(recent_directory / "metrics.json").write_text(json.dumps(recent, indent=2))
	results = {
		"recent_average": recent,
		"ridge": train_ridge(
			frame,
			models_directory / "wr_ridge",
			features=[
				feature for feature in YARDS_RIDGE_FEATURES
				if feature in BASE_NUMERIC_FEATURES
			],
		),
		"xgboost": train_xgboost(
			frame,
			models_directory / "wr_xgboost",
			features=[
				feature for feature in YARDS_XGBOOST_FEATURES
				if feature in BASE_NUMERIC_FEATURES
			],
		),
	}
	(models_directory / "wr_baseline_comparison.json").write_text(json.dumps(results, indent=2))
	return results


def train_wr_baselines_from_parquet(input_path: Path, models_directory: Path) -> dict[str, object]:
	return train_wr_baselines(pl.read_parquet(input_path), models_directory)


def train_wr_volume_models(
	frame: pl.DataFrame, models_directory: Path
) -> dict[str, object]:
	"""Train separate Ridge and XGBoost models for targets and receptions."""
	from the_script.models.boosting import train_xgboost

	results = {}
	for target_column in ("targets", "receptions"):
		output_directory = models_directory / "wr_volume" / target_column
		results[target_column] = {
			"ridge": train_ridge(
				frame,
				output_directory / "ridge",
				features=NUMERIC_FEATURES,
				target_column=target_column,
			),
			"xgboost": train_xgboost(
				frame,
				output_directory / "xgboost",
				features=NUMERIC_FEATURES,
				target_column=target_column,
			),
		}
	comparison_path = models_directory / "wr_volume" / "comparison.json"
	comparison_path.parent.mkdir(parents=True, exist_ok=True)
	comparison_path.write_text(json.dumps(results, indent=2))
	return results


def train_wr_volume_models_from_parquet(
	input_path: Path, models_directory: Path
) -> dict[str, object]:
	return train_wr_volume_models(pl.read_parquet(input_path), models_directory)


def _neural_predictions(model: Any, values: np.ndarray, batch_size: int) -> np.ndarray:
	import torch

	model.eval()
	predictions = []
	with torch.no_grad():
		for start in range(0, len(values), batch_size):
			batch = torch.from_numpy(values[start : start + batch_size])
			predictions.append(model(batch).numpy())
	return np.concatenate(predictions)


def train_wr_neural(
	frame: pl.DataFrame,
	output_directory: Path,
	*,
	batch_size: int = 128,
	learning_rate: float = 1e-3,
	max_epochs: int = 100,
	patience: int = 10,
	features: list[str] | None = None,
) -> dict[str, object]:
	"""Train the numerical-only MLP with validation-based early stopping."""
	import torch
	from torch import nn
	from torch.utils.data import DataLoader, TensorDataset

	from the_script.evaluation.metrics import holdout_metrics
	from the_script.models.neural_network import NumericalWRModel

	torch.manual_seed(42)
	np.random.seed(42)
	splits = chronological_splits(frame)
	preprocessor = NumericPreprocessor.fit(
		splits["train"], YARDS_NEURAL_FEATURES if features is None else features
	)
	arrays = {name: preprocessor.transform(split) for name, split in splits.items()}
	targets = {
		name: split[TARGET_COLUMN].to_numpy().astype(np.float32)
		for name, split in splits.items()
	}
	generator = torch.Generator().manual_seed(42)
	train_loader = DataLoader(
		TensorDataset(torch.from_numpy(arrays["train"]), torch.from_numpy(targets["train"])),
		batch_size=batch_size,
		shuffle=True,
		generator=generator,
	)
	model = NumericalWRModel(len(preprocessor.features))
	criterion = nn.HuberLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

	output_directory.mkdir(parents=True, exist_ok=True)
	checkpoint = output_directory / "model.pt"
	best_validation_loss = float("inf")
	epochs_without_improvement = 0
	history = []
	validation_values = torch.from_numpy(arrays["validation"])
	validation_targets = torch.from_numpy(targets["validation"])
	for epoch in range(1, max_epochs + 1):
		model.train()
		total_loss = 0.0
		total_rows = 0
		for numeric, target in train_loader:
			optimizer.zero_grad()
			prediction = model(numeric)
			loss = criterion(prediction, target)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			total_loss += loss.item() * len(target)
			total_rows += len(target)

		model.eval()
		with torch.no_grad():
			validation_loss = criterion(model(validation_values), validation_targets).item()
		history.append(
			{
				"epoch": epoch,
				"training_loss": total_loss / total_rows,
				"validation_loss": validation_loss,
			}
		)
		if validation_loss < best_validation_loss:
			best_validation_loss = validation_loss
			torch.save(model.state_dict(), checkpoint)
			epochs_without_improvement = 0
		else:
			epochs_without_improvement += 1
		if epochs_without_improvement >= patience:
			break

	model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
	results: dict[str, object] = {
		"model": "numerical_mlp",
		"architecture": [len(preprocessor.features), 64, 32, 1],
		"features": preprocessor.features,
		"loss": "HuberLoss",
		"optimizer": "AdamW",
		"learning_rate": learning_rate,
		"batch_size": batch_size,
		"epochs_completed": len(history),
		"best_validation_loss": best_validation_loss,
	}
	for split_name in ("validation", "test"):
		predictions = _neural_predictions(model, arrays[split_name], batch_size)
		results[split_name] = holdout_metrics(splits[split_name], predictions)

	joblib.dump(preprocessor, output_directory / "preprocessor.joblib")
	(output_directory / "metrics.json").write_text(json.dumps(results, indent=2))
	(output_directory / "training_history.json").write_text(json.dumps(history, indent=2))
	return results


def train_wr_neural_from_parquet(input_path: Path, output_directory: Path, **kwargs: Any) -> dict[str, object]:
	return train_wr_neural(pl.read_parquet(input_path), output_directory, **kwargs)


def train_wr_opponent_experiment(
	frame: pl.DataFrame, models_directory: Path
) -> dict[str, object]:
	"""Train opponent Ridge/XGBoost and compare all separately trained artifacts."""
	base_paths = {
		"ridge": models_directory / "wr_ridge" / "metrics.json",
		"xgboost": models_directory / "wr_xgboost" / "metrics.json",
		"numerical_mlp": models_directory / "wr_neural" / "metrics.json",
	}
	missing = [str(path) for path in base_paths.values() if not path.exists()]
	if missing:
		raise FileNotFoundError(f"Train base models before the opponent experiment: {missing}")
	opponent_directory = models_directory / "wr_opponent"
	neural_path = opponent_directory / "neural" / "metrics.json"
	if not neural_path.exists():
		raise FileNotFoundError("Run train-wr-opponent-neural before the comparison")
	opponent_results = {
		"ridge": train_ridge(
			frame, opponent_directory / "ridge", features=YARDS_RIDGE_FEATURES
		),
		"numerical_mlp": json.loads(neural_path.read_text()),
	}
	from the_script.models.boosting import train_xgboost

	opponent_results["xgboost"] = train_xgboost(
		frame, opponent_directory / "xgboost", features=YARDS_XGBOOST_FEATURES
	)
	results = {
		"base": {name: json.loads(path.read_text()) for name, path in base_paths.items()},
		"with_opponent": opponent_results,
	}
	for experiment in results.values():
		for metrics in experiment.values():
			test = metrics["test"].get("common_history_cohort", metrics["test"])
			metrics["common_test_mae"] = test["mae"]
			metrics["common_test_rmse"] = test["rmse"]
	comparison_path = opponent_directory / "comparison.json"
	comparison_path.parent.mkdir(parents=True, exist_ok=True)
	comparison_path.write_text(json.dumps(results, indent=2))
	return results


def train_wr_opponent_experiment_from_parquet(
	input_path: Path, output_directory: Path
) -> dict[str, object]:
	return train_wr_opponent_experiment(pl.read_parquet(input_path), output_directory)


def train_wr_opponent_neural_from_parquet(
	input_path: Path, models_directory: Path
) -> dict[str, object]:
	"""Train opponent MLP in a process that never loads XGBoost/OpenMP."""
	return train_wr_neural(
		pl.read_parquet(input_path),
		models_directory / "wr_opponent" / "neural",
		features=YARDS_NEURAL_FEATURES,
	)


def compare_trained_wr_models(models_directory: Path) -> dict[str, object]:
	"""Consolidate persisted metrics without retraining or touching holdout data."""
	metric_paths = {
		"recent_average": models_directory / "wr_recent_average" / "metrics.json",
		"ridge": models_directory / "wr_ridge" / "metrics.json",
		"xgboost": models_directory / "wr_xgboost" / "metrics.json",
		"numerical_mlp": models_directory / "wr_neural" / "metrics.json",
	}
	metrics = {name: json.loads(path.read_text()) for name, path in metric_paths.items()}
	common_test = {
		name: values["test"].get("common_history_cohort", values["test"])
		for name, values in metrics.items()
	}
	result = {
		"models": metrics,
		"common_test_cohort": common_test,
		"lowest_test_mae": min(common_test, key=lambda name: common_test[name]["mae"]),
		"lowest_test_rmse": min(common_test, key=lambda name: common_test[name]["rmse"]),
	}
	(models_directory / "wr_model_comparison.json").write_text(json.dumps(result, indent=2))
	return result
