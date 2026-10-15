"""Receiving-touchdown count and anytime probability models."""

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import PoissonRegressor

from the_script.evaluation.backtest import chronological_splits
from the_script.evaluation.metrics import touchdown_metrics
from the_script.models.preprocessing import NUMERIC_FEATURES, NumericPreprocessor

TOUCHDOWN_TARGET = "receiving_touchdowns"


def train_touchdown_model(
    frame: pl.DataFrame,
    output_directory: Path,
    *,
    model_name: str,
    features: list[str] | None = None,
) -> dict[str, object]:
    if TOUCHDOWN_TARGET not in frame.columns:
        raise ValueError(f"Training data is missing target column: {TOUCHDOWN_TARGET}")
    splits = chronological_splits(frame)
    preprocessor = NumericPreprocessor.fit(
        splits["train"], NUMERIC_FEATURES if features is None else features
    )
    if model_name == "poisson":
        model = PoissonRegressor(alpha=1.0, max_iter=1000)
    elif model_name == "xgboost_poisson":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="count:poisson",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported touchdown model: {model_name}")
    model.fit(
        preprocessor.transform(splits["train"]),
        splits["train"][TOUCHDOWN_TARGET].to_numpy(),
    )
    results: dict[str, object] = {
        "model": model_name,
        "target_column": TOUCHDOWN_TARGET,
        "features": preprocessor.features,
    }
    for split_name in ("validation", "test"):
        split = splits[split_name]
        predictions = np.clip(model.predict(preprocessor.transform(split)), 1e-6, None)
        results[split_name] = {
            "rows": split.height,
            **touchdown_metrics(split[TOUCHDOWN_TARGET], predictions),
        }
    output_directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_directory / "model.joblib")
    joblib.dump(preprocessor, output_directory / "preprocessor.joblib")
    (output_directory / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


def train_wr_touchdown_models(
    frame: pl.DataFrame, models_directory: Path
) -> dict[str, object]:
    """Train both touchdown families and select by validation Brier score."""
    output_directory = models_directory / "wr_touchdowns"
    results = {
        model_name: train_touchdown_model(
            frame, output_directory / model_name, model_name=model_name
        )
        for model_name in ("poisson", "xgboost_poisson")
    }
    selected_model = min(
        results,
        key=lambda name: results[name]["validation"]["brier_score"],
    )
    comparison = {"selected_model": selected_model, "models": results}
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "comparison.json").write_text(json.dumps(comparison, indent=2))
    return comparison


def train_wr_touchdown_models_from_parquet(
    input_path: Path, models_directory: Path
) -> dict[str, object]:
    return train_wr_touchdown_models(pl.read_parquet(input_path), models_directory)