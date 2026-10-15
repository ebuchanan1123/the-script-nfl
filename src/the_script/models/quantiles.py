"""Receiving-yards conditional quantile models."""

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from the_script.evaluation.backtest import chronological_splits
from the_script.evaluation.metrics import quantile_metrics
from the_script.models.preprocessing import NUMERIC_FEATURES, NumericPreprocessor

QUANTILES = {"q10": 0.1, "q50": 0.5, "q90": 0.9}


def train_wr_quantile_models(
    frame: pl.DataFrame, models_directory: Path
) -> dict[str, object]:
    """Train 10th/50th/90th percentile XGBoost models chronologically."""
    from xgboost import XGBRegressor

    splits = chronological_splits(frame)
    output_directory = models_directory / "wr_quantiles"
    models = {}
    predictions: dict[str, dict[str, np.ndarray]] = {
        "validation": {},
        "test": {},
    }
    for name, alpha in QUANTILES.items():
        preprocessor = NumericPreprocessor.fit(splits["train"], NUMERIC_FEATURES)
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
        model.fit(
            preprocessor.transform(splits["train"]),
            splits["train"]["receiving_yards"].to_numpy(),
        )
        model_directory = output_directory / name
        model_directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_directory / "model.joblib")
        joblib.dump(preprocessor, model_directory / "preprocessor.joblib")
        models[name] = {"alpha": alpha, "features": preprocessor.features}
        for split_name, split_predictions in predictions.items():
            split_predictions[name] = np.clip(
                model.predict(preprocessor.transform(splits[split_name])), 0.0, None
            )
    results: dict[str, object] = {"models": models}
    for split_name in predictions:
        ordered = np.sort(
            np.column_stack(
                [predictions[split_name][name] for name in QUANTILES]
            ),
            axis=1,
        )
        ordered_predictions = {
            name: ordered[:, index] for index, name in enumerate(QUANTILES)
        }
        results[split_name] = {
            "rows": splits[split_name].height,
            **quantile_metrics(
                splits[split_name]["receiving_yards"], ordered_predictions
            ),
        }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


def train_wr_quantile_models_from_parquet(
    input_path: Path, models_directory: Path
) -> dict[str, object]:
    return train_wr_quantile_models(pl.read_parquet(input_path), models_directory)