"""Validation-fitted yards ensemble and structured receiving decomposition."""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

from the_script.evaluation.backtest import chronological_splits
from the_script.evaluation.metrics import holdout_metrics
from the_script.models.preprocessing import NUMERIC_FEATURES, NumericPreprocessor

DIRECT_MODEL_PATHS = {
    "ridge": ("joblib", Path("wr_opponent/ridge")),
    "xgboost": ("joblib", Path("wr_opponent/xgboost")),
    "numerical_mlp": ("neural", Path("wr_opponent/neural")),
}


def _score_artifact(
    model_type: str, directory: Path, frame_path: Path
) -> np.ndarray:
    frame = pl.read_parquet(frame_path)
    preprocessor = joblib.load(directory / "preprocessor.joblib")
    values = preprocessor.transform(frame)
    if model_type == "neural":
        import torch

        from the_script.models.neural_network import NumericalWRModel

        model = NumericalWRModel(len(preprocessor.features))
        model.load_state_dict(
            torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        )
        model.eval()
        with torch.no_grad():
            return model(torch.from_numpy(values)).numpy()
    model = joblib.load(directory / "model.joblib")
    return model.predict(values)


def _direct_predictions(
    models_directory: Path,
    splits: dict[str, pl.DataFrame],
    work_directory: Path,
) -> dict[str, dict[str, np.ndarray]]:
    work_directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split_name in ("validation", "test"):
        path = work_directory / f"{split_name}.parquet"
        splits[split_name].write_parquet(path)
        paths[split_name] = path
    predictions: dict[str, dict[str, np.ndarray]] = {
        split_name: {} for split_name in paths
    }
    context = multiprocessing.get_context("spawn")
    for model_name, (model_type, relative_path) in DIRECT_MODEL_PATHS.items():
        for split_name, frame_path in paths.items():
            with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
                future = executor.submit(
                    _score_artifact,
                    model_type,
                    models_directory / relative_path,
                    frame_path,
                )
                predictions[split_name][model_name] = np.asarray(future.result())
    return predictions


def _fit_decomposition(
    splits: dict[str, pl.DataFrame], output_directory: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    preprocessor = NumericPreprocessor.fit(splits["train"], NUMERIC_FEATURES)
    train = splits["train"]
    values = preprocessor.transform(train)
    targets_model = Ridge(alpha=10.0).fit(values, train["targets"].to_numpy())
    catch_rate = (
        train["receptions"] / train["targets"].cast(pl.Float64).replace(0, None)
    ).fill_null(0.0)
    yards_per_reception = (
        train["receiving_yards"]
        / train["receptions"].cast(pl.Float64).replace(0, None)
    ).fill_null(0.0).clip(0.0, 40.0)
    catch_rate_model = Ridge(alpha=10.0).fit(values, catch_rate.to_numpy())
    ypr_model = Ridge(alpha=10.0).fit(values, yards_per_reception.to_numpy())
    predictions = {}
    for split_name in ("validation", "test"):
        transformed = preprocessor.transform(splits[split_name])
        expected_targets = np.clip(targets_model.predict(transformed), 0.0, 20.0)
        expected_catch_rate = np.clip(
            catch_rate_model.predict(transformed), 0.0, 1.0
        )
        expected_ypr = np.clip(ypr_model.predict(transformed), 0.0, 40.0)
        predictions[split_name] = (
            expected_targets * expected_catch_rate * expected_ypr
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(targets_model, output_directory / "targets_model.joblib")
    joblib.dump(catch_rate_model, output_directory / "catch_rate_model.joblib")
    joblib.dump(ypr_model, output_directory / "yards_per_reception_model.joblib")
    joblib.dump(preprocessor, output_directory / "preprocessor.joblib")
    metadata = {
        "formula": "targets * catch_rate * yards_per_reception",
        "features": preprocessor.features,
        "training_seasons": "2016-2023",
    }
    (output_directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return predictions, metadata


def _convex_weights(predictions: np.ndarray, actual: np.ndarray) -> np.ndarray:
    number_of_models = predictions.shape[1]
    initial = np.full(number_of_models, 1.0 / number_of_models)
    result = minimize(
        lambda weights: np.mean(np.abs(actual - predictions @ weights)),
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * number_of_models,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Could not fit ensemble weights: {result.message}")
    weights = np.clip(result.x, 0.0, 1.0)
    return weights / weights.sum()


def evaluate_yards_challengers(
    frame: pl.DataFrame,
    models_directory: Path,
) -> dict[str, Any]:
    """Fit challengers without using 2025 outcomes until final scoring."""
    splits = chronological_splits(frame)
    output_directory = models_directory / "wr_ensemble"
    direct = _direct_predictions(
        models_directory, splits, output_directory / "prediction_frames"
    )
    decomposition, decomposition_metadata = _fit_decomposition(
        splits, models_directory / "wr_decomposition"
    )
    candidate_names = [*DIRECT_MODEL_PATHS, "decomposition"]
    matrices = {
        split_name: np.column_stack(
            [
                *[direct[split_name][name] for name in DIRECT_MODEL_PATHS],
                decomposition[split_name],
            ]
        )
        for split_name in ("validation", "test")
    }
    validation_actual = splits["validation"]["receiving_yards"].to_numpy()
    weights = _convex_weights(matrices["validation"], validation_actual)
    results: dict[str, Any] = {
        "candidate_names": candidate_names,
        "weights": dict(zip(candidate_names, weights.tolist(), strict=True)),
        "weight_fit_season": 2024,
        "decomposition": decomposition_metadata,
    }
    validation_weeks = (
        splits["validation"]["game_id"].str.slice(5, 2).cast(pl.Int32).to_numpy()
    )
    rolling_validation = []
    for training_end, scoring_end in ((6, 10), (10, 14), (14, 18)):
        training_mask = validation_weeks <= training_end
        scoring_mask = (validation_weeks > training_end) & (
            validation_weeks <= scoring_end
        )
        rolling_weights = _convex_weights(
            matrices["validation"][training_mask], validation_actual[training_mask]
        )
        rolling_prediction = matrices["validation"][scoring_mask] @ rolling_weights
        mlp_prediction = matrices["validation"][scoring_mask][
            :, candidate_names.index("numerical_mlp")
        ]
        actual = validation_actual[scoring_mask]
        rolling_validation.append(
            {
                "weight_fit_weeks": f"1-{training_end}",
                "score_weeks": f"{training_end + 1}-{scoring_end}",
                "rows": int(scoring_mask.sum()),
                "weights": dict(
                    zip(candidate_names, rolling_weights.tolist(), strict=True)
                ),
                "ensemble_mae": float(np.mean(np.abs(actual - rolling_prediction))),
                "mlp_mae": float(np.mean(np.abs(actual - mlp_prediction))),
            }
        )
    results["rolling_validation"] = rolling_validation
    for split_name in ("validation", "test"):
        ensemble_prediction = matrices[split_name] @ weights
        results[split_name] = {
            "ensemble": holdout_metrics(splits[split_name], ensemble_prediction),
            "candidates": {
                name: holdout_metrics(splits[split_name], matrices[split_name][:, index])
                for index, name in enumerate(candidate_names)
            },
        }
        pl.DataFrame(
            {
                "game_id": splits[split_name]["game_id"],
                "player_id": splits[split_name]["player_id"],
                "receiving_yards": splits[split_name]["receiving_yards"],
                **{
                    name: matrices[split_name][:, index]
                    for index, name in enumerate(candidate_names)
                },
                "ensemble": ensemble_prediction,
            }
        ).write_parquet(output_directory / f"{split_name}_predictions.parquet")
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "comparison.json").write_text(json.dumps(results, indent=2))
    return results