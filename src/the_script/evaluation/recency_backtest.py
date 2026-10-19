"""Controlled validation experiment for recency-weighted WR/RB/TE usage features."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.base import clone

from the_script.evaluation.backtest import chronological_splits
from the_script.evaluation.metrics import regression_metrics
from the_script.features.nextgen import NGS_RECEIVING_FEATURE_COLUMNS
from the_script.models.multiposition import POSITION_CONFIGS, position_feature_columns
from the_script.models.preprocessing import NumericPreprocessor

RECENCY_FEATURES = (
    "targets_recency_5",
    "target_share_recency_5",
    "red_zone_target_share_recency_5",
)
NEXTGEN_FEATURES = tuple(NGS_RECEIVING_FEATURE_COLUMNS)


def _report_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    errors = predicted - actual
    return {
        "rows": len(actual),
        **regression_metrics(pl.Series(actual), pl.Series(predicted)),
        "bias": float(errors.mean()),
    }


def _ridge_experiment(
    frame: pl.DataFrame,
    target: str,
    artifact: Path,
    available_features: list[str],
    added_features: tuple[str, ...],
    variant_name: str,
) -> dict[str, Any]:
    model = joblib.load(artifact / "model.joblib")
    baseline_preprocessor = joblib.load(artifact / "preprocessor.joblib")
    baseline_features = list(baseline_preprocessor.features)
    absent = sorted(set(baseline_features).difference(frame.columns))
    if absent:
        raise ValueError(f"{artifact} baseline inputs absent from rebuilt table: {absent}")
    extras = [feature for feature in added_features if feature in available_features]
    if len(extras) != len(added_features):
        raise ValueError(f"Missing required {variant_name} features: {sorted(set(added_features) - set(extras))}")
    candidate_features = list(dict.fromkeys([*baseline_features, *extras]))
    splits = chronological_splits(frame)
    output: dict[str, Any] = {
        "baseline_features": baseline_features,
        "added_features": [feature for feature in candidate_features if feature not in baseline_features],
        "splits": {},
    }
    for name in ("validation", "test"):
        data = splits[name]
        actual = data[target].to_numpy().astype(float)
        predictions = {}
        for variant, features in (("baseline", baseline_features), (variant_name, candidate_features)):
            preprocessor = NumericPreprocessor.fit(splits["train"], features)
            estimator = clone(model)
            estimator.fit(preprocessor.transform(splits["train"]), splits["train"][target].to_numpy())
            predicted = np.clip(estimator.predict(preprocessor.transform(data)), 0.0, None)
            predictions[variant] = predicted
        output["splits"][name] = {
            "baseline": _report_metrics(actual, predictions["baseline"]),
            variant_name: _report_metrics(actual, predictions[variant_name]),
            f"mae_delta_{variant_name}_minus_baseline": float(
                np.mean(np.abs(predictions[variant_name] - actual))
                - np.mean(np.abs(predictions["baseline"] - actual))
            ),
        }
    return output


def run_recency_backtest(
    *,
    processed_directory: Path = Path("data/processed"),
    models_directory: Path = Path("models"),
    positions: tuple[str, ...] = ("WR", "RB", "TE"),
    output_path: Path | None = Path("models/recency_backtest.json"),
) -> dict[str, Any]:
    """Compare an artifact-matched Ridge baseline with only three new recency inputs.

    Model choices/hyperparameters come from the committed Ridge artifacts. Models are
    refit on 2016-2023 and evaluated on 2024 validation. The 2025 test split is only
    reported for a final, once-only check; it must not be used to choose features.
    Production model artifacts are never overwritten.
    """
    results: dict[str, Any] = {
        "protocol": "Ridge-only controlled ablation; same artifact features and hyperparameters; add targets, target-share, and red-zone-target-share 5-game exponential recency features; chronological train 2016-2023, validation 2024, test 2025.",
        "recency_decay": 0.75,
        "positions": {},
    }
    for position in positions:
        position = position.upper()
        if position not in {"WR", "RB", "TE"}:
            raise ValueError("Recency ablation supports WR, RB, and TE only; QB is intentionally excluded")
        path = processed_directory / f"{position.lower()}_player_features.parquet"
        frame = pl.read_parquet(path)
        missing = sorted(set(RECENCY_FEATURES).difference(frame.columns))
        if missing:
            raise ValueError(f"{path} lacks recency features: {missing}")
        features = position_feature_columns(position) if position != "WR" else frame.columns
        if position == "WR":
            outputs = {
                "receiving_yards": models_directory / "wr_opponent" / "ridge",
                "targets": models_directory / "wr_volume" / "targets" / "ridge",
                "receptions": models_directory / "wr_volume" / "receptions" / "ridge",
            }
        else:
            outputs = {
                outcome: models_directory / position.lower() / outcome / "ridge"
                for outcome in POSITION_CONFIGS[position].outcomes
            }
        position_result = {}
        for target, artifact in outputs.items():
            if target not in frame.columns:
                raise ValueError(f"{path} does not contain target {target}")
            position_result[target] = _ridge_experiment(
                frame, target, artifact, features, RECENCY_FEATURES, "recency"
            )
        results["positions"][position] = position_result
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2) + "\n")
    return results


def run_nextgen_backtest(
    *,
    processed_directory: Path = Path("data/processed"),
    models_directory: Path = Path("models"),
    positions: tuple[str, ...] = ("WR", "TE"),
    output_path: Path | None = Path("models/nextgen_backtest.json"),
) -> dict[str, Any]:
    """Screen NGS receiving-history features for positions present in NGS data."""
    results: dict[str, Any] = {
        "protocol": "Ridge-only controlled ablation; baseline feature set and hyperparameters are fixed by saved artifacts; add prior-game NGS receiving metrics; train 2016-2023, validation 2024, test 2025. Test metrics are for final diagnostic only.",
        "source": "NFL Next Gen Stats receiving weekly player data (2016-2025); week 0 season summaries excluded from feature history.",
        "positions": {},
    }
    for position in positions:
        position = position.upper()
        if position not in {"WR", "TE"}:
            raise ValueError("Next Gen receiving data supports the WR and TE backtest only")
        path = processed_directory / f"{position.lower()}_player_features.parquet"
        frame = pl.read_parquet(path)
        if position == "WR":
            outputs = {
                "receiving_yards": models_directory / "wr_opponent" / "ridge",
                "targets": models_directory / "wr_volume" / "targets" / "ridge",
                "receptions": models_directory / "wr_volume" / "receptions" / "ridge",
            }
        else:
            outputs = {
                target: models_directory / "te" / target / "ridge"
                for target in POSITION_CONFIGS["TE"].outcomes
            }
        results["positions"][position] = {
            target: _ridge_experiment(
                frame, target, artifact, frame.columns, NEXTGEN_FEATURES, "nextgen"
            )
            for target, artifact in outputs.items()
        }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2) + "\n")
    return results
