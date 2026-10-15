import joblib
import polars as pl

from the_script.models.baseline import train_ridge
from the_script.models.preprocessing import NUMERIC_FEATURES
from the_script.models.touchdowns import train_touchdown_model
from the_script.models.train import train_wr_neural


def _training_frame() -> pl.DataFrame:
    rows = []
    for season in range(2016, 2026):
        for row_number in range(8):
            row = {
                "season": season,
                "receiving_yards": float(20 + row_number * 4),
                "targets": float(3 + row_number),
                "receptions": float(2 + row_number * 0.75),
                "receiving_touchdowns": float(row_number % 4 == 0),
            }
            row.update({feature: float(row_number + 1) for feature in NUMERIC_FEATURES})
            row["home_game"] = row_number % 2 == 0
            rows.append(row)
    return pl.DataFrame(rows)


def test_ridge_trains_and_persists_without_holdout_fitting(tmp_path) -> None:
    results = train_ridge(_training_frame(), tmp_path)

    assert results["validation"]["rows"] == 8
    assert results["test"]["rows"] == 8
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "preprocessor.joblib").exists()
    assert (tmp_path / "metrics.json").exists()


def test_numerical_mlp_trains_and_persists(tmp_path) -> None:
    results = train_wr_neural(
        _training_frame(), tmp_path, batch_size=16, max_epochs=2, patience=1
    )

    assert results["epochs_completed"] <= 2
    assert results["validation"]["rows"] == 8
    assert results["test"]["rows"] == 8
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "preprocessor.joblib").exists()
    assert (tmp_path / "training_history.json").exists()


def test_ridge_can_train_and_persist_a_receptions_model(tmp_path) -> None:
    frame = _training_frame()
    results = train_ridge(frame, tmp_path, target_column="receptions")
    model = joblib.load(tmp_path / "model.joblib")
    preprocessor = joblib.load(tmp_path / "preprocessor.joblib")

    prediction = model.predict(preprocessor.transform(frame.tail(1)))[0]

    assert results["target_column"] == "receptions"
    assert results["test"]["mae"] >= 0
    assert prediction >= 0


def test_poisson_touchdown_model_persists_probability_metrics(tmp_path) -> None:
    results = train_touchdown_model(_training_frame(), tmp_path, model_name="poisson")

    assert results["test"]["poisson_deviance"] >= 0
    assert 0 <= results["test"]["brier_score"] <= 1
    assert (tmp_path / "model.joblib").exists()