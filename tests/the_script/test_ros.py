from pathlib import Path

import numpy as np

from the_script import ros
from the_script.ros import standard_ppr_points


def test_standard_ppr_formula() -> None:
    assert standard_ppr_points(100.0, 7.0, 1.0) == 23.0


def test_neural_scorer_uses_neural_artifact(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def score(model_type: str, directory: Path, frame_path: Path) -> np.ndarray:
        calls.append((model_type, directory, frame_path))
        return np.array([12.0])

    monkeypatch.setattr(ros, "_score_artifact", score)
    result = ros._score_neural_safely(tmp_path / "neural", tmp_path / "features.parquet")

    assert result.tolist() == [12.0]
    assert calls == [("neural", tmp_path / "neural", tmp_path / "features.parquet")]
