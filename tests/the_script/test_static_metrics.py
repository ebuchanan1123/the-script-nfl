import math

import pytest

from the_script.static_export import _public_metrics, _write_json


def test_public_metrics_excludes_estimator_nan_configuration(tmp_path):
    payload = {"test": {"mae": 20.0}, "parameters": {"missing": math.nan}}
    _write_json(tmp_path / "metrics.json", _public_metrics(payload))
    assert "parameters" not in (tmp_path / "metrics.json").read_text()


def test_actual_nonfinite_metrics_are_still_rejected(tmp_path):
    with pytest.raises(ValueError):
        _write_json(tmp_path / "metrics.json", _public_metrics({"mae": math.nan}))