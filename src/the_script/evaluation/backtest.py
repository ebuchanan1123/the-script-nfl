"""Fixed out-of-season WR backtesting splits."""

import polars as pl


def chronological_splits(frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
	"""Split without randomizing seasons or leaking future rows into training."""
	return {
		"train": frame.filter(pl.col("season").is_between(2016, 2023)),
		"validation": frame.filter(pl.col("season") == 2024),
		"test": frame.filter(pl.col("season") == 2025),
	}