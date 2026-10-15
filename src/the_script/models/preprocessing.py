"""Training-only numerical preprocessing for WR receiving-yards models."""

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from the_script.features.advanced import ADVANCED_FEATURE_COLUMNS, QB_CONTEXT_COLUMNS
from the_script.features.injuries import INJURY_FEATURE_COLUMNS
from the_script.features.opponent import OPPONENT_FEATURE_COLUMNS
from the_script.features.role import ROLE_FEATURE_COLUMNS
from the_script.features.tendencies import TENDENCY_FEATURE_COLUMNS

TARGET_COLUMN = "receiving_yards"
BASE_NUMERIC_FEATURES = [
    "yards_last_game",
    "yards_avg_last_3",
    "yards_avg_last_5",
    "yards_avg_last_16",
    "targets_avg_last_3",
    "targets_avg_last_5",
    "targets_avg_last_16",
    "target_share_last_game",
    "target_share_avg_last_3",
    "target_share_avg_last_5",
    "target_share_avg_last_16",
    "air_yards_share_last_game",
    "air_yards_share_avg_last_3",
    "air_yards_share_avg_last_5",
    "wopr_last_game",
    "wopr_avg_last_3",
    "wopr_avg_last_5",
    "receptions_avg_last_5",
    "receptions_avg_last_16",
    "snap_share_last_game",
    "snap_share_avg_last_3",
    "season_yards_per_game_before_game",
    "games_played_before_game",
    "home_game",
    "days_rest",
    *ROLE_FEATURE_COLUMNS,
    *TENDENCY_FEATURE_COLUMNS,
    *INJURY_FEATURE_COLUMNS,
    *ADVANCED_FEATURE_COLUMNS,
]
NUMERIC_FEATURES = [*BASE_NUMERIC_FEATURES, *OPPONENT_FEATURE_COLUMNS]
LONG_HORIZON_FEATURES = [
    "yards_avg_last_16",
    "targets_avg_last_16",
    "target_share_avg_last_16",
    "receptions_avg_last_16",
]
CORE_NUMERIC_FEATURES = [
    feature
    for feature in NUMERIC_FEATURES
    if feature not in {*INJURY_FEATURE_COLUMNS, *ADVANCED_FEATURE_COLUMNS}
]
YARDS_RIDGE_FEATURES = [
    *CORE_NUMERIC_FEATURES,
    *INJURY_FEATURE_COLUMNS,
    *QB_CONTEXT_COLUMNS,
]
YARDS_XGBOOST_FEATURES = [*CORE_NUMERIC_FEATURES, *QB_CONTEXT_COLUMNS]
YARDS_NEURAL_FEATURES = [
    feature for feature in NUMERIC_FEATURES if feature not in LONG_HORIZON_FEATURES
]

FORBIDDEN_CURRENT_GAME_FEATURES = {
    "receiving_yards",
    "targets",
    "receptions",
    "receiving_touchdowns",
    "target_share",
    "air_yards_share",
    "wopr",
    "snap_share",
}


@dataclass
class NumericPreprocessor:
    """Median-impute and standardize using statistics fitted on training rows only."""

    features: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler

    @classmethod
    def fit(cls, frame: pl.DataFrame, features: list[str] | None = None) -> "NumericPreprocessor":
        selected = NUMERIC_FEATURES if features is None else features
        forbidden = sorted(set(selected).intersection(FORBIDDEN_CURRENT_GAME_FEATURES))
        if forbidden:
            raise ValueError(f"Current-game columns cannot be model features: {', '.join(forbidden)}")
        missing = sorted(set(selected).difference(frame.columns))
        if missing:
            raise ValueError(f"Training data is missing features: {', '.join(missing)}")
        values = frame.select(selected).to_numpy().astype(np.float64)
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        scaler.fit(imputer.fit_transform(values))
        return cls(features=list(selected), imputer=imputer, scaler=scaler)

    def transform(self, frame: pl.DataFrame) -> np.ndarray:
        values = frame.select(self.features).to_numpy().astype(np.float64)
        return self.scaler.transform(self.imputer.transform(values)).astype(np.float32)