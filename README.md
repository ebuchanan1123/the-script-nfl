# The Script

> The Script turns historical NFL data into matchup-specific player forecasts. It combines leakage-aware feature engineering with Ridge, XGBoost, and neural models, then delivers WR, QB, RB, and TE projections through an interactive React dashboard or lightweight static JSON releases.

The goal is not just to train a model—it is to build an end-to-end forecasting system: prepare trustworthy data, model player outcomes in football context, evaluate on future seasons, and make the results easy to explore.

## Full Application

The production-style application is split into a FastAPI prediction service and a React/Vite
frontend. Start each process from the repository root:

```bash
# Terminal 1
uv sync --extra data --extra baseline --extra neural --extra api --extra dev
uv run the-script-api

# Terminal 2
npm --prefix frontend install
npm --prefix frontend run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173`. The API runs at `http://127.0.0.1:8000`, with interactive
documentation at `/docs`. Set `VITE_API_URL` when the API is hosted elsewhere.

The UI supports WR, QB, RB, and TE search, matchup resolution, forecasts, rankings, recent trends,
injury context, prediction history, and model metrics. FastAPI mode runs inference on request;
static mode selects forecasts from a versioned JSON release. Static forecast exports are generated
with `uv run python -m the_script.static_export`; incomplete exports never replace the last release.

## Prediction Contract

```text
Prediction time: 24 hours before kickoff
Target: receiving_yards
Positions: WR, QB, RB, TE
Population: players who participated in the game
Season type: regular season only
Seasons: 2016-2025
```

Features are cutoff-aware: the model uses only information available before its target game.
Feature groups include player form and role, team and opponent tendencies, game script, injuries,
matchup context, and betting/implied-score inputs when available. Static releases record source
freshness and preserve the last successful release when refreshes fail.

## Current Milestone

The project includes trained WR receiving and QB/RB/TE outcome models, a forecast API, ranking
snapshots, and the interactive application. The WR player-game pipeline below documents the
original audited data contract and remains the basis of the receiving-yards model.

The accepted base-table milestone does six things:

1. Download weekly player statistics for 2016-2025.
2. Download schedules for 2016-2025.
3. Save both unchanged sources as Parquet.
4. Build one canonical row per regular-season WR-game.
5. Add opponent, game date, home/away, and rest days.
6. Print ten rows for manual inspection.

The multi-position feature pipeline combines snap-verified player history with leakage-safe team,
opponent, matchup, play-calling, and injury context. Model artifacts are versioned under `models/`;
raw NFL data and generated processed tables are excluded from Git and must be rebuilt or supplied to
run training/export jobs.

## Setup

```bash
uv sync --extra data --extra baseline --extra neural --extra dev
```

The project uses `nflreadpy`, not the deprecated `nfl_data_py` package.
On macOS, XGBoost also requires `brew install libomp`.

## Build The First Dataset

```bash
uv run the-script download-core --start-season 2016 --end-season 2025
uv run the-script build-wr-table
uv run the-script inspect-wr-table --rows 10
uv run the-script build-player-features
uv run the-script evaluate-recent-average
uv run the-script train-wr-baselines
uv run the-script train-wr-neural
uv run the-script compare-wr-models
uv run the-script train-wr-opponent-neural
uv run the-script train-wr-opponent-experiment
```

Default outputs:

```text
data/raw/player_stats.parquet
data/raw/schedules.parquet
data/raw/snap_counts.parquet
data/raw/players.parquet
data/interim/source_schemas.json
data/processed/wr_player_games.parquet
data/processed/wr_player_features.parquet
```

The build fails if canonical identifiers are null, `(game_id, player_id)` is duplicated, postseason
rows remain, a team cannot be matched to the scheduled home or away team, or the source season falls
outside 2016-2025.

## Canonical WR-Game Contract

```text
game_id
game_date
season
week
player_id
team
opponent
home_game
days_rest
position
receiving_yards
targets
receptions
receiving_touchdowns
snap_share
```

Snap counts use PFR player IDs, while weekly statistics use GSIS IDs. The player registry bridges
those IDs. Only WR rows with positive offensive snaps are retained; unmatched rows are excluded
rather than treated as participants.

## Shifted Player Features

`wr_player_features.parquet` adds:

```text
yards_last_game
yards_avg_last_3
yards_avg_last_5
targets_avg_last_3
targets_avg_last_5
receptions_avg_last_5
snap_share_last_game
snap_share_avg_last_3
season_yards_per_game_before_game
games_played_before_game
```

Every outcome/opportunity window applies `shift(1)` before its rolling calculation. Tests mutate the
current target and truncate future source rows to prove historical features remain unchanged.

## WR Model Results

The recent-average baseline predicts `receiving_yards = yards_avg_last_5`. It has no fitted
parameters and uses the fixed out-of-season split.

All models use the same 2,320-game 2025 common-history cohort below.

| Model | Test MAE | Test RMSE | Median absolute error |
|---|---:|---:|---:|
| Recent five-game average | 21.74 | 29.52 | 16.00 |
| Ridge | 21.19 | 28.37 | 16.58 |
| XGBoost | 21.30 | **28.32** | 16.47 |
| Numerical MLP | **20.24** | 28.91 | **13.91** |

The numerical MLP uses 12 features, layers `64 -> 32 -> 1`, Huber loss, AdamW, batch size 128,
and validation-based early stopping. It stopped after 30 epochs. XGBoost has the lowest test RMSE,
while the MLP has the lowest MAE and median absolute error. The consolidated report is saved to
`models/wr_model_comparison.json`; each model directory also contains its own preprocessing and
metrics artifacts.

## Opponent WR Allowed Experiment

The opponent experiment aggregates all snap-verified WR production faced by each defense in one
game, shifts the defense timeline by one game, and adds prior-game, rolling 3/5-game, and
season-to-date averages for:

```text
WR receiving yards allowed
WR PPR points allowed = receptions + receiving_yards / 10 + 6 * receiving_touchdowns
```

No production from the predicted game enters these features. The MLP is trained in a separate
process from XGBoost on macOS to avoid mixing PyTorch and OpenMP native runtimes.

Common 2025 cohort results:

| Model | Base MAE | + Opponent MAE | Change | Base RMSE | + Opponent RMSE |
|---|---:|---:|---:|---:|---:|
| Ridge | 21.194 | **21.089** | -0.105 | 28.370 | **28.364** |
| XGBoost | 21.299 | **21.298** | -0.001 | **28.320** | 28.390 |
| Numerical MLP | 20.236 | **20.200** | -0.036 | **28.909** | 28.930 |

Opponent WR allowed history provides a small MAE improvement for Ridge and the MLP, but does not
materially improve XGBoost. This is useful signal, not yet a large effect. Results and separately
fitted artifacts are under `models/wr_opponent/`.

## Live Pregame Prediction

After the prior week is complete, build an outcome-null upcoming row and score all opponent-context
models with:

```bash
uv run the-script predict-live-wr \
	--player "Amon-Ra St. Brown" \
	--season 2026 \
	--week 3
```

The command downloads current schedules, weekly stats, snaps, and the player registry; appends the
future game without target, target opportunity, or snap values; computes shifted player/opponent
history; and saves the exact inputs and predictions under `predictions/`.

For Amon-Ra St. Brown versus the Jets on September 27, 2026:

```text
Ridge:          84.4 receiving yards
XGBoost:        70.9 receiving yards
Numerical MLP:  86.5 receiving yards
```

The MLP is the primary estimate because it has the lowest 2025 MAE. Its historical MAE is about
20.2 yards, which describes typical error, not a formal prediction interval. These research models
were fit on 2016-2023 and evaluated on 2024/2025; a production release should refit the selected
architecture through 2025 after the evaluation protocol is frozen.

## Chronological Split Contract

Future modeling will use:

```text
Train:      2016-2023
Validation: 2024
Test:       2025
```

Rolling features, imputers, scalers, baselines, and neural networks are deliberately deferred until
the canonical table is accepted.