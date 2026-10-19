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
opponent, matchup, play-calling, and injury context. An experimental feature pipeline can also join
public NFL Next Gen Stats receiving data for separation, cushion, air-yard share, catch rate, and
YAC over expectation. Model artifacts are versioned under `models/`;
raw NFL data and generated processed tables are excluded from Git and must be rebuilt or supplied to
run training/export jobs.

## Setup

```bash
uv sync --extra data --extra baseline --extra neural --extra dev
```

The project uses `nflreadpy`, not the deprecated `nfl_data_py` package.
On macOS, XGBoost also requires `brew install libomp`.

## Build Data and Train Models

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
uv run the-script evaluate-yards-challengers
uv run the-script compare-wr-models
```

`download-core` saves weekly player stats, schedules, snap counts, depth charts, injuries, play by
play, player registry, and Next Gen Stats receiving data for 2016–2025. The WR pipeline above builds
its feature table and trains/evaluates WR models. Build multi-position data and models with:

```bash
for position in QB RB TE; do
	uv run the-script build-position-features --position "$position"
	uv run the-script train-position-models --position "$position"
	uv run the-script train-position-quantiles --position "$position"
done
```

Default outputs:

```text
data/raw/player_stats.parquet
data/raw/schedules.parquet
data/raw/snap_counts.parquet
data/raw/players.parquet
data/raw/nextgen_receiving.parquet
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

The WR comparison uses a shared 2,320-game 2025 common-history cohort. Ensemble weights were
selected on the 2024 validation season; 2025 is the out-of-season test set.

| Model | Test MAE | Test RMSE | Median absolute error |
|---|---:|---:|---:|
| Recent five-game average | 21.74 | 29.52 | 16.00 |
| Ridge | 20.62 | 28.17 | 15.52 |
| XGBoost | 21.16 | **28.27** | 15.98 |
| Numerical MLP | 20.52 | 28.67 | 14.30 |
| Validation-weighted ensemble | **20.25** | 28.25 | **13.98** |

The numerical MLP uses 77 features, layers `64 -> 32 -> 1`, Huber loss, AdamW, batch size 128,
and validation-based early stopping. Among standalone model families, the MLP has the lowest MAE;
the validation-weighted ensemble has the lowest MAE, RMSE, and median absolute error in this test
cohort. The model-family report is saved to
`models/wr_model_comparison.json`; ensemble results are in `models/wr_ensemble/comparison.json`.

### Expanding-Window Validation Diagnostics

Run `uv run the-script evaluate-wr-rolling-validation` to evaluate the saved 2024 validation
predictions. It fits blend weights on weeks 1-6, 1-10, and 1-14, then scores only weeks 7-10,
11-14, and 15-18 respectively. Model and five-game-average results use the same 1,547-row cohort
with a pregame average available; inactive players are outside the snap-verified training target.
The report is written to `models/wr_ensemble/rolling_validation_diagnostics.json`.

In the current run, the rolling blend's MAE was 20.34 yards versus 22.26 for the five-game-average
baseline. A walk-forward additive bias correction reduced later-fold bias but increased MAE from
19.69 to 20.23 yards, so it is not enabled in production. This illustrates evidence-driven model
selection: use the 2024 rolling folds to select experiments, and do not tune against 2025 test
results.

### Recency-Weighted Usage Feature Screen

`uv run the-script backtest-recency-features` compares artifact-matched Ridge baselines with the
same models refit after adding five-game exponentially weighted targets, target share, and red-zone
target share. The newest prior game has weight 1.0 and each older observation is multiplied by 0.75.
This feature screen covers WR, RB, and TE; QB is deliberately excluded. In the current 2024
validation run, MAE changes were negligible (best: TE receiving yards improved by 0.021 yards;
worst: RB receiving yards worsened by 0.005 yards). This did not justify replacing the production
models. The 2025 comparison is recorded in `models/recency_backtest.json` as a once-only diagnostic,
not as a tuning target.

### Next Gen Stats Receiving Experiment

`uv run the-script download-core --start-season 2016 --end-season 2025` also downloads the public
`nflreadpy.load_nextgen_stats(stat_type="receiving")` table. It includes weekly separation, cushion,
air-yard share, catch rate, and actual/expected YAC fields, but **does not include route counts**.
Current coverage contains NGS history for about 69% of WR rows and 56% of TE rows; NGS receiving
does not supply RB records. Week-0 season aggregates are excluded, and every NGS feature uses only
prior participated games. The regular production feature lists leave these new features out pending
evidence; the opt-in screen is `uv run the-script backtest-nextgen-features`.

The current artifact-matched Ridge screen changed 2024 validation MAE by -0.014 yards for WR
receiving yards and +0.037 yards for TE receiving yards. A separately retrained WR yards ensemble
using the NGS fields changed expanding-window 2024 out-of-fold MAE from 20.260 to 20.250 yards,
while RMSE worsened by 0.023 yards and negative bias increased slightly. A full WR candidate
ensemble also showed only a 0.010-yard MAE gain in expanding-window validation. That is not a
compelling improvement, so NGS remains excluded from default training and serving artifacts. The
one-time 2025 diagnostic showed MAE down 0.142 yards but RMSE up 0.221 yards; it was not used to
select or tune the feature set.

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
| Ridge | 20.620 | **20.581** | -0.039 | 28.166 | 28.193 |
| XGBoost | 21.159 | **20.928** | -0.231 | 28.270 | **28.084** |
| Numerical MLP | **20.521** | 20.711 | +0.190 | 28.674 | 28.562 |

Opponent WR allowed history slightly improves Ridge MAE and improves XGBoost MAE/RMSE, while the
opponent-context MLP is worse on MAE and slightly better on RMSE. Effects are small and model-family
dependent. Results and separately fitted artifacts are under `models/wr_opponent/`.

## Live Pregame Prediction

After the prior week is complete, build an outcome-null upcoming row and score all opponent-context
models with:

```bash
uv run the-script predict-live-wr \
	--player "Amon-Ra St. Brown" \
	--season 2026 \
	--week 3
```

The command loads current schedules, weekly stats, snaps, the player registry, and optional Next Gen
Stats receiving history; appends the future game without target, target opportunity, or snap values;
computes shifted player/opponent history; and saves the exact inputs and predictions under
`predictions/`. Next Gen source data is cached locally with the other live sources.

For Amon-Ra St. Brown versus the Jets on September 27, 2026:

```text
Ridge:          84.4 receiving yards
XGBoost:        70.9 receiving yards
Numerical MLP:  86.5 receiving yards
```

The serving application reports the validation-weighted ensemble rather than any single model.
The example above is an archived prediction snapshot, not a current recommendation; its historical
MAE is typical absolute error, not a formal prediction interval. Model-family weights and selection
are based on the 2024 validation split. The 2025 holdout is for final reporting only and must not be
repeatedly used to tune production choices.

## Chronological Split Contract

WR evaluation uses:

```text
Train:      2016-2023
Validation: 2024
Test:       2025
```

All target-position pipelines use chronological training/validation/test separation. Rolling
features are shifted so the predicted game's outcomes are excluded, and imputers/scalers are fitted
on training rows only. The fixed WR split is 2016–2023 train, 2024 validation, and 2025 test. The
QB/RB/TE trainer uses the same chronological season split. Validation selects model settings and
blend weights; test metrics are reported as a final, held-out diagnostic.