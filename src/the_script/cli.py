"""WR-only command line workflow for The Script's first milestone."""

import argparse
import json
from pathlib import Path

import polars as pl

from the_script.data.build_dataset import build_wr_player_games_from_parquet
from the_script.data.download import download_core_data
from the_script.features.player import build_player_features_from_parquet
from the_script.models.baseline import evaluate_recent_average_from_parquet
from the_script.models.ensemble import evaluate_yards_challengers
from the_script.models.multiposition import (
    POSITION_CONFIGS,
    build_position_features_from_parquet,
    predict_live_position,
    train_position_models,
    train_position_quantiles_from_parquet,
)
from the_script.models.quantiles import train_wr_quantile_models_from_parquet
from the_script.models.touchdowns import train_wr_touchdown_models_from_parquet
from the_script.models.train import (
    compare_trained_wr_models,
    train_wr_baselines_from_parquet,
    train_wr_neural_from_parquet,
    train_wr_opponent_experiment_from_parquet,
    train_wr_opponent_neural_from_parquet,
    train_wr_volume_models_from_parquet,
)
from the_script.predict import predict_live_wr


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="the-script")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download-core")
    download.add_argument("--start-season", type=int, default=2016)
    download.add_argument("--end-season", type=int, default=2025)
    download.add_argument("--raw-directory", type=Path, default=Path("data/raw"))
    download.add_argument("--interim-directory", type=Path, default=Path("data/interim"))
    download.add_argument("--overwrite", action="store_true")
    build = commands.add_parser("build-wr-table")
    build.add_argument("--raw-directory", type=Path, default=Path("data/raw"))
    build.add_argument("--output", type=Path, default=Path("data/processed/wr_player_games.parquet"))
    features = commands.add_parser("build-player-features")
    features.add_argument(
        "--input", type=Path, default=Path("data/processed/wr_player_games.parquet")
    )
    features.add_argument(
        "--output", type=Path, default=Path("data/processed/wr_player_features.parquet")
    )
    features.add_argument(
        "--player-stats", type=Path, default=Path("data/raw/player_stats.parquet")
    )
    features.add_argument(
        "--depth-charts", type=Path, default=Path("data/raw/depth_charts.parquet")
    )
    features.add_argument(
        "--play-by-play", type=Path, default=Path("data/raw/play_by_play.parquet")
    )
    features.add_argument(
        "--schedules", type=Path, default=Path("data/raw/schedules.parquet")
    )
    features.add_argument(
        "--injuries", type=Path, default=Path("data/raw/injuries.parquet")
    )
    inspect = commands.add_parser("inspect-wr-table")
    inspect.add_argument("--input", type=Path, default=Path("data/processed/wr_player_games.parquet"))
    inspect.add_argument("--rows", type=int, default=10)
    baseline = commands.add_parser("evaluate-recent-average")
    baseline.add_argument(
        "--input", type=Path, default=Path("data/processed/wr_player_features.parquet")
    )
    baseline.add_argument(
        "--output", type=Path, default=Path("models/wr_recent_average/metrics.json")
    )
    train_baselines = commands.add_parser("train-wr-baselines")
    train_baselines.add_argument(
        "--input", type=Path, default=Path("data/processed/wr_player_features.parquet")
    )
    train_baselines.add_argument("--models-directory", type=Path, default=Path("models"))
    neural = commands.add_parser("train-wr-neural")
    neural.add_argument(
        "--input", type=Path, default=Path("data/processed/wr_player_features.parquet")
    )
    neural.add_argument("--output", type=Path, default=Path("models/wr_neural"))
    neural.add_argument("--batch-size", type=int, default=128)
    neural.add_argument("--max-epochs", type=int, default=100)
    neural.add_argument("--patience", type=int, default=10)
    compare = commands.add_parser("compare-wr-models")
    compare.add_argument("--models-directory", type=Path, default=Path("models"))
    opponent = commands.add_parser("train-wr-opponent-experiment")
    opponent.add_argument(
        "--input", type=Path, default=Path("data/processed/wr_player_features.parquet")
    )
    opponent.add_argument("--models-directory", type=Path, default=Path("models"))
    opponent_neural = commands.add_parser("train-wr-opponent-neural")
    opponent_neural.add_argument(
        "--input", type=Path, default=Path("data/processed/wr_player_features.parquet")
    )
    opponent_neural.add_argument("--models-directory", type=Path, default=Path("models"))
    volume = commands.add_parser("train-wr-volume-models")
    volume.add_argument(
        "--input", type=Path, default=Path("data/processed/wr_player_features.parquet")
    )
    volume.add_argument("--models-directory", type=Path, default=Path("models"))
    touchdowns = commands.add_parser("train-wr-touchdowns")
    touchdowns.add_argument(
        "--input", type=Path, default=Path("data/processed/wr_player_features.parquet")
    )
    touchdowns.add_argument("--models-directory", type=Path, default=Path("models"))
    quantiles = commands.add_parser("train-wr-quantiles")
    quantiles.add_argument(
        "--input", type=Path, default=Path("data/processed/wr_player_features.parquet")
    )
    quantiles.add_argument("--models-directory", type=Path, default=Path("models"))
    ensemble = commands.add_parser("evaluate-yards-challengers")
    ensemble.add_argument(
        "--input", type=Path, default=Path("data/processed/wr_player_features.parquet")
    )
    ensemble.add_argument("--models-directory", type=Path, default=Path("models"))
    build_positions = commands.add_parser("build-position-features")
    build_positions.add_argument(
        "--position", choices=sorted(POSITION_CONFIGS), required=True
    )
    build_positions.add_argument("--raw-directory", type=Path, default=Path("data/raw"))
    build_positions.add_argument(
        "--output-directory", type=Path, default=Path("data/processed")
    )
    train_positions = commands.add_parser("train-position-models")
    train_positions.add_argument(
        "--position", choices=sorted(POSITION_CONFIGS), required=True
    )
    train_positions.add_argument(
        "--input-directory", type=Path, default=Path("data/processed")
    )
    train_positions.add_argument("--models-directory", type=Path, default=Path("models"))
    train_position_quantiles = commands.add_parser("train-position-quantiles")
    train_position_quantiles.add_argument(
        "--position", choices=sorted(POSITION_CONFIGS), required=True
    )
    train_position_quantiles.add_argument(
        "--input-directory", type=Path, default=Path("data/processed")
    )
    train_position_quantiles.add_argument("--models-directory", type=Path, default=Path("models"))
    predict_position = commands.add_parser("predict-live-position")
    predict_position.add_argument(
        "--position", choices=sorted(POSITION_CONFIGS), required=True
    )
    predict_position.add_argument("--player", required=True)
    predict_position.add_argument("--team")
    predict_position.add_argument("--season", type=int, required=True)
    predict_position.add_argument("--week", type=int, required=True)
    predict_position.add_argument(
        "--processed-directory", type=Path, default=Path("data/processed")
    )
    predict_position.add_argument("--models-directory", type=Path, default=Path("models"))
    predict = commands.add_parser("predict-live-wr")
    predict.add_argument("--player", required=True)
    predict.add_argument("--season", type=int, required=True)
    predict.add_argument("--week", type=int, required=True)
    predict.add_argument("--team")
    predict.add_argument("--exclude-current-season-data", action="store_true")
    predict.add_argument("--models-directory", type=Path, default=Path("models"))
    predict.add_argument("--output-directory", type=Path, default=Path("predictions"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "download-core":
        if args.end_season < args.start_season:
            raise ValueError("end-season must be greater than or equal to start-season")
        paths = download_core_data(
            args.raw_directory,
            args.interim_directory,
            list(range(args.start_season, args.end_season + 1)),
            overwrite=args.overwrite,
        )
        print("\n".join(f"{name}: {path}" for name, path in paths.items()))
    elif args.command == "build-wr-table":
        frame = build_wr_player_games_from_parquet(args.raw_directory, args.output)
        print(f"wrote {frame.height:,} rows to {args.output}")
    elif args.command == "build-player-features":
        frame = build_player_features_from_parquet(
            args.input,
            args.output,
            player_stats_path=args.player_stats,
            depth_charts_path=args.depth_charts,
            play_by_play_path=args.play_by_play,
            schedules_path=args.schedules,
            injuries_path=args.injuries,
        )
        print(f"wrote {frame.height:,} rows with shifted player features to {args.output}")
    elif args.command == "inspect-wr-table":
        with pl.Config(tbl_rows=args.rows, tbl_cols=-1, tbl_width_chars=180):
            print(pl.read_parquet(args.input).head(args.rows))
    elif args.command == "evaluate-recent-average":
        results = evaluate_recent_average_from_parquet(args.input, args.output)
        print(json.dumps(results, indent=2))
    elif args.command == "train-wr-baselines":
        results = train_wr_baselines_from_parquet(args.input, args.models_directory)
        print(json.dumps(results, indent=2))
    elif args.command == "train-wr-neural":
        results = train_wr_neural_from_parquet(
            args.input,
            args.output,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        print(json.dumps(results, indent=2))
    elif args.command == "compare-wr-models":
        results = compare_trained_wr_models(args.models_directory)
        print(json.dumps(results["common_test_cohort"], indent=2))
        print(f"lowest test MAE: {results['lowest_test_mae']}")
        print(f"lowest test RMSE: {results['lowest_test_rmse']}")
    elif args.command == "train-wr-opponent-experiment":
        results = train_wr_opponent_experiment_from_parquet(
            args.input, args.models_directory
        )
        for experiment_name, experiment in results.items():
            print(experiment_name)
            for model_name, metrics in experiment.items():
                print(
                    f"  {model_name}: MAE={metrics['common_test_mae']:.3f}, "
                    f"RMSE={metrics['common_test_rmse']:.3f}"
                )
    elif args.command == "train-wr-opponent-neural":
        results = train_wr_opponent_neural_from_parquet(args.input, args.models_directory)
        print(json.dumps(results, indent=2))
    elif args.command == "train-wr-volume-models":
        results = train_wr_volume_models_from_parquet(args.input, args.models_directory)
        for target_name, models in results.items():
            print(target_name)
            for model_name, metrics in models.items():
                test = metrics["test"]
                print(f"  {model_name}: MAE={test['mae']:.3f}, RMSE={test['rmse']:.3f}")
    elif args.command == "train-wr-touchdowns":
        results = train_wr_touchdown_models_from_parquet(args.input, args.models_directory)
        print(f"selected model: {results['selected_model']}")
        for model_name, metrics in results["models"].items():
            test = metrics["test"]
            print(
                f"  {model_name}: deviance={test['poisson_deviance']:.3f}, "
                f"Brier={test['brier_score']:.3f}, log loss={test['log_loss']:.3f}"
            )
    elif args.command == "train-wr-quantiles":
        results = train_wr_quantile_models_from_parquet(args.input, args.models_directory)
        for split_name in ("validation", "test"):
            metrics = results[split_name]
            print(
                f"{split_name}: coverage={metrics['interval_coverage']:.3f}, "
                f"width={metrics['mean_interval_width']:.1f}, "
                f"median pinball={metrics['q50_pinball_loss']:.3f}"
            )
    elif args.command == "evaluate-yards-challengers":
        results = evaluate_yards_challengers(
            pl.read_parquet(args.input), args.models_directory
        )
        print(json.dumps(results["weights"], indent=2))
        for split_name in ("validation", "test"):
            metrics = results[split_name]["ensemble"]
            print(
                f"{split_name}: MAE={metrics['mae']:.3f}, "
                f"RMSE={metrics['rmse']:.3f}"
            )
    elif args.command == "build-position-features":
        frame = build_position_features_from_parquet(
            args.raw_directory, args.output_directory, args.position
        )
        print(
            f"wrote {frame.height:,} {args.position} rows to "
            f"{args.output_directory / f'{args.position.lower()}_player_features.parquet'}"
        )
    elif args.command == "train-position-models":
        input_path = (
            args.input_directory / f"{args.position.lower()}_player_features.parquet"
        )
        results = train_position_models(
            pl.read_parquet(input_path), args.models_directory, args.position
        )
        for target, values in results["outcomes"].items():
            selected = values["selected_model"]
            metrics = values["weighted_blend"]["test"] if selected == "weighted_blend" else values["models"][selected]["test"]
            print(
                f"{target}: {selected} MAE={metrics['mae']:.3f}, "
                f"RMSE={metrics['rmse']:.3f}"
            )
    elif args.command == "train-position-quantiles":
        results = train_position_quantiles_from_parquet(
            args.input_directory, args.models_directory, args.position
        )
        print(json.dumps(results, indent=2))
    elif args.command == "predict-live-position":
        result = predict_live_position(
            args.player,
            args.position,
            args.season,
            args.week,
            team=args.team,
            processed_directory=args.processed_directory,
            models_directory=args.models_directory,
        )
        print(json.dumps(result, indent=2))
    else:
        result = predict_live_wr(
            args.player,
            args.season,
            args.week,
            team=args.team,
            exclude_current_season_data=args.exclude_current_season_data,
            models_directory=args.models_directory,
            output_directory=args.output_directory,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()