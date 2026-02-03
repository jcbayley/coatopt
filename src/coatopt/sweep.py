#!/usr/bin/env python3
import argparse
import configparser
import tempfile
from pathlib import Path

import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback

from coatopt.run import run_experiment


def create_trial_config(
    base_config_path: str, trial: optuna.Trial, sweep_params: dict
) -> str:
    """Create a temporary config with trial parameters.

    Args:
        base_config_path: Path to base configuration file
        trial: Optuna trial object
        sweep_params: Dictionary of parameters to sweep

    Returns:
        Path to temporary config file for this trial
    """
    parser = configparser.ConfigParser()
    parser.read(base_config_path)

    # Suggest parameters based on sweep config
    for param_name, param_config in sweep_params.items():
        section, key = param_name.rsplit(".", 1)
        param_type = param_config["type"]

        if param_type == "float":
            value = trial.suggest_float(
                param_name,
                param_config["min"],
                param_config["max"],
                log=param_config.get("log", False),
            )
        elif param_type == "int":
            value = trial.suggest_int(
                param_name, param_config["min"], param_config["max"]
            )
        elif param_type == "categorical":
            value = trial.suggest_categorical(param_name, param_config["choices"])
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

        parser.set(section, key, str(value))

    # Update run_name to include trial number
    if parser.has_option("general", "run_name"):
        base_run_name = parser.get("general", "run_name")
        parser.set("general", "run_name", f"{base_run_name}_trial{trial.number}")
    else:
        parser.set("general", "run_name", f"trial{trial.number}")

    # Save trial config to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
        parser.write(f)
        trial_config_path = f.name

    return trial_config_path


def objective(
    trial: optuna.Trial,
    base_config_path: str,
    sweep_params: dict,
    metric: str,
    direction: str,
):
    """Optuna objective function.

    Args:
        trial: Optuna trial object
        base_config_path: Path to base config
        sweep_params: Parameters to sweep
        metric: MLflow metric to optimize
        direction: 'maximize' or 'minimize'

    Returns:
        Metric value for this trial
    """
    # Create config with trial parameters
    trial_config = create_trial_config(base_config_path, trial, sweep_params)

    try:
        # Run experiment (already logs to MLflow)
        run_experiment(trial_config)

        # Get metric from MLflow
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[
                mlflow.get_experiment_by_name(
                    client.get_run(
                        mlflow.last_active_run().info.run_id
                    ).info.experiment_id
                ).experiment_id
            ],
            order_by=[f"attributes.start_time DESC"],
            max_results=1,
        )

        if not runs:
            raise optuna.TrialPruned()

        run = runs[0]

        # Try to get the metric
        if metric in run.data.metrics:
            metric_value = run.data.metrics[metric]
        else:
            # If metric not found, try to get from metric history
            metric_history = client.get_metric_history(run.info.run_id, metric)
            if not metric_history:
                print(f"Warning: Metric '{metric}' not found for trial {trial.number}")
                raise optuna.TrialPruned()
            metric_value = metric_history[-1].value

        # Log trial info to Optuna's storage in MLflow
        mlflow.log_params({f"trial_{k}": v for k, v in trial.params.items()})
        mlflow.log_metric("trial_number", trial.number)

        return metric_value

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.TrialPruned()
    finally:
        # Clean up temporary config
        Path(trial_config).unlink(missing_ok=True)


def run_sweep(config_path: str, n_trials: int = 50, study_name: str = None):
    """Run hyperparameter sweep with Optuna.

    Args:
        config_path: Path to base configuration file
        n_trials: Number of trials to run
        study_name: Optional name for the study
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    parser = configparser.ConfigParser()
    parser.read(config_path)

    if "sweep" not in parser.sections():
        raise ValueError(
            "Config must have a [sweep] section. See example in config_sb3_simple.ini"
        )

    # Parse sweep parameters
    sweep_params = {}
    for key, value in parser["sweep"].items():
        if key.startswith("param_"):
            param_name = key.replace("param_", "", 1)
            try:
                sweep_params[param_name] = eval(value)
            except Exception as e:
                raise ValueError(f"Error parsing sweep parameter '{key}': {e}")

    if not sweep_params:
        raise ValueError(
            "No sweep parameters defined. Add 'param_*' entries to [Sweep] section."
        )

    # Get sweep configuration
    metric = parser.get("sweep", "metric", fallback="eval/mean_reward")
    direction = parser.get("sweep", "direction", fallback="maximize")

    if study_name is None:
        study_name = f"sweep_{config_path.stem}"

    print(f"Starting Optuna sweep: {study_name}")
    print(f"Optimizing metric: {metric} ({direction})")
    print(f"Number of trials: {n_trials}")
    print(f"Parameters to sweep:")
    for param_name, param_config in sweep_params.items():
        print(f"  - {param_name}: {param_config}")
    print()

    # Create study
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        load_if_exists=True,  # Allow resuming studies
    )

    # Run optimization
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(), metric_name=metric
    )

    study.optimize(
        lambda trial: objective(
            trial, str(config_path), sweep_params, metric, direction
        ),
        n_trials=n_trials,
        callbacks=[mlflow_callback],
        show_progress_bar=True,
    )

    # Print results
    print("\n" + "=" * 80)
    print("SWEEP RESULTS")
    print("=" * 80)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best {metric}: {study.best_value:.6f}")
    print("\nBest parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    print("\nTop 5 trials:")
    for i, trial in enumerate(
        sorted(
            study.trials,
            key=lambda t: t.value or float("-inf"),
            reverse=(direction == "maximize"),
        )[:5]
    ):
        if trial.value is not None:
            print(f"  {i+1}. Trial {trial.number}: {trial.value:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hyperparameter sweep with Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base configuration file with [Sweep] section",
    )
    parser.add_argument(
        "--n-trials", type=int, default=50, help="Number of trials to run (default: 50)"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optional study name (default: sweep_<config_name>)",
    )

    args = parser.parse_args()
    run_sweep(args.config, args.n_trials, args.study_name)
