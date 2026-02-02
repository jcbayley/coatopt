#!/usr/bin/env python3
import argparse
import configparser
from pathlib import Path
import shutil
import warnings
from datetime import datetime
import mlflow


def run_experiment(config_path: str):
    """Run experiment based on config file.

    Handles all MLflow setup, directory creation, and dispatches to algorithm-specific training.

    Args:
        config_path: Path to INI configuration file
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Read config
    parser = configparser.ConfigParser()
    parser.read(config_path)

    # Determine algorithm from section names
    algorithm_sections = {'sb3_discrete', 'sb3_discrete_lstm', 'sb3_dqn', 'sb3_simple', 'morl', 'nsga2', 'hppo'}
    algorithm = None
    for section in parser.sections():
        if section.lower() in algorithm_sections:
            algorithm = section.lower()
            break

    if algorithm is None:
        raise ValueError(f"Config must have one of these algorithm sections: {algorithm_sections}")

    print(f"Running algorithm: {algorithm}")

    # ===== Common Setup (shared across all algorithms) =====

    # [General] section
    base_save_dir = parser.get('General', 'save_dir')
    run_name = parser.get('General', 'run_name', fallback='')

    # Get or generate experiment name (problem definition)
    experiment_name = parser.get('General', 'experiment_name', fallback=None)

    # [Data] section - read for experiment name generation
    n_layers = parser.getint('Data', 'n_layers')
    min_thickness = parser.getfloat('Data', 'min_thickness', fallback=0.1)
    max_thickness = parser.getfloat('Data', 'max_thickness', fallback=0.5)

    # Auto-generate experiment name if not specified: e.g., "20layer-0.1-0.5"
    if not experiment_name:
        experiment_name = f"{n_layers}layer-{min_thickness:.1f}-{max_thickness:.1f}"

    print(f"MLflow experiment: {experiment_name}")

    # Create run directory: YYYYMMDD-algorithm-runname
    date_str = datetime.now().strftime("%Y%m%d")
    if run_name:
        run_dir_name = f"{date_str}-{algorithm}-{run_name}"
    else:
        run_dir_name = f"{date_str}-{algorithm}"

    save_dir = Path(base_save_dir) / run_dir_name

    # Check if directory exists and warn
    if save_dir.exists():
        warnings.warn(
            f"\nWARNING: Run directory already exists: {save_dir}\n"
            f"    This will overwrite existing results!\n",
            UserWarning
        )

    save_dir.mkdir(parents=True, exist_ok=True)

    # Copy config file to run directory
    config_backup = save_dir / "config.ini"
    shutil.copy(config_path, config_backup)

    # Setup MLflow: experiment = problem definition, run = algorithm attempt
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_dir_name)
    mlflow.log_param("experiment_name", experiment_name)
    mlflow.log_param("algorithm", algorithm)
    mlflow.log_param("config_path", str(config_path))
    mlflow.log_param("run_directory", str(save_dir))

    print(f"Save directory: {save_dir}")
    print(f"MLflow run: {run_dir_name}")

    # ===== Dispatch to Algorithm =====

    try:
        # Dispatch to algorithm-specific training (pass save_dir, config_path, mlflow context)
        if algorithm == 'sb3_discrete':
            from coatopt.algorithms.train_sb3_discrete import train
            train(config_path=str(config_path), save_dir=str(save_dir))

        elif algorithm == 'sb3_discrete_lstm':
            from coatopt.algorithms.train_sb3_discrete_lstm import train
            train(config_path=str(config_path), save_dir=str(save_dir))

        elif algorithm == 'sb3_dqn':
            from coatopt.algorithms.train_sb3_discrete_dqn import train
            train(config_path=str(config_path), save_dir=str(save_dir))

        elif algorithm == 'sb3_simple':
            from coatopt.algorithms.train_sb3_continuous import train
            train(config_path=str(config_path), save_dir=str(save_dir))

        elif algorithm == 'morl':
            from coatopt.algorithms.train_morl_simple import train_morld as train
            train(config_path=str(config_path), save_dir=str(save_dir))

        elif algorithm == 'nsga2':
            from coatopt.algorithms.train_genetic_simple import train_genetic as train
            train(config_path=str(config_path), save_dir=str(save_dir))

        elif algorithm == 'hppo':
            from coatopt.algorithms.train_hppo_simple import train
            train(config_path=str(config_path), save_dir=str(save_dir))

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Must be one of: sb3_discrete, sb3_discrete_lstm, sb3_dqn, sb3_simple, morl, nsga2, hppo")

    finally:
        # Always end MLflow run
        mlflow.end_run()
        print(f"\n✓ MLflow run completed: {run_dir_name}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CoatOpt experiment from config file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config file",
    )

    args = parser.parse_args()
    run_experiment(args.config)
