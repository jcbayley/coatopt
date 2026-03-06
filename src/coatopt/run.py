#!/usr/bin/env python3
import argparse
import configparser
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import mlflow

from coatopt.utils.utils import save_training_results


def run_experiment(
    config_path: str,
    run_name_override: str = None,
    generate_comparison: bool = False,
    seed_override: int = None,
):
    """Run experiment based on config file.

    Handles all MLflow setup, directory creation, and dispatches to algorithm-specific training.

    Args:
        config_path: Path to INI configuration file
        run_name_override: Optional run name to override config value
        generate_comparison: Whether to run comparison after training
        seed_override: Optional seed to override config value
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Read config
    parser = configparser.ConfigParser()
    parser.read(config_path)

    # Determine algorithm from section names
    algorithm_sections = {
        "sb3_discrete",
        "sb3_discrete_lstm",
        "sb3_dqn",
        "sb3_simple",
        "morl",
        "morl_discrete",
        "nsga2",
        "sac_multiagent",
        "sac_hybrid",
        "hppo_multiagent",
        "hppo_sequential",
        "hppo_hybrid",
    }
    algorithm = None
    for section in parser.sections():
        if section.lower() in algorithm_sections:
            algorithm = section.lower()
            break

    if algorithm is None:
        raise ValueError(
            f"Config must have one of these algorithm sections: {algorithm_sections}"
        )

    # Override seed if provided
    if seed_override is not None:
        parser.set(algorithm, "seed", str(seed_override))
        print(f"Overriding seed to: {seed_override}")

    print(f"Running algorithm: {algorithm}")

    # [General] section
    base_save_dir = parser.get("general", "save_dir")
    run_name = (
        run_name_override
        if run_name_override
        else parser.get("general", "run_name", fallback="")
    )

    # Get or generate experiment name (problem definition)
    experiment_name = parser.get("general", "experiment_name", fallback=None)

    # [Data] section - read for experiment name generation
    n_layers = parser.getint("data", "n_layers")
    min_thickness = parser.getfloat("data", "min_thickness", fallback=0.1)
    max_thickness = parser.getfloat("data", "max_thickness", fallback=0.5)

    if not experiment_name:
        experiment_name = f"{n_layers}layer-{min_thickness:.2f}-{max_thickness:.2f}"

    # create run dir
    date_str = datetime.now().strftime("%Y%m%d")
    if run_name:
        run_dir_name = f"{date_str}-{algorithm}-{run_name}"
    else:
        run_dir_name = f"{date_str}-{algorithm}"

    # Directory structure mirrors MLflow: runs/experiment/run
    save_dir = Path(base_save_dir) / experiment_name / run_dir_name

    # Check if directory exists and warn
    if save_dir.exists():
        warnings.warn(
            f"\nWARNING: Run directory already exists: {save_dir}\n"
            f"    This will overwrite existing results!\n",
            UserWarning,
        )
        sys.exit()

    save_dir.mkdir(parents=True, exist_ok=True)

    # Copy config file to run directory
    config_backup = save_dir / "config.ini"
    shutil.copy(config_path, config_backup)

    # Setup MLflow
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_dir_name)
    mlflow.log_param("experiment_name", experiment_name)
    mlflow.log_param("algorithm", algorithm)
    mlflow.log_param("config_path", str(config_path))
    mlflow.log_param("run_directory", str(save_dir))

    print(f"Save directory: {save_dir}")
    print(f"MLflow run: {run_dir_name}")

    # Algorithm-specific training
    start_time = time.time()

    if algorithm == "sb3_discrete":
        from coatopt.algorithms.train_sb3_discrete import train

        results = train(config_path=str(config_path), save_dir=str(save_dir))

    elif algorithm == "nsga2":
        from coatopt.algorithms.train_genetic_simple import train_genetic as train

        results = train(config_path=str(config_path), save_dir=str(save_dir))

    elif algorithm == "sb3_dqn":
        from coatopt.algorithms.train_sb3_discrete_dqn import train

        results = train(config_path=str(config_path), save_dir=str(save_dir))

    elif algorithm == "sb3_simple":
        from coatopt.algorithms.train_sb3_continuous import train

        results = train(config_path=str(config_path), save_dir=str(save_dir))

    elif algorithm == "morl":
        from coatopt.algorithms.train_morl_simple import train

        # Read sub-algorithm from [morl] section (method = pgmorl / morld / moppo)
        morl_section = "morl" if parser.has_section("morl") else "general"
        sub_algo = parser.get(morl_section, "method", fallback="morld")
        results = train(
            config_path=str(config_path),
            algorithm=sub_algo,
            save_dir=str(save_dir),
        )

    elif algorithm == "morl_discrete":
        from coatopt.algorithms.train_morl_discrete import train

        sub_algo = parser.get("morl_discrete", "sub_algorithm", fallback="gpipd")
        results = train(
            config_path=str(config_path),
            algorithm=sub_algo,
            save_dir=str(save_dir),
        )

    elif algorithm == "sac_multiagent":
        from coatopt.algorithms.train_sac_multiagent import train

        results = train(config_path=str(config_path), save_dir=str(save_dir))

    elif algorithm == "sac_hybrid":
        from coatopt.algorithms.train_sac_hybrid import train

        results = train(config_path=str(config_path), save_dir=str(save_dir))

    elif algorithm == "hppo_multiagent":
        from coatopt.algorithms.train_hppo_multiagent import train

        results = train(config_path=str(config_path), save_dir=str(save_dir))

    elif algorithm == "hppo_sequential":
        from coatopt.algorithms.train_hppo_sequential import train

        results = train(config_path=str(config_path), save_dir=str(save_dir))

    elif algorithm == "hppo_hybrid":
        from coatopt.algorithms.train_hppo_hybrid import train

        results = train(config_path=str(config_path), save_dir=str(save_dir))

    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. Must be one of: sb3_discrete, sb3_discrete_lstm, sb3_dqn, sb3_simple, morl, morl_discrete, nsga2, hppo, sac_multiagent, sac_hybrid, ppo_multiagent, ppo_sequential, hppo_hybrid"
        )

    end_time = time.time()

    # Save all results in a standardized format
    save_training_results(
        results=results,
        save_dir=save_dir,
        algorithm_name=algorithm,
        start_time=start_time,
        end_time=end_time,
        config_path=str(config_path),
    )

    # Generate interactive Pareto front visualization (only if results are non-empty)
    try:
        from coatopt.utils.plot_interactive_pareto import (
            create_interactive_plot,
            load_materials,
        )
        from coatopt.utils.utils import load_pareto_front

        materials_path = parser.get("general", "materials_path")
        materials = load_materials(materials_path)
        designs_df, values_df, _ = load_pareto_front(save_dir)

        if not values_df.empty:
            print("\nGenerating interactive Pareto front visualization...")
            fig = create_interactive_plot(
                designs_df, values_df, materials, max_designs=10
            )
            html_path = save_dir / "pareto_interactive.html"
            fig.write_html(str(html_path))
            print(f"Saved interactive visualization to {html_path}")
        else:
            print("\nSkipping interactive visualization (empty Pareto front).")
    except Exception as e:
        print(f"\nWarning: could not generate interactive visualization: {e}")

    # Run comparison across all runs in this experiment if requested
    if generate_comparison:
        import subprocess

        print("\nRunning comparison across all runs in experiment...")
        alldirs = Path(base_save_dir) / experiment_name
        compare_cmd = [
            "python",
            "-m",
            "coatopt.compare_outputs",
            "--alldirs",
            str(alldirs),
            "--add-reference",
            "--config",
            str(config_path),
            "--top-n",
            "5",
            "--reference-layers",
            str(n_layers),
        ]
        subprocess.run(compare_cmd, check=True)
        print("Comparison complete.")

    mlflow.end_run()


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
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override run name from config file (useful for parallel runs)",
    )
    parser.add_argument(
        "--generate-comparison",
        action="store_true",
        help="Run comparison across all runs in experiment after training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config file (useful for parallel runs)",
    )

    args = parser.parse_args()
    run_experiment(
        args.config,
        run_name_override=args.run_name,
        generate_comparison=args.generate_comparison,
        seed_override=args.seed,
    )
