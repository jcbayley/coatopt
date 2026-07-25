#!/usr/bin/env python3
import argparse
import ast
import configparser
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
    continue_run: bool = False,
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
        "hppo_preference",
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
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    console = Console()

    if seed_override is not None:
        parser.set(algorithm, "seed", str(seed_override))
        console.print(f"[bold yellow]⚠ Overriding seed to: {seed_override}[/bold yellow]")

    console.print(Panel.fit(
        f"[bold cyan]CoatOpt Reinforcement Learning & Optimization[/bold cyan]\n"
        f"Algorithm:   [bold yellow]{algorithm}[/bold yellow]\n"
        f"Config Path: [bold green]{config_path.resolve()}[/bold green]",
        title="[bold white]Startup[/bold white]",
        border_style="cyan"
    ))

    # [General] section
    base_save_dir = parser.get("general", "save_dir")
    run_name = (
        run_name_override
        if run_name_override
        else parser.get("general", "run_name", fallback="")
    )

    # Validate materials before starting the run
    materials_path = parser.get("general", "materials_path", fallback=None)
    if materials_path:
        from coatopt.utils.utils import validate_materials
        validate_materials(materials_path)

    # Get or generate experiment name (problem definition)
    experiment_name = parser.get("general", "experiment_name", fallback=None)

    # [Data] section - read for experiment name generation
    n_layers = parser.getint("data", "n_layers")
    min_thickness = parser.getfloat("data", "min_thickness", fallback=0.1)
    max_thickness = parser.getfloat("data", "max_thickness", fallback=0.5)

    if not experiment_name:
        optimise_parameters = ast.literal_eval(
            parser.get("data", "optimise_parameters", fallback="[]")
        )
        n_objectives = len(optimise_parameters)
        experiment_name = (
            f"{n_objectives}obj_{n_layers}layer-{min_thickness:.2f}-{max_thickness:.2f}"
        )

    # create run dir
    date_str = datetime.now().strftime("%Y%m%d")
    if run_name:
        run_dir_name = f"{date_str}-{algorithm}-{run_name}"
    else:
        run_dir_name = f"{date_str}-{algorithm}"

    # Directory structure mirrors MLflow: runs/experiment/run
    save_dir = Path(base_save_dir) / experiment_name / run_dir_name

    # Check if directory exists
    if save_dir.exists():
        if not continue_run:
            console.print(Panel.fit(
                f"[bold red]❌ WARNING: Save directory already exists:[/bold red]\n"
                f"[yellow]{save_dir}[/yellow]\n\n"
                f"[bold white]Use --continue to resume training from a checkpoint.[/bold white]",
                border_style="red",
                title="[bold red]Error[/bold red]"
            ))
            sys.exit()
        checkpoint_path = save_dir / "checkpoint_latest.pt"
        if checkpoint_path.exists():
            console.print(f"[bold green]✓ Continuing training from checkpoint:[/bold green] [cyan]{checkpoint_path.name}[/cyan]")
        else:
            console.print(f"[bold yellow]⚠ No checkpoint found in {save_dir.name}, starting from scratch.[/bold yellow]")

    save_dir.mkdir(parents=True, exist_ok=True)

    # Write modified config to run directory (includes any overrides)
    config_backup = save_dir / "config.ini"
    with open(config_backup, "w") as f:
        parser.write(f)

    # Setup MLflow
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_dir_name)
    mlflow.log_param("experiment_name", experiment_name)
    mlflow.log_param("algorithm", algorithm)
    mlflow.log_param("config_path", str(config_path))
    mlflow.log_param("run_directory", str(save_dir))

    # Print Configuration Summary in a premium Table
    table = Table(title="[bold magenta]Experiment Run Configuration Summary[/bold magenta]", show_header=True, header_style="bold cyan")
    table.add_column("Parameter", style="bold white")
    table.add_column("Value", style="yellow")
    
    from coatopt.utils.configs import load_config
    parsed_cfg = load_config(str(config_path))
    data_cfg = parsed_cfg.data

    table.add_row("Experiment Name", str(experiment_name))
    table.add_row("Algorithm", str(algorithm))
    table.add_row("Save Directory", str(save_dir))
    table.add_row("MLflow Run Name", str(run_dir_name))
    table.add_row("Wavelength", f"{data_cfg.wavelength * 1e9:.1f} nm ({data_cfg.wavelength:.4e} m)")
    table.add_row("Beam Radius (wBeam)", f"{data_cfg.wBeam * 1e3:.1f} mm ({data_cfg.wBeam:.4f} m)")
    table.add_row("Frequency", f"{data_cfg.frequency:.1f} Hz")
    table.add_row("Temperature", f"{data_cfg.temperature:.1f} K")
    table.add_row("Template Layers", str(n_layers))
    table.add_row("Thickness Bounds", f"{min_thickness:.2f} to {max_thickness:.2f}")
    if seed_override is not None:
        table.add_row("Seed Override", str(seed_override))
    
    console.print(table)
    console.print(f"\n[bold green]🚀 Commencing training via {algorithm}...[/bold green]\n")

    # Algorithm-specific training
    start_time = time.time()

    if algorithm == "sb3_discrete":
        from coatopt.algorithms.train_sb3_discrete import train

        results = train(config_path=str(config_backup), save_dir=str(save_dir))

    elif algorithm == "nsga2":
        from coatopt.algorithms.train_genetic_simple import train_genetic as train

        results = train(config_path=str(config_backup), save_dir=str(save_dir))

    elif algorithm == "sb3_dqn":
        from coatopt.algorithms.train_sb3_discrete_dqn import train

        results = train(config_path=str(config_backup), save_dir=str(save_dir))

    elif algorithm == "sb3_simple":
        from coatopt.algorithms.train_sb3_continuous import train

        results = train(config_path=str(config_backup), save_dir=str(save_dir))

    elif algorithm == "morl":
        from coatopt.algorithms.train_morl_simple import train

        # Read sub-algorithm from [morl] section (method = pgmorl / morld / moppo)
        morl_section = "morl" if parser.has_section("morl") else "general"
        sub_algo = parser.get(morl_section, "method", fallback="morld")
        results = train(
            config_path=str(config_backup),
            algorithm=sub_algo,
            save_dir=str(save_dir),
        )

    elif algorithm == "morl_discrete":
        from coatopt.algorithms.train_morl_discrete import train

        sub_algo = parser.get("morl_discrete", "sub_algorithm", fallback="gpipd")
        results = train(
            config_path=str(config_backup),
            algorithm=sub_algo,
            save_dir=str(save_dir),
        )

    elif algorithm == "sac_multiagent":
        from coatopt.algorithms.train_sac_multiagent import train

        results = train(config_path=str(config_backup), save_dir=str(save_dir))

    elif algorithm == "sac_hybrid":
        from coatopt.algorithms.train_sac_hybrid import train

        results = train(config_path=str(config_backup), save_dir=str(save_dir))

    elif algorithm == "hppo_multiagent":
        from coatopt.algorithms.train_hppo_multiagent import train

        results = train(config_path=str(config_backup), save_dir=str(save_dir))

    elif algorithm == "hppo_sequential":
        from coatopt.algorithms.train_hppo_sequential import train

        results = train(config_path=str(config_backup), save_dir=str(save_dir))

    elif algorithm == "hppo_hybrid":
        from coatopt.algorithms.train_hppo_hybrid import train

        results = train(config_path=str(config_backup), save_dir=str(save_dir))

    elif algorithm == "hppo_preference":
        from coatopt.algorithms.train_hppo_preference import train

        results = train(config_path=str(config_backup), save_dir=str(save_dir))

    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. Must be one of: sb3_discrete, sb3_discrete_lstm, sb3_dqn, sb3_simple, morl, morl_discrete, nsga2, hppo, sac_multiagent, sac_hybrid, ppo_multiagent, ppo_sequential, hppo_hybrid, hppo_preference"
        )

    end_time = time.time()

    # Save all results in a standardized format
    save_training_results(
        results=results,
        save_dir=save_dir,
        algorithm_name=algorithm,
        start_time=start_time,
        end_time=end_time,
        config_path=str(config_backup),
    )

    # Generate interactive Pareto front visualization (only if results are non-empty)
    try:
        from coatopt.utils.utils import load_pareto_front
        designs_df, values_df, _ = load_pareto_front(save_dir)

        if not values_df.empty:
            from coatopt.utils.plot_interactive_3d_rank import generate_3d_rank_dashboard

            # Read comparison targets from [comparison] section if available
            compare_refl = None
            compare_abs = None
            compare_tn = None
            compare_thick = None
            compare_label = "Reference Design"
            color_mode = "reflectivity_log"
            rank_by_utility = True
            precompute_tmm_count = -1
            target_refl = None
            target_abs = None
            target_tn = None
            target_thick = None

            if parser.has_section("comparison"):
                compare_refl = parser.getfloat("comparison", "compare_refl", fallback=None)
                compare_abs = parser.getfloat("comparison", "compare_abs", fallback=None)
                compare_tn = parser.getfloat("comparison", "compare_tn", fallback=None)
                compare_thick = parser.getfloat("comparison", "compare_thick", fallback=None)
                compare_label = parser.get("comparison", "compare_label", fallback="Reference Design")
                color_mode = parser.get("comparison", "color_mode", fallback="reflectivity_log")
                rank_by_utility = parser.getboolean("comparison", "rank_by_utility", fallback=True)
                precompute_tmm_count = parser.getint("comparison", "precompute_tmm_count", fallback=-1)
                target_refl = parser.getfloat("comparison", "target_refl", fallback=None)
                target_abs = parser.getfloat("comparison", "target_abs", fallback=None)
                target_tn = parser.getfloat("comparison", "target_tn", fallback=None)
                target_thick = parser.getfloat("comparison", "target_thick", fallback=None)

            print("\nGenerating interactive Pareto front visualization (3D Rank Dashboard)...")
            html_path = generate_3d_rank_dashboard(
                directory=save_dir,
                output=save_dir / "pareto_3d_rank.html",
                light=False,
                color_by_loss=False,
                no_open=True,
                compare_refl=compare_refl,
                compare_abs=compare_abs,
                compare_tn=compare_tn,
                compare_label=compare_label,
                compare_thick=compare_thick,
                min_refl=None,
                max_abs=None,
                max_tn=None,
                rank_by_utility=rank_by_utility,
                top=None,
                target_refl=target_refl,
                target_abs=target_abs,
                target_tn=target_tn,
                target_thick=target_thick,
                precompute_tmm_count=precompute_tmm_count,
                color_mode=color_mode,
            )
            print(f"Saved interactive visualization to {html_path}")

            if not designs_df.empty:
                from coatopt.utils.plot_design_diversity import plot_design_diversity

                print("\nGenerating design diversity plot...")
                plot_design_diversity(designs_df, values_df, save_dir)
        else:
            print("\nSkipping interactive visualization (empty Pareto front).")
    except Exception as e:
        print(f"\nWarning: could not generate interactive visualization: {e}")

    # Run comparison across all runs in this experiment if requested
    if generate_comparison:
        from coatopt.compare_outputs import main as compare_main

        print("\nRunning comparison across all runs in experiment...")
        alldirs = Path(base_save_dir) / experiment_name

        # Save original sys.argv and set new args for compare_outputs
        original_argv = sys.argv
        sys.argv = [
            "compare_outputs",
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

        try:
            compare_main()
            print("Comparison complete.")
        finally:
            # Restore original sys.argv
            sys.argv = original_argv

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
    parser.add_argument(
        "--continue",
        action="store_true",
        dest="continue_run",
        help="Continue training from an existing checkpoint in the run directory",
    )

    args = parser.parse_args()
    run_experiment(
        args.config,
        run_name_override=args.run_name,
        generate_comparison=args.generate_comparison,
        seed_override=args.seed,
        continue_run=args.continue_run,
    )
