#!/usr/bin/env python3
"""
Multi-objective genetic algorithm training script for coating design.
Refactored to match the HPPO training structure and use structured configuration.

Usage:
    python -m coatopt_cli.train_genetic_multiopt -c config.ini [options]
    python train_genetic_multiopt.py -c config.ini [options]
"""
import argparse
import os
import sys
from pathlib import Path

# Add coatopt to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from coatopt.config import read_config, read_materials
from coatopt.config.structured_config import CoatingOptimisationConfig
from coatopt.factories import setup_genetic_optimisation_pipeline
from coatopt.utils.evaluation import (
    create_enhanced_pareto_plots,
    run_evaluation_pipeline,
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train genetic algorithm for multi-objective coating optimisation"
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run training",
    )
    parser.add_argument(
        "--test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run testing/evaluation",
    )
    parser.add_argument(
        "-n",
        "--n-samples",
        type=int,
        required=False,
        default=1000,
        help="Number of samples for evaluation",
    )
    return parser.parse_args()


def load_configuration(config_path: str):
    """Load and structure configuration."""
    print(f"Loading configuration from: {config_path}")
    raw_config = read_config(os.path.abspath(config_path))
    structured_config = CoatingOptimisationConfig.from_config_parser(raw_config)

    # Validate genetic configuration
    if structured_config.genetic is None:
        raise ValueError(
            "Genetic algorithm configuration section [Genetic] is required"
        )

    materials = read_materials(structured_config.general.materials_file)
    print(f"Loaded {len(materials)} materials")

    return structured_config, materials


def run_training(trainer):
    """Execute training process."""
    print("Starting genetic algorithm optimisation...")
    trainer.train()
    print("Training completed.")


def run_evaluation(trainer, env, n_samples: int, output_dir: str):
    """Execute evaluation and visualization."""
    print("Starting evaluation...")

    # Use trainer's evaluate method which handles genetic algorithm specifics
    sampled_states, results, sampled_weights = trainer.evaluate(n_samples)

    # The genetic trainer already saves results and creates basic plots
    # We can add enhanced plots if needed
    try:
        create_enhanced_pareto_plots(
            trainer, env, results, sampled_states, sampled_weights, output_dir
        )
    except Exception as e:
        print(f"Warning: Could not create enhanced plots: {e}")

    return sampled_states, results, sampled_weights


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Load configuration
    config, materials = load_configuration(args.config)

    # Setup genetic optimisation pipeline
    env, trainer = setup_genetic_optimisation_pipeline(config, materials)

    # Execute requested operations
    if args.train:
        run_training(trainer)

    if args.test:
        run_evaluation(trainer, env, args.n_samples, config.general.root_dir)

    if not args.train and not args.test:
        print("No operation specified. Use --train and/or --test flags.")
        print("Run with --help for usage information.")


if __name__ == "__main__":
    main()
