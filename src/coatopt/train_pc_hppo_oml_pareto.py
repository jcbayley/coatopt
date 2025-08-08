"""
Multi-objective Pareto optimization training script for coating design.
Refactored for improved structure and maintainability.
"""
import os
import argparse
from coatopt.config import read_config, read_materials
from coatopt.config.structured_config import CoatingOptimizationConfig
from coatopt.factories import setup_optimization_pipeline
from coatopt.utils.evaluation import run_evaluation_pipeline, create_enhanced_pareto_plots


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PC-HPPO-OML for multi-objective coating optimization")
    parser.add_argument("-c", "--config", type=str, required=False, default="none",
                        help="Path to configuration file")
    parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=False,
                        help="Run training")
    parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=False,
                        help="Run testing/evaluation")
    parser.add_argument('--continue-training', action=argparse.BooleanOptionalAction, default=False,
                        help="Continue training from checkpoint")
    parser.add_argument('-n', "--n-samples", type=int, required=False, default=1000,
                        help="Number of samples for evaluation")
    return parser.parse_args()


def load_configuration(config_path: str):
    """Load and structure configuration."""
    print(f"Loading configuration from: {config_path}")
    raw_config = read_config(os.path.abspath(config_path))
    structured_config = CoatingOptimizationConfig.from_config_parser(raw_config)
    
    materials = read_materials(structured_config.general.materials_file)
    print(f"Loaded {len(materials)} materials")
    
    return structured_config, materials


def run_training(trainer):
    """Execute training process."""
    print("Starting training...")
    trainer.train()
    print("Training completed.")


def run_evaluation(trainer, env, n_samples: int, output_dir: str):
    """Execute evaluation and visualization."""
    print("Starting evaluation...")
    sampled_states, results, sampled_weights = run_evaluation_pipeline(trainer, env, n_samples, output_dir)
    
    # Create enhanced pareto plots with training data if available
    create_enhanced_pareto_plots(trainer, env, results, sampled_states, sampled_weights, output_dir)
    
    return sampled_states, results, sampled_weights


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config, materials = load_configuration(args.config)
    
    # Determine if continuing training
    continue_training = args.continue_training or config.general.continue_training
    
    # Setup optimization pipeline
    env, agent, trainer = setup_optimization_pipeline(config, materials, continue_training)
    
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
