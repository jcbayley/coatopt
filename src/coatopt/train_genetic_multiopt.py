"""
Multi-objective genetic algorithm training script for coating design.
Refactored to match the HPPO training structure and use structured configuration.
"""
import os
import argparse
from coatopt.config import read_config, read_materials
from coatopt.config.structured_config import CoatingOptimisationConfig
from coatopt.factories import setup_genetic_optimisation_pipeline
from coatopt.utils.evaluation import run_evaluation_pipeline, create_enhanced_pareto_plots


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train genetic algorithm for multi-objective coating optimisation")
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to configuration file")
    parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=False,
                        help="Run training")
    parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=False,
                        help="Run testing/evaluation")
    parser.add_argument('-n', "--n-samples", type=int, required=False, default=1000,
                        help="Number of samples for evaluation")
    return parser.parse_args()


def load_configuration(config_path: str):
    """Load and structure configuration."""
    print(f"Loading configuration from: {config_path}")
    raw_config = read_config(os.path.abspath(config_path))
    structured_config = CoatingOptimisationConfig.from_config_parser(raw_config)
    
    # Validate genetic configuration
    if structured_config.genetic is None:
        raise ValueError("Genetic algorithm configuration section [Genetic] is required")
    
    materials = read_materials(structured_config.general.materials_file)
    print(f"Loaded {len(materials)} materials")
    
    return structured_config, materials


def run_training(trainer, env, output_dir):
    """Execute training process."""
    print("Starting genetic algorithm optimisation...")
    trainer.train()
    print("Training completed successfully.")
    print(f"Results have been saved to: {trainer.output_dir}")
    
    # Always create Pareto front plots after training
    print("Creating Pareto front visualization...")
    try:
        create_pareto_front_plots(trainer, env, output_dir)
        print("Pareto front plots created successfully.")
    except Exception as e:
        print(f"Warning: Could not create Pareto front plots: {e}")


def create_pareto_front_plots(trainer, env, output_dir):
    """Create Pareto front plots after training."""
    if trainer.result is None:
        print("No optimization results available for plotting.")
        return
    
    # Get the Pareto front data
    pareto_states, pareto_results = trainer._process_pareto_front()
    
    # Create enhanced Pareto plots if available
    try:
        # Try to use the enhanced plotting function
        sampled_weights = None  # Not used for genetic algorithms
        create_enhanced_pareto_plots(trainer, env, pareto_results, pareto_states, sampled_weights, output_dir)
    except Exception as e:
        print(f"Enhanced plots failed, creating basic Pareto plot: {e}")
        # Fallback to basic Pareto plot
        trainer._create_pareto_plot(pareto_results)


def run_evaluation(trainer, env, n_samples: int, output_dir: str):
    """Execute evaluation and visualization."""
    print("Starting evaluation...")
    
    # Use trainer's evaluate method which handles genetic algorithm specifics
    sampled_states, results, sampled_weights = trainer.evaluate(n_samples)
    
    # The genetic trainer already saves results and creates basic plots
    # We can add enhanced plots if needed
    try:
        create_enhanced_pareto_plots(trainer, env, results, sampled_states, sampled_weights, output_dir)
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
        run_training(trainer, env, config.general.root_dir)
    
    if args.test:
        # If test is requested, ensure we have results (either from training or previously saved)
        if trainer.result is None:
            print("No optimization results found. Loading from saved state if available...")
            result_path = os.path.join(config.general.root_dir, "optimizer_result.pkl")
            if os.path.exists(result_path):
                trainer.load_optimizer_state(result_path)
                print("Loaded previous optimization results.")
            else:
                print("No saved results found. Please run training first with --train flag.")
                return
        
        run_evaluation(trainer, env, args.n_samples, config.general.root_dir)
    
    if not args.train and not args.test:
        # Default behavior: run training if no flags specified
        print("No operation specified. Running training by default...")
        print("Use --train and/or --test flags for explicit control.")
        run_training(trainer, env, config.general.root_dir)


if __name__ == "__main__":
    main()
