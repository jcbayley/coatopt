"""
Multi-objective MCTS training script for coating design.
Uses Monte Carlo Tree Search for systematic exploration of material sequences
combined with continuous policy networks for thickness optimization.
"""
import os
import argparse
from coatopt.config import read_config, read_materials
from coatopt.config.structured_config import CoatingOptimisationConfig
from coatopt.factories import setup_mcts_optimisation_pipeline
from coatopt.utils.evaluation import run_evaluation_pipeline, create_enhanced_pareto_plots


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MCTS for multi-objective coating optimisation")
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
    
    # Validate MCTS configuration
    if structured_config.mcts is None:
        print("Warning: No [MCTS] configuration section found. Using default MCTS parameters.")
    
    materials = read_materials(structured_config.general.materials_file)
    print(f"Loaded {len(materials)} materials")
    
    return structured_config, materials


def run_training(trainer, env, output_dir):
    """Execute MCTS training process."""
    print("Starting MCTS training...")
    trainer.train()
    print("MCTS training completed successfully.")
    print(f"Results have been saved to: {trainer.output_dir}")
    
    # Create final visualizations
    print("Creating final Pareto front visualization...")
    try:
        create_pareto_visualizations(trainer, env, output_dir)
        print("Pareto front plots created successfully.")
    except Exception as e:
        print(f"Warning: Could not create Pareto front plots: {e}")


def create_pareto_visualizations(trainer, env, output_dir):
    """Create comprehensive Pareto front visualizations."""
    if not trainer.pareto_solutions:
        print("No Pareto solutions available for visualization.")
        return
    
    # Extract data from Pareto solutions
    pareto_states = [sol['coating_state'] for sol in trainer.pareto_solutions]
    pareto_results = [[sol['metrics']['reflectivity'], 
                      sol['metrics']['thermal_noise'], 
                      sol['metrics']['absorption']] for sol in trainer.pareto_solutions]
    
    # Get the objective weights used (may not be directly available)
    sampled_weights = [sol.get('objective_weights', [0.33, 0.33, 0.34]) for sol in trainer.pareto_solutions]
    
    # Create enhanced Pareto plots
    try:
        create_enhanced_pareto_plots(trainer, env, pareto_results, pareto_states, sampled_weights, output_dir)
    except Exception as e:
        print(f"Enhanced plots failed: {e}")
        # Fallback to trainer's built-in plots
        trainer._create_pareto_plots(suffix="_final")


def run_evaluation(trainer, env, n_samples: int, output_dir: str):
    """Execute MCTS evaluation and visualization."""
    print("Starting MCTS evaluation...")
    
    # Use trainer's evaluate method
    sampled_states, results, sampled_weights = trainer.evaluate(n_samples)
    
    if not results:
        print("No valid evaluation results generated.")
        return sampled_states, results, sampled_weights
    
    print(f"Evaluation completed with {len(results)} valid samples.")
    
    # Create evaluation visualizations
    try:
        create_enhanced_pareto_plots(trainer, env, results, sampled_states, sampled_weights, output_dir)
        print("Evaluation plots created successfully.")
    except Exception as e:
        print(f"Warning: Could not create evaluation plots: {e}")
    
    return sampled_states, results, sampled_weights


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Load configuration
        config, materials = load_configuration(args.config)
        
        # Setup MCTS optimisation pipeline
        print("Setting up MCTS optimisation pipeline...")
        env, trainer = setup_mcts_optimisation_pipeline(config, materials)
        
        output_dir = config.general.root_dir
        
        # Execute requested operations
        if args.train:
            run_training(trainer, env, output_dir)
        
        if args.test:
            run_evaluation(trainer, env, args.n_samples, output_dir)
        
        if not args.train and not args.test:
            print("No operation specified. Use --train and/or --test flags.")
            print("Example: python train_mcts_pareto.py -c config.ini --train --test")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
