#!/usr/bin/env python3
"""
Command-line training interface for layer editing coating optimization

This script provides a specialized interface for training layer editing agents,
with sensible defaults and validation for the layer editing environment.

Usage:
    coatopt-train-layer-edit -c config.ini [options]
    python -m coatopt_cli.train_layer_edit -c config.ini [options]
"""

import argparse
import os
import sys
import signal
import time
from pathlib import Path
import traceback

# Add coatopt to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from coatopt.config import read_config, read_materials
from coatopt.config.structured_config import CoatingOptimisationConfig
from coatopt.factories import setup_optimisation_pipeline
from coatopt.algorithms.hppo.training.trainer import HPPOTrainer, create_cli_callbacks, HPPOConstants
from coatopt.utils.evaluation import run_evaluation_pipeline, create_enhanced_pareto_plots
from coatopt.utils.plotting import TrainingPlotManager


class LayerEditingTrainer:
    """Specialized command-line interface for layer editing training."""
    
    def __init__(self, config_path: str, continue_training: bool = True, 
                 verbose: bool = True, save_plots: bool = False, 
                 evaluate_only: bool = False, n_eval_samples: int = 1000):
        """
        Initialize layer editing trainer.
        
        Args:
            config_path: Path to configuration file
            continue_training: Whether to continue from existing checkpoint
            verbose: Whether to show detailed progress
            save_plots: Whether to save training plots
            evaluate_only: Whether to only run evaluation (no training)
            n_eval_samples: Number of samples for evaluation
        """
        self.config_path = os.path.abspath(config_path)
        self.verbose = verbose
        self.save_plots = save_plots
        self.evaluate_only = evaluate_only
        self.n_eval_samples = n_eval_samples
        self.continue_training = continue_training
        
        self.config = None
        self.materials = None
        self.trainer = None
        self.training_interrupted = False
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print("\nReceived interrupt signal. Stopping training gracefully...")
        self.training_interrupted = True
        if self.trainer:
            self.trainer.stop_training()

    def load_configuration(self):
        """Load and validate configuration for layer editing."""
        if self.verbose:
            print(f"Loading configuration from: {self.config_path}")
        
        try:
            # Load configuration and materials (following train.py pattern)
            raw_config = read_config(self.config_path)
            self.config = CoatingOptimisationConfig.from_config_parser(raw_config)
            self.materials = read_materials(self.config.general.materials_file)
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")
        
        # Validate layer editing configuration
        self._validate_layer_editing_config()
        
        if self.verbose:
            print(f"✓ Configuration loaded successfully")
            print(f"  - Model type: {self.config.network.model_type}")
            print(f"  - Max layers: {self.config.data.n_layers}")
            print(f"  - Materials: {len(self.materials)}")
            print(f"  - Max edits per episode: {getattr(self.config.data, 'max_edits_per_episode', 'not set')}")

    def _validate_layer_editing_config(self):
        """Validate that the configuration is suitable for layer editing."""
        # Check model type
        if self.config.network.model_type != "layer_editing":
            raise ValueError(f"Config model_type must be 'layer_editing', got '{self.config.network.model_type}'")
        
        # Check required parameters
        required_params = ['n_layers', 'min_thickness', 'max_thickness']
        for param in required_params:
            if not hasattr(self.config.data, param):
                raise ValueError(f"Missing required parameter: data.{param}")
        
        # Check layer editing specific parameters
        if not hasattr(self.config.data, 'initial_stack_size_range'):
            print("Warning: initial_stack_size_range not set, using default (2, 5)")
        
        if not hasattr(self.config.data, 'max_edits_per_episode'):
            print("Warning: max_edits_per_episode not set, using default (10)")
        
        # Validate materials
        if len(self.materials) < 3:
            raise ValueError("Layer editing requires at least 3 materials (air, substrate, +1 coating material)")
        
        # Check for air and substrate materials (informational only)
        air_found = any(mat.get('name', '').lower() == 'air' for mat in self.materials.values())
        substrate_found = any(mat.get('name', '').lower() in ['substrate', 'sub', 'sio2'] for mat in self.materials.values())
        
        if not air_found:
            print("Info: No material explicitly named 'air' found")
        if not substrate_found:
            print("Info: No material explicitly named 'substrate', 'sub', or 'sio2' found")
            print("      (substrate is typically defined by material index in environment setup)")

    def setup_training(self):
        """Set up the training pipeline for layer editing."""
        if self.verbose:
            print("Setting up layer editing training pipeline...")
        
        try:
            # Setup optimization components first (needed to get clean parameter names)
            from coatopt.factories import setup_optimisation_pipeline
            self.env, self.agent, temp_trainer = setup_optimisation_pipeline(
                self.config, 
                self.materials, 
                continue_training=self.continue_training,
                init_pareto_front=False
            )
            
            # Create plot manager if saving plots
            plot_manager = None
            if self.save_plots:
                from coatopt.utils.plotting import TrainingPlotManager
                plot_manager = TrainingPlotManager(
                    save_plots=True,
                    output_dir=self.config.general.root_dir,
                    ui_mode=False
                )
                # Extract clean parameter names for lookups in targets/criteria dicts
                if hasattr(self.env, 'get_parameter_names'):
                    param_names = self.env.get_parameter_names()
                else:
                    # Fallback: assume no direction suffixes in legacy configs
                    param_names = self.config.data.optimise_parameters
                    
                target_list = [self.config.data.optimise_targets[param] for param in param_names]
                design_list = [self.config.data.design_criteria[param] for param in param_names]
                plot_manager.set_objective_info(param_names, target_list, design_list)
            
            # Create callbacks for progress reporting
            callbacks = create_cli_callbacks(verbose=self.verbose, plot_manager=plot_manager)
            callbacks.save_plots = self.save_plots
            
            # Create trainer using the already created env and agent
            self.trainer = HPPOTrainer(
                self.agent, self.env,
                n_iterations=self.config.training.n_iterations,
                n_layers=self.config.data.n_layers,
                root_dir=self.config.general.root_dir,
                entropy_beta_start=getattr(self.config.training, 'entropy_beta_start', 1.0),
                entropy_beta_end=getattr(self.config.training, 'entropy_beta_end', 0.001),
                entropy_beta_decay_length=getattr(self.config.training, 'entropy_beta_decay_length', None),
                entropy_beta_decay_start=getattr(self.config.training, 'entropy_beta_decay_start', 0),
                n_epochs_per_update=self.config.training.n_episodes_per_update,
                use_obs=self.config.data.use_observation,
                scheduler_start=getattr(self.config.training, 'scheduler_start', 0),
                scheduler_end=getattr(self.config.training, 'scheduler_end', float('inf')),
                continue_training=self.continue_training,
                callbacks=callbacks
            )
            
            if self.verbose:
                print(f"✓ Training pipeline setup complete")
                print(f"  - Environment: {type(self.trainer.env).__name__}")
                print(f"  - Agent: {type(self.trainer.agent).__name__}")
                print(f"  - Continue training: {self.continue_training}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to setup training pipeline: {e}")

    def run_training(self):
        """Run the layer editing training process."""
        try:
            if self.verbose:
                print("Starting layer editing training...")
                print(f"  - Model type: {self.config.network.model_type}")
                print(f"  - Training episodes: {self.config.training.n_episodes_per_update}")
                print(f"  - Updates per episode: {self.config.training.n_episodes_per_update}")
                print("-" * 40)
            
            # Run training
            success = self.trainer.train()
            
            if success and not self.training_interrupted:
                if self.verbose:
                    print("\n" + "=" * 40)
                    print("✓ Layer editing training completed successfully!")
                return True
            else:
                print("Training stopped or failed")
                return False
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            return False
        except Exception as e:
            print(f"Training failed: {e}")
            if self.verbose:
                traceback.print_exc()
            return False

    def run_evaluation(self):
        """Run evaluation on trained layer editing model."""
        try:
            if self.verbose:
                print("Running layer editing model evaluation...")
            
            # Run evaluation using the trainer's evaluation pipeline
            results = self.trainer.evaluate(n_samples=self.n_eval_samples)
            
            if self.verbose and results:
                print("\n" + "=" * 40)
                print("✓ Evaluation completed successfully!")
                if hasattr(results, 'pareto_front'):
                    print(f"  - Pareto front solutions: {len(results.pareto_front)}")
                
            return True
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            if self.verbose:
                traceback.print_exc()
            return False


def create_sample_layer_editing_config(output_path: str):
    """Create a sample configuration file for layer editing."""
    config_content = """
# Layer Editing Configuration
# This config sets up iterative layer editing for coating optimization

[data]
n_layers = 10
min_thickness = 1e-9
max_thickness = 1e-6
use_optical_thickness = false
optimise_parameters = reflectivity,thermal_noise,absorption
use_observation = false
use_intermediate_reward = false
ignore_air_option = false
ignore_substrate_option = false
reward_function = default
combine = product

# Layer editing specific parameters
initial_stack_size_range = 2,6
max_edits_per_episode = 15

[data.optimise_targets]
reflectivity = 0.99999
thermal_noise = 5.394480540642821e-21
absorption = 0.01

[data.design_criteria]  
reflectivity = 0.99999
thermal_noise = 5.394480540642821e-21
absorption = 0.01

[network]
model_type = layer_editing
hidden_size = 128
include_layer_number = true
include_material_in_policy = false
pre_network_type = linear
n_pre_layers = 2
n_continuous_layers = 2
n_discrete_layers = 2
n_value_layers = 2
discrete_hidden_size = 64
continuous_hidden_size = 64
value_hidden_size = 64
hyper_networks = false
use_mixture_of_experts = false

[training]
n_episodes = 2000
n_episodes_per_update = 1
entropy_beta_start = 1.0
entropy_beta_end = 0.001
entropy_beta_decay_length = 500
lr_discrete_policy = 1e-4
lr_continuous_policy = 1e-4
lr_value = 2e-4
lr_step = 10
lr_min = 1e-6
clip_ratio = 0.5
gamma = 0.99
optimiser = adam

[checkpointing]
save_every = 100
max_checkpoints = 5
save_path = checkpoints/layer_editing

[plotting]
plot_every = 50
save_plots = true
plot_path = plots/layer_editing
"""
    
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    print(f"Sample layer editing config created at: {output_path}")


def main():
    """Main entry point for layer editing training CLI."""
    parser = argparse.ArgumentParser(
        description='Train layer editing coating optimization models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default config
    python -m coatopt_cli.train_layer_edit -c layer_edit_config.ini
    
    # Train from scratch (ignore checkpoints)
    python -m coatopt_cli.train_layer_edit -c config.ini --retrain
    
    # Evaluate only
    python -m coatopt_cli.train_layer_edit -c config.ini --evaluate
    
    # Generate sample config
    python -m coatopt_cli.train_layer_edit --generate-config sample_layer_edit.ini
    
Note: This CLI is specifically designed for layer editing environments.
For standard HPPO training, use coatopt_cli.train instead.
        """
    )
    
    # Configuration
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument('-c', '--config', type=str,
                        help='Path to layer editing configuration file (required)')
    config_group.add_argument('--generate-config', type=str, metavar='OUTPUT_PATH',
                        help='Generate a sample layer editing config file and exit')
    
    # Training options
    parser.add_argument('--retrain', action='store_true',
                        help='Start training from scratch (ignore existing checkpoints)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation only (no training)')
    parser.add_argument('-n', '--n-samples', type=int, default=1000,
                        help='Number of samples for evaluation (default: 1000)')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save training plots and visualizations')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Minimal output (less verbose)')
    parser.add_argument('-v', '--version', action='version', version='coatopt-train-layer-edit 1.0.0')
    
    args = parser.parse_args()
    
    # Handle config generation
    if args.generate_config:
        try:
            create_sample_layer_editing_config(args.generate_config)
            print(f"✓ Sample configuration created successfully!")
            print(f"  Edit {args.generate_config} to customize your layer editing setup")
            sys.exit(0)
        except Exception as e:
            print(f"Error creating config: {e}")
            sys.exit(1)
    
    # Validate config file
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found")
        print("Use --generate-config to create a sample configuration file")
        sys.exit(1)
    
    try:
        # Create and run trainer
        trainer = LayerEditingTrainer(
            config_path=args.config,
            continue_training=not args.retrain,
            verbose=not args.quiet,
            save_plots=args.save_plots,
            evaluate_only=args.evaluate,
            n_eval_samples=args.n_samples
        )
        
        # Load configuration
        trainer.load_configuration()
        
        # Set up training pipeline
        trainer.setup_training()
        
        # Run appropriate operation
        if args.evaluate:
            if not args.quiet:
                print("Running layer editing evaluation...")
            success = trainer.run_evaluation()
        else:
            if not args.quiet:
                print("Starting layer editing training...")
            success = trainer.run_training()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if not args.quiet:
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
