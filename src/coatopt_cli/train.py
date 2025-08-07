#!/usr/bin/env python3
"""
Command-line training interface for coatopt

This script allows running HPPO coating optimization training without the GUI,
using the unified HPPOTrainer for simplified and consistent behavior.

Usage:
    coatopt-train -c config.ini [options]
    python -m coatopt_cli.train -c config.ini [options]
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
from coatopt.config.structured_config import CoatingoptimisationConfig
from coatopt.factories import setup_optimisation_pipeline
from coatopt.algorithms.hppo_trainer import HPPOTrainer, create_cli_callbacks


class CommandLineTrainer:
    """Command-line interface using the unified HPPOTrainer."""
    
    def __init__(self, config_path: str, continue_training: bool = True, 
                 verbose: bool = True, save_plots: bool = False):
        """
        Initialize command-line trainer.
        
        Args:
            config_path: Path to configuration file
            continue_training: Whether to continue from existing checkpoint
            verbose: Whether to show detailed progress
            save_plots: Whether to save training plots
        """
        self.config_path = os.path.abspath(config_path)
        self.verbose = verbose
        self.save_plots = save_plots
        self.save_visualizations = save_plots  # Save visualizations if plots enabled
        self.continue_training = continue_training
        self.should_stop = False
        
        # Components will be loaded in load_configuration
        self.config = None
        self.materials = None
        self.env = None
        self.agent = None
        self.trainer = None
        
        # Setup signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print(f"\nReceived signal {signum}. Stopping training gracefully...")
        self.should_stop = True
    
    def load_configuration(self):
        """Load configuration file and setup training components."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        print(f"Loading configuration from: {self.config_path}")
        
        # Load configuration and materials
        raw_config = read_config(self.config_path)
        self.config = CoatingoptimisationConfig.from_config_parser(raw_config)
        self.materials = read_materials(self.config.general.materials_file)
        
        print(f"Loaded {len(self.materials)} materials")
        print(f"Optimization parameters: {self.config.data.optimise_parameters}")
        print(f"Number of objectives: {len(self.config.data.optimise_parameters)}")
        
        # Setup optimization components
        self.env, self.agent, temp_trainer = setup_optimisation_pipeline(
            self.config, 
            self.materials, 
            continue_training=self.continue_training,
            init_pareto_front=False
        )
        
        # Create callbacks for progress reporting
        callbacks = create_cli_callbacks(verbose=self.verbose)
        callbacks.should_stop = lambda: self.should_stop
        callbacks.save_plots = self.save_plots
        callbacks.save_visualizations = self.save_visualizations
        
        # Create unified trainer with callbacks
        self.trainer = HPPOTrainer(
            self.agent, self.env,
            n_iterations=self.config.training.n_iterations,
            n_layers=self.config.data.n_layers,
            root_dir=self.config.general.root_dir,
            entropy_beta_start=self.config.training.entropy_beta_start,
            entropy_beta_end=self.config.training.entropy_beta_end,
            entropy_beta_decay_length=self.config.training.entropy_beta_decay_length,
            entropy_beta_decay_start=self.config.training.entropy_beta_decay_start,
            n_epochs_per_update=self.config.training.n_epochs_per_update,
            use_obs=self.config.data.use_observation,
            scheduler_start=self.config.training.scheduler_start,
            scheduler_end=self.config.training.scheduler_end,
            weight_network_save=self.config.training.weight_network_save,
            use_unified_checkpoints=True,
            save_plots=self.save_plots,
            save_episode_visualizations=self.save_visualizations,
            continue_training=self.continue_training,
            callbacks=callbacks
        )
        
        # Check if we have existing data
        if self.continue_training:
            if hasattr(self.env, 'pareto_front') and len(self.env.pareto_front) > 0:
                print(f"Continuing training with existing Pareto front: {len(self.env.pareto_front)} points")
            
            if hasattr(self.trainer, 'start_episode') and self.trainer.start_episode > 0:
                print(f"Resuming from episode {self.trainer.start_episode}")
        else:
            print("Starting fresh training (retraining mode)")
    
    def run_training(self):
        """Run the training process using unified HPPOTrainer."""
        if self.trainer is None:
            raise RuntimeError("Configuration must be loaded before training")
        
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        print(f"Configuration: {self.config_path}")
        print(f"Continue training: {self.continue_training}")
        print(f"Output directory: {self.config.general.root_dir}")
        print("="*60)
        print("\nPress Ctrl+C to stop training gracefully\n")
        
        # Add training start/completion callbacks to existing callbacks
        def on_training_start(info):
            print(f"Training for {info['total_episodes']} episodes...")
            print(f"Starting from episode {info['start_episode']}")
        
        def on_training_complete(info):
            if info['success']:
                print(f"\nTraining complete! Final episode: {info['final_episode']}")
                print(f"Total training time: {info['total_time']/3600:.2f} hours")
                print("\n" + "="*60)
                print("TRAINING COMPLETED SUCCESSFULLY")
                print("="*60)
                print(f"Output directory: {self.config.general.root_dir}")
                print(f"Final metrics saved to training checkpoint")
                if self.save_plots:
                    print(f"Training plots saved to output directory")
                print("="*60)
            else:
                print(f"\nTraining stopped at episode {info['final_episode']}")
                print(f"Total training time: {info['total_time']/3600:.2f} hours")
                if 'error' in info:
                    print(f"Error: {info['error']}")
        
        # Update callbacks
        self.trainer.callbacks.on_training_start = on_training_start
        self.trainer.callbacks.on_training_complete = on_training_complete
        
        try:
            # Initialize Pareto front if not continuing
            if not self.continue_training:
                print("Initializing Pareto front...")
                self.trainer.init_pareto_front(n_solutions=1000)
            
            # Run training using unified trainer
            final_metrics, best_state = self.trainer.train()
            return True
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            return False
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            if self.verbose:
                traceback.print_exc()
            return False


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Command-line training interface for coatopt HPPO optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with config file, continuing from checkpoint
  coatopt-train -c config.ini
  
  # Train from scratch (ignore existing data)  
  coatopt-train -c config.ini --retrain
  
  # Train with plot saving enabled
  coatopt-train -c config.ini --save-plots
  
  # Run with minimal output
  coatopt-train -c config.ini --quiet
        """)
    
    parser.add_argument('-c', '--config', required=True,
                        help='Path to configuration file (required)')
    parser.add_argument('--retrain', action='store_true',
                        help='Start training from scratch (ignore existing checkpoints)')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save training plots and visualizations')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Minimal output (less verbose)')
    parser.add_argument('-v', '--version', action='version', version='coatopt-train 1.0.0')
    
    args = parser.parse_args()
    
    # Validate config file
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found")
        sys.exit(1)
    
    try:
        # Create and run trainer
        trainer = CommandLineTrainer(
            config_path=args.config,
            continue_training=not args.retrain,
            verbose=not args.quiet,
            save_plots=args.save_plots
        )
        
        # Load configuration
        trainer.load_configuration()
        
        # Run training
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
