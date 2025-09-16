#!/usr/bin/env python3

import argparse
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from coatopt.algorithms.hppo.training.trainer import (
    HPPOConstants,
    HPPOTrainer,
    create_cli_callbacks,
)
from coatopt.config import read_config, read_materials
from coatopt.config.structured_config import CoatingOptimisationConfig
from coatopt.factories import setup_optimisation_pipeline
from coatopt.utils.evaluation import (
    create_enhanced_pareto_plots,
    run_evaluation_pipeline,
)
from coatopt.utils.mlflow_tracking import MLflowTracker, create_mlflow_tracker
from coatopt.utils.plotting import TrainingPlotManager
from coatopt.utils.plotting.core import TrainingPlotManager as CoreTrainingPlotManager

# Add coatopt to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class CommandLineTrainer:
    """Command-line interface using the unified HPPOTrainer."""

    def __init__(
        self,
        config_path: str,
        continue_training: bool = True,
        verbose: bool = True,
        save_plots: bool = False,
        evaluate_only: bool = False,
        n_eval_samples: int = 1000,
        enable_mlflow: bool = True,
        experiment_name: str = None,
        mlflow_uri: str = None,
        mlflow_dir: str = None,
    ):
        """
        Initialize command-line trainer.

        Args:
            config_path: Path to configuration file
            continue_training: Whether to continue from existing checkpoint
            verbose: Whether to show detailed progress
            save_plots: Whether to save training plots
            evaluate_only: Whether to only run evaluation (no training)
            n_eval_samples: Number of samples for evaluation
            enable_mlflow: Whether to enable MLflow experiment tracking
            experiment_name: Optional MLflow experiment name override
            mlflow_uri: Optional MLflow tracking URI override
            mlflow_dir: Optional MLflow directory (creates file:// URI)
        """
        self.config_path = os.path.abspath(config_path)
        self.verbose = verbose
        self.save_plots = save_plots
        self.save_visualizations = save_plots  # Save visualizations if plots enabled
        self.continue_training = continue_training
        self.evaluate_only = evaluate_only
        self.n_eval_samples = n_eval_samples
        self.should_stop = False

        # MLflow configuration
        self.enable_mlflow = enable_mlflow
        self.experiment_name = experiment_name
        self.mlflow_uri = mlflow_uri
        self.mlflow_dir = mlflow_dir
        self.mlflow_tracker = None

        # Components will be loaded in load_configuration
        self.config = None
        self.materials = None
        self.env = None
        self.agent = None
        self.trainer = None
        self.plot_manager = None

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
        self.config = CoatingOptimisationConfig.from_config_parser(raw_config)
        self.materials = read_materials(self.config.general.materials_file)

        print(f"Loaded {len(self.materials)} materials")
        print(f"Optimization parameters: {self.config.data.optimise_parameters}")
        print(f"Number of objectives: {len(self.config.data.optimise_parameters)}")

        # Setup MLflow tracking
        if self.enable_mlflow:
            # Convert mlflow_dir to URI if provided
            mlflow_uri = self.mlflow_uri
            if self.mlflow_dir and not mlflow_uri:
                mlflow_dir_path = Path(self.mlflow_dir).expanduser().resolve()
                mlflow_dir_path.mkdir(parents=True, exist_ok=True)
                mlflow_uri = f"file://{mlflow_dir_path}"
                print(f"MLflow experiments directory: {mlflow_dir_path}")

            self.mlflow_tracker = create_mlflow_tracker(
                config=self.config,
                experiment_name=self.experiment_name,
                tracking_uri=mlflow_uri,
                enable_logging=True,
            )
            print(
                f"MLflow tracking enabled. Experiment: {self.mlflow_tracker.experiment_id}"
            )
        else:
            print("MLflow tracking disabled")

        # Setup optimization components
        self.env, self.agent, temp_trainer = setup_optimisation_pipeline(
            self.config,
            self.materials,
            continue_training=self.continue_training,
            init_pareto_front=False,
        )

        # Create plot manager if saving plots
        if self.save_plots:
            self.plot_manager = CoreTrainingPlotManager(
                save_plots=True, output_dir=self.config.general.root_dir, ui_mode=False
            )
            # Extract clean parameter names for lookups in targets/criteria dicts
            if hasattr(self.env, "get_parameter_names"):
                param_names = self.env.get_parameter_names()
            else:
                # Fallback: assume no direction suffixes in legacy configs
                param_names = self.config.data.optimise_parameters

            target_list = [
                self.config.data.optimise_targets[param] for param in param_names
            ]
            design_list = [
                self.config.data.design_criteria[param] for param in param_names
            ]
            self.plot_manager.set_objective_info(param_names, target_list, design_list)

        # Create callbacks for progress reporting (trainer will be set later)
        callbacks = create_cli_callbacks(
            verbose=self.verbose, plot_manager=self.plot_manager, trainer=None
        )
        callbacks.should_stop = lambda: self.should_stop
        callbacks.save_plots = self.save_plots
        callbacks.save_visualizations = self.save_visualizations

        # Create unified trainer with callbacks
        self.trainer = HPPOTrainer(
            self.agent,
            self.env,
            n_iterations=self.config.training.n_iterations,
            n_layers=self.config.data.n_layers,
            root_dir=self.config.general.root_dir,
            entropy_beta_start=self.config.training.entropy_beta_start,
            entropy_beta_end=self.config.training.entropy_beta_end,
            entropy_beta_decay_length=self.config.training.entropy_beta_decay_length,
            entropy_beta_decay_start=self.config.training.entropy_beta_decay_start,
            entropy_beta_discrete_start=getattr(
                self.config.training, "entropy_beta_discrete_start", None
            ),
            entropy_beta_discrete_end=getattr(
                self.config.training, "entropy_beta_discrete_end", None
            ),
            entropy_beta_continuous_start=getattr(
                self.config.training, "entropy_beta_continuous_start", None
            ),
            entropy_beta_continuous_end=getattr(
                self.config.training, "entropy_beta_continuous_end", None
            ),
            entropy_beta_use_restarts=getattr(
                self.config.training, "entropy_beta_use_restarts", False
            ),
            n_episodes_per_epoch=self.config.training.n_episodes_per_epoch,
            n_updates_per_epoch=self.config.training.n_updates_per_epoch,
            use_obs=self.config.data.use_observation,
            scheduler_start=self.config.training.scheduler_start,
            scheduler_end=self.config.training.scheduler_end,
            weight_network_save=self.config.training.weight_network_save,
            save_plots=self.save_plots,
            save_episode_visualizations=self.save_visualizations,
            continue_training=self.continue_training,
            callbacks=callbacks,
        )

        # Update callbacks with trainer reference for constraint tracking
        if self.plot_manager:
            callbacks = create_cli_callbacks(
                verbose=self.verbose,
                plot_manager=self.plot_manager,
                trainer=self.trainer,
            )
            callbacks.should_stop = lambda: self.should_stop
            callbacks.save_plots = self.save_plots
            callbacks.save_visualizations = self.save_visualizations
            self.trainer.callbacks = callbacks

        # Load historical data for plotting if continuing training
        if self.continue_training and self.plot_manager:
            self.plot_manager.load_context_data(self.trainer.checkpoint_manager)

            # Additional constraint history loading is now handled within
            # load_context_data

        # Check if we have existing data
        if self.continue_training:
            if hasattr(self.env, "pareto_front") and len(self.env.pareto_front) > 0:
                print(
                    f"Continuing training with existing Pareto front: {len(self.env.pareto_front)} points"
                )

            if (
                hasattr(self.trainer, "start_episode")
                and self.trainer.start_episode > 0
            ):
                print(f"Resuming from episode {self.trainer.start_episode}")
        else:
            print("Starting fresh training (retraining mode)")

    def run_training(self):
        """Run the training process using unified HPPOTrainer."""
        if self.trainer is None:
            raise RuntimeError("Configuration must be loaded before training")

        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        print(f"Configuration: {self.config_path}")
        print(f"Continue training: {self.continue_training}")
        print(f"Output directory: {self.config.general.root_dir}")
        if self.enable_mlflow:
            print(f"MLflow tracking: Enabled")
        print("=" * 60)
        print("\nPress Ctrl+C to stop training gracefully\n")

        # Start MLflow run
        if self.enable_mlflow:
            self.mlflow_tracker.start_run()
            # Log config file
            self.mlflow_tracker.log_config_file(self.config_path)
            print(f"MLflow run started: {self.mlflow_tracker.run_id}")

        # Add training start/completion callbacks to existing callbacks
        def on_training_start(info):
            print(f"Training for {info['total_episodes']} episodes...")
            print(f"Starting from episode {info['start_episode']}")

            if self.enable_mlflow:
                self.mlflow_tracker.log_training_metrics(
                    episode=info["start_episode"], metrics={"training_started": 1}
                )

        def on_training_complete(info):
            if info["success"]:
                print(f"\nTraining complete! Final episode: {info['final_episode']}")
                print(f"Total training time: {info['total_time']/3600:.2f} hours")
                print("\n" + "=" * 60)
                print("TRAINING COMPLETED SUCCESSFULLY")
                print("=" * 60)
                print(f"Output directory: {self.config.general.root_dir}")
                print(f"Final metrics saved to training checkpoint")
                if self.save_plots:
                    print(f"Training plots saved to output directory")
                print("=" * 60)

                if self.enable_mlflow:
                    # Log final metrics
                    self.mlflow_tracker.log_training_metrics(
                        episode=info["final_episode"],
                        metrics={
                            "training_completed": 1,
                            "total_training_time_hours": info["total_time"] / 3600,
                            "final_episode": info["final_episode"],
                        },
                    )

                    # Log final plots
                    self.mlflow_tracker.log_plots(self.config.general.root_dir)

                    print(
                        f"MLflow run completed: {self.mlflow_tracker.get_run_info()['tracking_uri']}"
                    )
            else:
                print(f"\nTraining stopped at episode {info['final_episode']}")
                print(f"Total training time: {info['total_time']/3600:.2f} hours")
                if "error" in info:
                    print(f"Error: {info['error']}")

                if self.enable_mlflow:
                    self.mlflow_tracker.log_training_metrics(
                        episode=info.get("final_episode", 0),
                        metrics={
                            "training_stopped": 1,
                            "total_training_time_hours": (
                                info["total_time"] / 3600 if "total_time" in info else 0
                            ),
                        },
                    )

        # Enhanced episode callback for MLflow logging
        original_episode_callback = self.trainer.callbacks.on_episode_complete

        def enhanced_episode_callback(episode: int, info: dict):
            # Call original callback
            if original_episode_callback:
                original_episode_callback(episode, info)

            # MLflow logging
            if (
                self.enable_mlflow and episode % 10 == 0
            ):  # Log every 10 episodes to avoid spam
                metrics = {}

                # Basic episode metrics
                if "reward" in info:
                    metrics["episode_reward"] = info["reward"]
                if "max_reward" in info:
                    metrics["max_reward_seen"] = info["max_reward"]
                if "training_time" in info:
                    metrics["training_time_hours"] = info["training_time"] / 3600

                # Extract detailed metrics from the metrics dict
                if "metrics" in info and isinstance(info["metrics"], dict):
                    episode_metrics = info["metrics"]

                    # Get objective names dynamically from the trainer environment
                    obj_names = self.trainer.env.get_parameter_names()
                    # Individual objective values (physical values)
                    for obj_name in obj_names:
                        if obj_name in episode_metrics:
                            metrics[f"value_{obj_name}"] = episode_metrics[obj_name]

                    # Individual objective rewards
                    for obj_name in obj_names:
                        reward_key = f"{obj_name}_reward"
                        if reward_key in episode_metrics:
                            metrics[f"reward_{obj_name}"] = episode_metrics[reward_key]

                    # Objective weights used for this episode
                    for key, value in episode_metrics.items():
                        if key.endswith("_reward_weights"):
                            metrics[key] = value

                    # Training-specific metrics (loss, entropy, etc.)
                    for metric_key in [
                        "loss",
                        "entropy",
                        "policy_loss",
                        "value_loss",
                        "lr",
                    ]:
                        if metric_key in episode_metrics:
                            metrics[metric_key] = episode_metrics[metric_key]

                # Pareto front metrics
                if "pareto_front_size" in info:
                    metrics["pareto_front_size"] = info["pareto_front_size"]

                # Enhanced Pareto front statistics
                if (
                    hasattr(self.trainer, "pareto_front_rewards")
                    and len(self.trainer.pareto_front_rewards) > 0
                ):
                    pareto_rewards = np.array(self.trainer.pareto_front_rewards)
                    pareto_values = (
                        np.array(self.trainer.pareto_front_values)
                        if hasattr(self.trainer, "pareto_front_values")
                        and len(self.trainer.pareto_front_values) > 0
                        else None
                    )

                    # Get objective names dynamically (use the same logic as above)
                    obj_names = self.trainer.env.get_parameter_names()
                    # Pareto front reward statistics
                    if pareto_rewards.ndim == 2 and len(obj_names) > 0:
                        for i, obj_name in enumerate(
                            obj_names[: pareto_rewards.shape[1]]
                        ):
                            obj_rewards = pareto_rewards[:, i]
                            metrics[f"pareto_reward_{obj_name}_best"] = float(
                                np.max(obj_rewards)
                            )
                            metrics[f"pareto_reward_{obj_name}_worst"] = float(
                                np.min(obj_rewards)
                            )
                            metrics[f"pareto_reward_{obj_name}_mean"] = float(
                                np.mean(obj_rewards)
                            )

                    # Pareto front value statistics (physical values)
                    if (
                        pareto_values is not None
                        and pareto_values.ndim == 2
                        and len(obj_names) > 0
                    ):
                        for i, obj_name in enumerate(
                            obj_names[: pareto_values.shape[1]]
                        ):
                            obj_values = pareto_values[:, i]
                            metrics[f"pareto_value_{obj_name}_best"] = float(
                                np.max(obj_values)
                            )
                            metrics[f"pareto_value_{obj_name}_worst"] = float(
                                np.min(obj_values)
                            )
                            metrics[f"pareto_value_{obj_name}_mean"] = float(
                                np.mean(obj_values)
                            )

                # Log all metrics to MLflow
                if metrics:
                    self.mlflow_tracker.log_training_metrics(episode, metrics)

        # Update callbacks
        self.trainer.callbacks.on_training_start = on_training_start
        self.trainer.callbacks.on_training_complete = on_training_complete
        self.trainer.callbacks.on_episode_complete = enhanced_episode_callback

        try:
            # Initialize Pareto front if not continuing
            if not self.continue_training:
                print("Initializing Pareto front...")
                self.trainer.init_pareto_front(
                    n_solutions=self.config.training.n_init_solutions
                )

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
            if self.enable_mlflow:
                self.mlflow_tracker.log_training_metrics(
                    episode=0, metrics={"training_error": 1, "error_message": str(e)}
                )
            return False
        finally:
            # End MLflow run
            if self.enable_mlflow:
                self.mlflow_tracker.end_run()

    def run_evaluation(self):
        """Run evaluation and enhanced visualization."""
        if self.trainer is None:
            raise RuntimeError("Configuration must be loaded before evaluation")

        print("\n" + "=" * 60)
        print("STARTING EVALUATION")
        print("=" * 60)
        print(f"Configuration: {self.config_path}")
        print(f"Number of samples: {self.n_eval_samples}")
        print(f"Output directory: {self.config.general.root_dir}")
        if self.enable_mlflow:
            print(f"MLflow tracking: Enabled")
        print("=" * 60)

        # Start MLflow run for evaluation
        if self.enable_mlflow:
            run_name = f"evaluation_{time.strftime('%Y%m%d_%H%M%S')}"
            self.mlflow_tracker.start_run(run_name)
            self.mlflow_tracker.log_config_file(self.config_path)
            print(f"MLflow evaluation run started: {self.mlflow_tracker.run_id}")

        try:
            # Run evaluation pipeline
            print(f"Generating {self.n_eval_samples} solution samples...")
            sampled_states, results, sampled_weights = run_evaluation_pipeline(
                self.trainer,
                self.env,
                self.n_eval_samples,
                os.path.join(
                    self.config.general.root_dir, HPPOConstants.EVALUATION_DIR
                ),
            )

            # Log evaluation metrics to MLflow
            if self.enable_mlflow:
                final_pareto_size = (
                    len(self.env.pareto_front)
                    if hasattr(self.env, "pareto_front")
                    else 0
                )
                self.mlflow_tracker.log_evaluation_results(
                    evaluation_dir=os.path.join(
                        self.config.general.root_dir, HPPOConstants.EVALUATION_DIR
                    ),
                    n_samples=self.n_eval_samples,
                    final_pareto_size=final_pareto_size,
                )

                # Log summary metrics
                evaluation_metrics = {
                    "evaluation_samples_generated": len(sampled_states),
                    "final_pareto_front_size": final_pareto_size,
                }

                # Calculate additional evaluation metrics if results are available
                if results:
                    # Assuming results contain objective values
                    objectives = (
                        list(results.keys()) if isinstance(results, dict) else []
                    )
                    for obj in objectives:
                        if obj in results and len(results[obj]) > 0:
                            evaluation_metrics[f"eval_mean_{obj}"] = float(
                                np.mean(results[obj])
                            )
                            evaluation_metrics[f"eval_std_{obj}"] = float(
                                np.std(results[obj])
                            )
                            evaluation_metrics[f"eval_min_{obj}"] = float(
                                np.min(results[obj])
                            )
                            evaluation_metrics[f"eval_max_{obj}"] = float(
                                np.max(results[obj])
                            )

                self.mlflow_tracker.log_training_metrics(0, evaluation_metrics)

                print(
                    f"MLflow evaluation run completed: {self.mlflow_tracker.get_run_info()['tracking_uri']}"
                )

            print("\n" + "=" * 60)
            print("EVALUATION COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"Output directory: {self.config.general.root_dir}")
            print(f"Samples generated: {len(sampled_states)}")
            print(
                f"Enhanced Pareto plots created combining training and evaluation data"
            )
            print("=" * 60)

            return True

        except Exception as e:
            print(f"\nEvaluation failed with error: {e}")
            if self.verbose:
                traceback.print_exc()
            if self.enable_mlflow:
                self.mlflow_tracker.log_training_metrics(
                    0, {"evaluation_error": 1, "error_message": str(e)}
                )
            return False
        finally:
            # End MLflow run
            if self.enable_mlflow:
                self.mlflow_tracker.end_run()


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

  # Run evaluation only with 2000 samples
  coatopt-train -c config.ini --evaluate -n 2000

  # Train with plot saving enabled
  coatopt-train -c config.ini --save-plots

  # Run with minimal output
  coatopt-train -c config.ini --quiet

  # Train with MLflow tracking disabled
  coatopt-train -c config.ini --no-mlflow

  # Train with custom MLflow experiment name
  coatopt-train -c config.ini --mlflow-experiment "my_experiment"

  # Use custom MLflow directory for all experiments
  coatopt-train -c config.ini --mlflow-dir "/path/to/my/experiments"
        """,
    )

    parser.add_argument(
        "-c", "--config", required=True, help="Path to configuration file (required)"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Start training from scratch (ignore existing checkpoints)",
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Run evaluation only (no training)"
    )
    parser.add_argument(
        "-n",
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples for evaluation (default: 1000)",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save training plots and visualizations",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Minimal output (less verbose)"
    )

    # MLflow options
    parser.add_argument(
        "--mlflow", action="store_true", help="Disable MLflow experiment tracking"
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        help="MLflow experiment name (default: auto-generated from config)",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        help="MLflow tracking URI (default: local mlruns directory)",
    )
    parser.add_argument(
        "--mlflow-dir",
        type=str,
        help="MLflow directory path (creates file:// URI, default: ~/mlflow_experiments/coating_optimisation)",
    )

    parser.add_argument(
        "-v", "--version", action="version", version="coatopt-train 1.0.0"
    )

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
            save_plots=args.save_plots,
            evaluate_only=args.evaluate,
            n_eval_samples=args.n_samples,
            enable_mlflow=args.mlflow,
            experiment_name=args.mlflow_experiment,
            mlflow_uri=args.mlflow_uri,
            mlflow_dir=args.mlflow_dir,
        )

        # Load configuration
        trainer.load_configuration()

        # Run appropriate operation
        if args.evaluate:
            success = trainer.run_evaluation()
        else:
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


if __name__ == "__main__":
    main()
