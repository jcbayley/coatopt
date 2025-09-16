"""
MLflow tracking utilities for CoatOpt experiments.

This module provides utilities for tracking experiments using MLflow,
including experiment management, parameter logging, metric tracking,
and artifact management.
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from mlflow import MlflowClient

from coatopt.config.structured_config import CoatingOptimisationConfig


class MLflowTracker:
    """
    MLflow experiment tracker for CoatOpt training runs.

    Handles experiment setup, parameter logging, metric tracking,
    and artifact management for coating optimisation experiments.
    """

    def __init__(
        self,
        experiment_name: str = "coating_optimisation",
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[CoatingOptimisationConfig] = None,
        enable_logging: bool = True,
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (defaults to local mlruns directory)
            run_name: Optional name for the run (auto-generated if None)
            config: CoatOpt configuration object
            enable_logging: Whether to enable MLflow logging
        """
        self.enable_logging = enable_logging
        self.config = config
        self.run_id = None
        self.experiment_id = None

        if not self.enable_logging:
            return

        # Set tracking URI
        if tracking_uri is None:
            # Default to local mlruns directory in current working directory
            tracking_uri = f"file://{os.path.abspath('mlruns')}"

        mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except Exception:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)

        # Generate run name if not provided
        if run_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_name = f"coatopt_run_{timestamp}"

        self.run_name = run_name

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new MLflow run."""
        if not self.enable_logging:
            return

        if run_name is not None:
            self.run_name = run_name

        mlflow.start_run(run_name=self.run_name)
        self.run_id = mlflow.active_run().info.run_id

        # Log system info
        self._log_system_info()

        # Log configuration if provided
        if self.config is not None:
            self._log_config_parameters()

    def end_run(self) -> None:
        """End the current MLflow run."""
        if not self.enable_logging:
            return

        if mlflow.active_run() is not None:
            mlflow.end_run()

    def _log_system_info(self) -> None:
        """Log system and environment information."""
        if not self.enable_logging:
            return

        try:
            import torch

            mlflow.log_param("pytorch_version", torch.__version__)
            mlflow.log_param("cuda_available", torch.cuda.is_available())
            if torch.cuda.is_available():
                mlflow.log_param("cuda_device_count", torch.cuda.device_count())
                mlflow.log_param("cuda_device_name", torch.cuda.get_device_name(0))
        except ImportError:
            pass

        mlflow.log_param("python_version", os.sys.version)

    def _log_config_parameters(self) -> None:
        """Log CoatOpt configuration parameters."""
        if not self.enable_logging or self.config is None:
            return

        # General parameters
        mlflow.log_param("n_layers", self.config.data.n_layers)
        mlflow.log_param("optimise_parameters", self.config.data.optimise_parameters)
        mlflow.log_param("materials_file", self.config.general.materials_file)

        # Training parameters
        mlflow.log_param("n_iterations", self.config.training.n_iterations)
        mlflow.log_param("n_init_solutions", self.config.training.n_init_solutions)
        mlflow.log_param(
            "n_episodes_per_epoch", self.config.training.n_episodes_per_epoch
        )
        mlflow.log_param("lr_value", self.config.training.lr_value)
        mlflow.log_param("lr_discrete_policy", self.config.training.lr_discrete_policy)
        mlflow.log_param(
            "lr_continuous_policy", self.config.training.lr_continuous_policy
        )

        # Entropy parameters
        mlflow.log_param("entropy_beta_start", self.config.training.entropy_beta_start)
        mlflow.log_param("entropy_beta_end", self.config.training.entropy_beta_end)
        mlflow.log_param(
            "entropy_beta_decay_length", self.config.training.entropy_beta_decay_length
        )

        # Network parameters
        mlflow.log_param("pre_network_type", self.config.network.pre_network_type)
        mlflow.log_param(
            "discrete_policy_network_nlayer", self.config.network.n_discrete_layers
        )
        mlflow.log_param(
            "continuous_policy_network_nlayer", self.config.network.n_continuous_layers
        )
        mlflow.log_param("value_network_nlayer", self.config.network.n_value_layers)
        mlflow.log_param("value_network_size", self.config.network.value_hidden_size)
        mlflow.log_param(
            "discrete_hidden_size", self.config.network.discrete_hidden_size
        )
        mlflow.log_param(
            "continuous_hidden_size", self.config.network.continuous_hidden_size
        )

        # Environment parameters
        mlflow.log_param("environment_type", self.config.network.model_type)
        mlflow.log_param("min_thickness", self.config.data.min_thickness)
        mlflow.log_param("max_thickness", self.config.data.max_thickness)

        # Log optimization targets
        for param, target in self.config.data.optimise_targets.items():
            mlflow.log_param(f"target_{param}", target)

        # Log design criteria
        for param, criteria in self.config.data.design_criteria.items():
            mlflow.log_param(f"criteria_{param}", criteria)

    def log_training_metrics(
        self, episode: int, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """
        Log training metrics for a specific episode.

        Args:
            episode: Training episode number
            metrics: Dictionary of metric name -> value
            step: Optional step number (uses episode if not provided)
        """
        if not self.enable_logging:
            return

        if step is None:
            step = episode

        for metric_name, value in metrics.items():
            # Handle numpy/torch tensors
            if isinstance(value, (np.ndarray, torch.Tensor)):
                value = float(value.item() if hasattr(value, "item") else value)
            elif isinstance(value, (list, tuple)):
                value = float(value[0]) if len(value) == 1 else str(value)

            mlflow.log_metric(metric_name, value, step=step)

    def log_model_checkpoint(
        self, model: torch.nn.Module, checkpoint_path: str, episode: int
    ) -> None:
        """
        Log model checkpoint as an artifact.

        Args:
            model: PyTorch model to save
            checkpoint_path: Path to the checkpoint file
            episode: Current episode number
        """
        if not self.enable_logging:
            return

        try:
            # Log the checkpoint file as an artifact
            mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")

            # Also log with MLflow's model logging for easy loading
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = os.path.join(temp_dir, "model")
                torch.save(model.state_dict(), model_path)
                mlflow.pytorch.log_model(
                    model,
                    artifact_path=f"model_episode_{episode}",
                    registered_model_name=None,
                )
        except Exception as e:
            print(f"Warning: Failed to log model checkpoint: {e}")

    def log_plots(
        self, plots_dir: str, plot_patterns: List[str] = ["*.png", "*.pdf", "*.svg"]
    ) -> None:
        """
        Log plot files as artifacts.

        Args:
            plots_dir: Directory containing plot files
            plot_patterns: List of file patterns to match
        """
        if not self.enable_logging:
            return

        plots_path = Path(plots_dir)
        if not plots_path.exists():
            return

        for pattern in plot_patterns:
            for plot_file in plots_path.glob(pattern):
                try:
                    mlflow.log_artifact(str(plot_file), artifact_path="plots")
                except Exception as e:
                    print(f"Warning: Failed to log plot {plot_file}: {e}")

    def log_config_file(self, config_path: str) -> None:
        """
        Log the configuration file as an artifact.

        Args:
            config_path: Path to the configuration file
        """
        if not self.enable_logging:
            return

        try:
            mlflow.log_artifact(config_path, artifact_path="config")
        except Exception as e:
            print(f"Warning: Failed to log config file: {e}")

    def log_evaluation_results(
        self, evaluation_dir: str, n_samples: int, final_pareto_size: int
    ) -> None:
        """
        Log evaluation results and artifacts.

        Args:
            evaluation_dir: Directory containing evaluation outputs
            n_samples: Number of evaluation samples
            final_pareto_size: Final Pareto front size
        """
        if not self.enable_logging:
            return

        # Log evaluation metrics
        mlflow.log_metric("evaluation_samples", n_samples)
        mlflow.log_metric("final_pareto_size", final_pareto_size)

        # Log evaluation artifacts
        eval_path = Path(evaluation_dir)
        if eval_path.exists():
            for file in eval_path.rglob("*"):
                if file.is_file():
                    try:
                        rel_path = file.relative_to(eval_path)
                        mlflow.log_artifact(
                            str(file), artifact_path=f"evaluation/{rel_path.parent}"
                        )
                    except Exception as e:
                        print(f"Warning: Failed to log evaluation file {file}: {e}")

    def get_run_info(self) -> Dict[str, Any]:
        """
        Get information about the current run.

        Returns:
            Dictionary with run information
        """
        if not self.enable_logging or self.run_id is None:
            return {}

        client = MlflowClient()
        run = client.get_run(self.run_id)

        return {
            "run_id": self.run_id,
            "run_name": run.info.run_name,
            "experiment_id": self.experiment_id,
            "start_time": run.info.start_time,
            "status": run.info.status,
            "artifact_uri": run.info.artifact_uri,
            "tracking_uri": mlflow.get_tracking_uri(),
        }

    def __enter__(self):
        """Context manager entry."""
        if self.enable_logging:
            self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.enable_logging:
            self.end_run()


def create_mlflow_tracker(
    config: CoatingOptimisationConfig,
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    enable_logging: bool = True,
) -> MLflowTracker:
    """
    Factory function to create an MLflow tracker from config.

    Args:
        config: CoatOpt configuration
        experiment_name: Optional experiment name override
        tracking_uri: Optional tracking URI override
        enable_logging: Whether to enable MLflow logging

    Returns:
        Configured MLflow tracker
    """
    if experiment_name is None:
        # Generate experiment name from config
        env_type = config.network.model_type
        objectives = "_".join(config.data.optimise_parameters)
        experiment_name = f"coatopt_{env_type}_{objectives}"

    return MLflowTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        config=config,
        enable_logging=enable_logging,
    )
