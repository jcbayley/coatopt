"""
Training Context - Centralized training state for all optimization strategies.

The TrainingContext acts as a single source of truth for training-related state
that needs to be shared between trainers and environments. This decouples the
trainer and environment while providing clean access to training information.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TrainingContext:
    """
    Centralized training state that both trainer and environment can access.

    This context object is passed between trainer and environment to provide
    clean access to training state without tight coupling.
    """

    # Training metrics storage
    training_metrics: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    episode_count: int = 0
    start_episode: int = 0
    current_episode: int = 0

    # Performance tracking
    best_reward: float = -np.inf
    best_state: Optional[np.ndarray] = None
    episode_rewards: List[float] = field(default_factory=list)
    episode_times: List[float] = field(default_factory=list)

    # Multi-objective tracking
    pareto_front_rewards: np.ndarray = field(default_factory=lambda: np.array([]))
    pareto_front_values: np.ndarray = field(default_factory=lambda: np.array([]))
    pareto_states: np.ndarray = field(default_factory=lambda: np.array([]))
    reference_point: np.ndarray = field(default_factory=lambda: np.array([]))
    all_rewards: List[List[float]] = field(default_factory=list)
    all_values: List[List[float]] = field(default_factory=list)

    # Training configuration
    training_config: Dict[str, Any] = field(default_factory=dict)
    environment_config: Dict[str, Any] = field(default_factory=dict)

    # Best states history (tuples of (reward, episode, state, rewards_dict, vals_dict))
    best_states: List[tuple] = field(default_factory=list)

    # Constraint tracking (for preference-constrained methods)
    constraint_history: List[Dict[str, Any]] = field(default_factory=list)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_episode_metrics(self, episode: int, metrics: Dict[str, Any]) -> None:
        """Add metrics for a single episode."""
        metrics["episode"] = episode
        new_row = pd.DataFrame([metrics])
        self.training_metrics = pd.concat(
            [self.training_metrics, new_row], ignore_index=True
        )
        self.current_episode = episode

        # Track episode rewards
        if "reward" in metrics:
            self.episode_rewards.append(metrics["reward"])

            # Update best reward/state tracking
            if metrics["reward"] > self.best_reward:
                self.best_reward = metrics["reward"]
                if "final_state" in metrics:
                    self.best_state = metrics["final_state"]

    def update_pareto_data(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        states: Optional[np.ndarray] = None,
    ) -> None:
        """Update Pareto front data."""
        self.pareto_front_rewards = rewards
        self.pareto_front_values = values
        if states is not None:
            self.pareto_states = states

        # Update reference point
        if len(rewards) > 0:
            self.reference_point = np.max(rewards, axis=0) * 1.1

    def add_constraint_entry(
        self,
        episode: int,
        phase: int,
        target_objective: str,
        constraints: Dict[str, Any],
        reward_bounds: Dict[str, Any],
    ) -> None:
        """Add constraint history entry."""
        entry = {
            "episode": episode,
            "phase": phase,
            "target_objective": target_objective,
            "constraints": constraints,
            "reward_bounds": reward_bounds,
        }
        self.constraint_history.append(entry)

    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get the most recent episode metrics."""
        if len(self.training_metrics) == 0:
            return {}
        return self.training_metrics.iloc[-1].to_dict()

    def get_metric_history(self, metric_name: str) -> List[float]:
        """Get history of a specific metric."""
        if metric_name in self.training_metrics.columns:
            return self.training_metrics[metric_name].tolist()
        return []

    def get_episode_range(self) -> tuple:
        """Get the range of episodes (start, end)."""
        if len(self.training_metrics) == 0:
            return (self.start_episode, self.start_episode)
        return (self.start_episode, self.current_episode)

    def clear_metrics(self) -> None:
        """Clear all metrics (for new training run)."""
        self.training_metrics = pd.DataFrame()
        self.episode_rewards = []
        self.episode_times = []
        self.best_reward = -np.inf
        self.best_state = None
        self.current_episode = 0
        self.best_states = []
        self.constraint_history = []

    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of the current training state."""
        return {
            "total_episodes": len(self.training_metrics),
            "current_episode": self.current_episode,
            "best_reward": self.best_reward,
            "pareto_front_size": len(self.pareto_front_rewards),
            "metrics_available": list(self.training_metrics.columns),
            "episode_range": self.get_episode_range(),
        }
