"""Shared utility functions for CoatOpt experiments."""

import json
import math
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def load_materials(path: str) -> dict:
    """Load materials from JSON, converting string keys to int.

    Args:
        path: Path to materials JSON file

    Returns:
        Dictionary mapping material indices (int) to material properties (dict)
    """
    with open(path) as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def evaluate_model(model, env, n_episodes: int = 10, use_action_masks: bool = False):
    """Evaluate trained model.

    Args:
        model: Trained SB3 model
        env: Gymnasium environment to evaluate on
        n_episodes: Number of evaluation episodes
        use_action_masks: Whether to use action masking (for MaskablePPO)

    Returns:
        None (prints evaluation results)
    """
    rewards = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        while not done:
            if use_action_masks and hasattr(env, "action_masks"):
                # MaskablePPO with action masking
                action_masks = env.action_masks()
                action, _ = model.predict(
                    obs, deterministic=True, action_masks=action_masks
                )
            else:
                # Standard PPO
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = done or truncated

        rewards.append(episode_reward)
        vals = info.get("vals", {})
        print(
            f"  Episode {ep + 1}: reward={episode_reward:.4f}, "
            f"steps={steps}, vals={vals}"
        )


def save_run_metadata(
    save_dir: str,
    algorithm_name: str,
    start_time: float,
    end_time: float,
    pareto_front_size: int = None,
    total_episodes: int = None,
    total_generations: int = None,
    config_path: str = None,
    additional_info: dict = None,
):
    """Save run metadata to JSON file.

    Args:
        save_dir: Directory to save metadata
        algorithm_name: Name of algorithm used
        start_time: Start time (from time.time())
        end_time: End time (from time.time())
        pareto_front_size: Final Pareto front size
        total_episodes: Total episodes run (for RL)
        total_generations: Total generations run (for evolutionary)
        config_path: Path to config file used
        additional_info: Additional metadata to include
    """
    save_dir = Path(save_dir)

    duration_seconds = end_time - start_time
    duration_minutes = duration_seconds / 60
    duration_hours = duration_minutes / 60

    metadata = {
        "algorithm": algorithm_name,
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.fromtimestamp(end_time).isoformat(),
        "duration_seconds": round(duration_seconds, 2),
        "duration_minutes": round(duration_minutes, 2),
        "duration_hours": round(duration_hours, 2),
        "pareto_front_size": pareto_front_size,
        "total_episodes": total_episodes,
        "total_generations": total_generations,
        "config_path": str(config_path) if config_path else None,
        "platform": {
            "system": platform.system(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
        },
    }

    # Add any additional info
    if additional_info:
        metadata.update(additional_info)

    # Save to JSON
    metadata_path = save_dir / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, indent=2, fp=f)

    print(f"\nRun metadata saved to {metadata_path}")
    print(f"  Duration: {duration_minutes:.1f} minutes ({duration_hours:.2f} hours)")
    print(f"  Final Pareto front size: {pareto_front_size}")
