"""Shared utility functions for CoatOpt experiments."""

import json
import math
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback


def get_git_hash() -> str:
    """Get current git commit hash.

    Returns:
        Git commit hash (short), or 'unknown' if not in git repo
    """
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("ascii")
            .strip()
        )
    except:
        return "unknown"


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


def convert_pymoo_to_dataframes(result, env):
    """Convert PyMOO result to standardized DataFrames.

    Args:
        result: PyMOO result object with X (designs) and F (objectives)
        env: CoatingEnvironment instance

    Returns:
        Tuple of (designs_df, values_df, rewards_df)
    """
    from coatopt.environments.state import CoatingState

    X = result.X  # Design variables
    F = result.F  # Objectives (minimized)

    design_data = []
    value_data = []
    reward_data = []

    for x in X:
        # Extract design variables
        thicknesses = x[: env.max_layers]
        materials_idx = np.floor(x[env.max_layers :]).astype(int)

        design_row = {}
        for j in range(env.max_layers):
            design_row[f"thickness_{j}"] = thicknesses[j]
            design_row[f"material_{j}"] = materials_idx[j]

        # Create state and compute rewards/values
        state = CoatingState(
            max_layers=env.max_layers,
            n_materials=env.n_materials,
            air_material_index=env.air_material_index,
            substrate_material_index=env.substrate_material_index,
            materials=env.materials,
        )

        # Fill state (handle air layer constraint)
        air_found = False
        for k in range(env.max_layers):
            if air_found or materials_idx[k] == env.air_material_index:
                air_found = True
                state.set_layer(k, 0.0, env.air_material_index)
            else:
                state.set_layer(k, thicknesses[k], materials_idx[k])

        # Get rewards and values
        normalised_rewards, vals = env.compute_reward(state, normalised=True)

        value_row = {}
        reward_row = {}
        for param in env.optimise_parameters:
            value_row[param] = vals.get(param, 0.0)
            reward_row[param] = normalised_rewards.get(param, 0.0)

        design_data.append(design_row)
        value_data.append(value_row)
        reward_data.append(reward_row)

    designs_df = pd.DataFrame(design_data)
    values_df = pd.DataFrame(value_data)
    rewards_df = pd.DataFrame(reward_data)

    return designs_df, values_df, rewards_df


def save_training_results(
    results: dict,
    save_dir: Path,
    algorithm_name: str,
    start_time: float,
    end_time: float,
    config_path: str,
):
    """Save training results in a standardized format.

    Args:
        results: Dict with keys:
            - 'pareto_designs': DataFrame with design variables
            - 'pareto_values': DataFrame with objective values
            - 'pareto_rewards': DataFrame with normalized rewards
            - 'model': Trained model (or None)
            - 'metadata': Dict with algorithm-specific metadata (optional)
        save_dir: Directory to save results
        algorithm_name: Name of algorithm used
        start_time: Training start time
        end_time: Training end time
        config_path: Path to config file
    """
    import mlflow

    save_dir = Path(save_dir)

    # Save combined Pareto front (designs + values + rewards in one file)
    pareto_path = save_dir / "pareto_front.csv"
    combined_pareto = pd.concat(
        [
            results["pareto_designs"],
            results["pareto_values"],
            results["pareto_rewards"].add_suffix("_reward"),
        ],
        axis=1,
    )
    combined_pareto.to_csv(pareto_path, index=False)
    # pareto_rewards is the canonical count (always populated; designs may be
    # empty for algorithms like PGMORL that use internal vectorised envs)
    pareto_size = len(results["pareto_rewards"])
    print(f"Saved {pareto_size} Pareto solutions to {pareto_path}")

    # Save model if available
    if results["model"] is not None:
        model_path = save_dir / f"{algorithm_name}_model"
        results["model"].save(str(model_path))
        print(f"Model saved to {model_path}")

    # Log to MLflow if enabled
    if mlflow.active_run():
        print("Logging to MLflow...")
        mlflow.log_metric("final_pareto_size", pareto_size)

        # Prefer physical values for stats; fall back to reward vectors
        stats_df = (
            results["pareto_values"]
            if not results["pareto_values"].empty
            else results["pareto_rewards"]
        )
        for col in stats_df.columns:
            mlflow.log_metric(f"pareto_best_{col}", stats_df[col].max())
            mlflow.log_metric(f"pareto_worst_{col}", stats_df[col].min())

    # Save run metadata
    additional_info = results.get("metadata", {}).copy()
    additional_info["training_time_seconds"] = end_time - start_time

    save_run_metadata(
        save_dir=save_dir,
        algorithm_name=algorithm_name,
        start_time=start_time,
        end_time=end_time,
        pareto_front_size=pareto_size,
        total_episodes=additional_info.pop("total_episodes", None),
        total_generations=additional_info.pop("total_generations", None),
        config_path=config_path,
        additional_info=additional_info,
    )


def load_pareto_front(run_dir: Path):
    """Load Pareto front data from a run directory.

    Args:
        run_dir: Directory containing pareto_front.csv

    Returns:
        Tuple of (designs_df, values_df, rewards_df) where:
        - designs_df: DataFrame with design variables (thickness_0, material_0, thickness_1, material_1, ...)
        - values_df: DataFrame with objective values (reflectivity, absorption, etc.)
        - rewards_df: DataFrame with normalized rewards
    """
    run_dir = Path(run_dir)

    # Load combined Pareto front CSV
    pareto_path = run_dir / "pareto_front.csv"

    if not pareto_path.exists():
        raise FileNotFoundError(f"pareto_front.csv not found in {run_dir}")

    combined = pd.read_csv(pareto_path)

    # Separate columns by type
    design_cols = [
        col
        for col in combined.columns
        if col.startswith("thickness_") or col.startswith("material_")
    ]
    reward_cols = [col for col in combined.columns if col.endswith("_reward")]
    value_cols = [
        col
        for col in combined.columns
        if col not in design_cols and col not in reward_cols
    ]

    # Extract dataframes
    designs_df = combined[design_cols].copy()
    values_df = combined[value_cols].copy()

    # Remove _reward suffix from reward columns
    rewards_df = combined[reward_cols].copy()
    rewards_df.columns = [col.replace("_reward", "") for col in rewards_df.columns]

    return designs_df, values_df, rewards_df


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
        "git_hash": get_git_hash(),
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
