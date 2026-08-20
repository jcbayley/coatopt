"""Shared utility functions for CoatOpt experiments."""

import ast
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


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
    except Exception:
        return "unknown"


def load_materials(
    path: str, use_materials: list = None, substrate: str = None
) -> dict:
    """Load materials from a name-keyed JSON library, returning an index-keyed dict.

    The library file maps material name to material properties. "air" must be
    present (case-insensitive) and is always assigned index 0. If use_materials
    is given, only those materials are loaded, indexed 1..N in the order listed;
    otherwise all materials in the file are used in file order. The material at
    index 1 acts as the substrate; pass substrate to select it by name instead
    of by position.

    Legacy index-keyed files ("0", "1", ...) are also accepted and converted
    using each entry's "name" field, preserving their original order.

    Args:
        path: Path to materials JSON library file
        use_materials: Optional list of material names to use from the library
        substrate: Optional name of the substrate material (must be selected)

    Returns:
        Dictionary mapping material indices (int) to material properties (dict)
    """
    with open(path) as f:
        data = json.load(f)

    if data and all(str(k).lstrip("-").isdigit() for k in data):
        data = {
            props["name"]: props
            for _, props in sorted(data.items(), key=lambda kv: int(kv[0]))
        }

    by_lower = {name.lower(): name for name in data}
    air_name = by_lower.get("air")
    if air_name is None:
        raise ValueError(
            f"No 'air' material found in {path}. "
            f"Available materials: {sorted(data)}"
        )

    if use_materials is None:
        selected = [name for name in data if name != air_name]
    else:
        selected = []
        for requested in use_materials:
            name = by_lower.get(str(requested).strip().lower())
            if name is None:
                raise ValueError(
                    f"Material '{requested}' not found in {path}. "
                    f"Available materials: {sorted(data)}"
                )
            if name != air_name and name not in selected:
                selected.append(name)
    if not selected:
        raise ValueError(f"No coating materials selected from {path}")

    if substrate is not None:
        name = by_lower.get(substrate.strip().lower())
        if name is None or name not in selected:
            raise ValueError(
                f"Substrate material '{substrate}' must be one of the "
                f"selected materials: {selected}"
            )
        selected.insert(0, selected.pop(selected.index(name)))

    return {
        index: {**data[name], "name": data[name].get("name", name)}
        for index, name in enumerate([air_name] + selected)
    }


def load_materials_from_parser(parser, config_path: str = None) -> dict:
    """Load materials as configured in a parsed INI config.

    Reads materials_path from [general] (resolved relative to the config file's
    directory, then the working directory), plus the optional [general] options
    materials (list of material names to use, e.g. "SiO2, aSi" or a Python
    list) and substrate_material (name of the substrate material).

    Args:
        parser: configparser.ConfigParser with the config loaded
        config_path: Path of the config file, used to resolve relative paths

    Returns:
        Dictionary mapping material indices (int) to material properties (dict)
    """
    materials_path = parser.get("general", "materials_path", fallback=None)
    if materials_path is None:
        raise ValueError("Config must set materials_path in the [general] section")

    resolved = Path(materials_path).expanduser()
    if not resolved.is_absolute():
        candidates = []
        if config_path is not None:
            candidates.append(Path(config_path).resolve().parent / resolved)
        candidates.append(Path.cwd() / resolved)
        resolved = next((c for c in candidates if c.exists()), resolved)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Materials file not found: {materials_path} "
            f"(tried relative to the config file and the working directory)"
        )

    use_materials = parser.get("general", "materials", fallback=None)
    if use_materials is not None:
        try:
            use_materials = list(ast.literal_eval(use_materials))
        except (ValueError, SyntaxError):
            use_materials = [m.strip() for m in use_materials.split(",") if m.strip()]

    substrate = parser.get("general", "substrate_material", fallback=None)

    materials = load_materials(str(resolved), use_materials, substrate)
    names = ", ".join(f"{i}={props['name']}" for i, props in materials.items())
    print(f"Loaded {len(materials)} materials from {resolved} ({names})")
    return materials


def save_materials_snapshot(materials: dict, save_dir) -> Path:
    """Write the resolved materials into a run directory for reproducibility.

    Saved name-keyed in index order, so loading the snapshot (with or without
    the config's materials selection) reproduces the same material indices.

    Args:
        materials: Index-keyed materials dict as returned by load_materials
        save_dir: Run directory to write materials.json into

    Returns:
        Path of the written snapshot file
    """
    snapshot = {materials[i]["name"]: materials[i] for i in sorted(materials)}
    path = Path(save_dir) / "materials.json"
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=4)
    return path


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

    design_data = []
    value_data = []
    reward_data = []

    for x in X:
        # Extract design variables
        thicknesses = x[: env.max_layers]
        materials_idx = np.floor(x[env.max_layers :]).astype(int)

        # Apply air cascade so saved designs match what was actually evaluated
        air_found = False
        for k in range(env.max_layers):
            if air_found or materials_idx[k] == env.air_material_index:
                air_found = True
                materials_idx[k] = env.air_material_index
                thicknesses[k] = 0.0

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

        # Fill state (air cascade already applied above)
        for k in range(env.max_layers):
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

    # Compute and print hypervolume summary
    try:
        from coatopt.compare_outputs import compute_hypervolume_from_df

        hv_reward = compute_hypervolume_from_df(
            results["pareto_rewards"], space="reward"
        )
        hv_value = compute_hypervolume_from_df(results["pareto_values"], space="value")
        print(f"Hypervolume (reward space): {hv_reward:.6f}")
        print(f"Hypervolume (value space):  {hv_value:.6f}")
    except Exception:
        pass

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
        # total_runtime is the whole train() call, including the checkpoint,
        # CSV and plot writes it does internally. algorithm_runtime is the
        # optimisation alone, reported by the trainer via its metadata dict
        # (None if that trainer does not separate the two yet).
        "total_runtime": round(duration_seconds, 2),
        "algorithm_runtime": None,
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
