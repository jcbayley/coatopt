#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from coatopt.environments.environment import CoatingEnvironment
from coatopt.environments.state import CoatingState
from coatopt.utils.configs import Config, DataConfig, TrainingConfig, load_config
from coatopt.utils.utils import load_materials
from coatopt.environments.state import CoatingState


class CoatingOptimizationProblem(ElementwiseProblem):
    """PyMOO problem wrapper for coating optimization."""

    def __init__(self, env: CoatingEnvironment):
        """Initialize optimization problem.

        Args:
            env: CoatingEnvironment instance
        """
        self.env = env

        # Variables: [thicknesses (max_layers), materials (max_layers)]
        self.n_var = env.max_layers * 2
        n_obj = len(env.optimise_parameters)

        # Define bounds
        thick_lower = np.repeat(env.min_thickness, env.max_layers)
        thick_upper = np.repeat(env.max_thickness, env.max_layers)
        material_lower = np.repeat(0, env.max_layers)
        material_upper = np.repeat(env.n_materials - 0.001, env.max_layers)  # Slightly less to avoid index error

        xl = np.concatenate((thick_lower, material_lower))
        xu = np.concatenate((thick_upper, material_upper))

        super().__init__(n_var=self.n_var, n_obj=n_obj, xl=xl, xu=xu)

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        """Evaluate objectives for given design variables.

        Args:
            x: Design variables [thicknesses..., materials...]
            out: Output dictionary for objectives
        """
        # Decode variables
        thicknesses = x[:self.env.max_layers]
        material_indices = np.floor(x[self.env.max_layers:]).astype(int)

        # Create coating state
        state = CoatingState(
            max_layers=self.env.max_layers,
            n_materials=self.env.n_materials,
            air_material_index=self.env.air_material_index,
            substrate_material_index=self.env.substrate_material_index,
            materials=self.env.materials,
        )

        # Air constraint: all layers after first air must be air
        air_found = False
        for i in range(self.env.max_layers):
            if air_found or material_indices[i] == self.env.air_material_index:
                air_found = True
                state.set_layer(i, 0.0, self.env.air_material_index)
            else:
                state.set_layer(i, thicknesses[i], material_indices[i])

        # Compute base rewards using environment's method (normalised=True)
        normalised_rewards, vals = self.env.compute_reward(state, normalised=True)

        # Build objectives from normalised rewards
        objectives = []
        for param in self.env.optimise_parameters:
            normalised_reward = normalised_rewards.get(param, 0.0)
            objectives.append(-normalised_reward)  # Negate for PyMOO minimization

        out["F"] = np.array(objectives)


class CoatingRepair(Repair):
    """Repair operator to enforce coating design constraints.

    Constraints:
    1. No consecutive layers can have the same material (except air)
    2. All layers after first air layer must be air
    """

    def __init__(self, env: CoatingEnvironment):
        super().__init__()
        self.env = env

    def _do(self, problem, X, **kwargs):
        """Repair population X to satisfy constraints."""
        # X shape: (population_size, n_var)
        for i in range(X.shape[0]):
            X[i] = self._repair_individual(X[i])
        return X

    def _repair_individual(self, x: np.ndarray) -> np.ndarray:
        """Repair a single individual to prevent consecutive same materials.

        Note: Air layer handling is done in _evaluate, not here.
        """
        materials_continuous = x[self.env.max_layers:].copy()
        materials_idx = np.floor(materials_continuous).astype(int)

        # Fix consecutive same materials (excluding air)
        for j in range(1, len(materials_idx)):
            if (materials_idx[j] == materials_idx[j-1] and
                materials_idx[j] != self.env.air_material_index):
                # Change to a different random material
                available = [m for m in range(self.env.n_materials)
                           if m != materials_idx[j-1]]
                if available:
                    materials_idx[j] = np.random.choice(available)

        # Write back repaired material indices
        # Convert to continuous representation (add 0.5 so floor gives correct int)
        x[self.env.max_layers:] = materials_idx + 0.5

        return x


def train_genetic(config_path: str):
    """Train genetic algorithm on CoatOpt environment.

    Args:
        config_path: Path to config INI file

    Returns:
        PyMOO result object
    """
    import configparser

    parser = configparser.ConfigParser()
    parser.read(config_path)

    # [General] section
    save_dir = parser.get('General', 'save_dir')
    materials_path = parser.get('General', 'materials_path')

    # [nsga2] section
    total_generations = parser.getint('nsga2', 'n_generations')
    population_size = parser.getint('nsga2', 'population_size')
    algorithm = parser.get('nsga2', 'algorithm')
    seed = parser.getint('nsga2', 'seed')
    crossover_prob = parser.getfloat('nsga2', 'crossover_probability')
    crossover_eta = parser.getfloat('nsga2', 'crossover_eta')
    mutation_prob = parser.getfloat('nsga2', 'mutation_probability')
    mutation_eta = parser.getfloat('nsga2', 'mutation_eta')
    verbose = True

    # [Data] section
    n_layers = parser.getint('Data', 'n_layers')

    # Setup
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load materials
    materials = load_materials(str(materials_path))

    # Load config from file
    config = load_config(config_path)
    config.data.n_layers = n_layers

    n_partitions = None
    n_neighbors = 20
    prob_neighbor_mating = 0.7

    # Create environment
    env = CoatingEnvironment(config, materials)

    # Create problem
    problem = CoatingOptimizationProblem(env)

    # Create repair operator
    repair = CoatingRepair(env)

    # Create algorithm
    if mutation_prob is None:
        mutation_prob = 1.0 / problem.n_var

    if algorithm == "NSGA2":
        algo = NSGA2(
            pop_size=population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=crossover_prob, eta=crossover_eta),
            mutation=PM(prob=mutation_prob, eta=mutation_eta),
            repair=repair,
            eliminate_duplicates=True,
        )
    elif algorithm == "NSGA3":
        if n_partitions is None:
            n_partitions = 12  # Default for 2 objectives
        ref_dirs = get_reference_directions(
            "uniform", len(env.optimise_parameters), n_partitions=n_partitions
        )
        algo = NSGA3(
            pop_size=population_size,
            ref_dirs=ref_dirs,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=crossover_prob, eta=crossover_eta),
            mutation=PM(prob=mutation_prob, eta=mutation_eta),
            repair=repair,
            eliminate_duplicates=True,
        )
    elif algorithm == "MOEAD":
        if n_partitions is None:
            n_partitions = population_size
        ref_dirs = get_reference_directions(
            "uniform", len(env.optimise_parameters), n_partitions=n_partitions
        )
        algo = MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=n_neighbors,
            prob_neighbor_mating=prob_neighbor_mating,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=crossover_prob, eta=crossover_eta),
            mutation=PM(prob=mutation_prob, eta=mutation_eta),
            repair=repair,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    if verbose:
        print(f"\nStarting {algorithm} optimization:")
        print(f"  Population size: {population_size}")
        print(f"  Generations: {total_generations}")
        print(f"  Crossover prob: {crossover_prob}")
        print(f"  Mutation prob: {mutation_prob}")

    # Run optimization
    result = minimize(
        problem,
        algo,
        ("n_gen", total_generations),
        seed=seed,
        verbose=verbose,
    )

    if verbose:
        print(f"\nOptimization complete!")
        print(f"Pareto front size: {len(result.F)}")

    # Save results
    save_results(result, env, materials, save_dir, verbose)

    return result


def save_results(result, env, materials, save_dir: Path, verbose: bool = True):
    """Save optimization results in both val and reward space."""

    # Extract Pareto front
    X = result.X  # Design variables
    F = result.F  # Objectives (minimized)

    if verbose:
        print(f"\nSaving results to {save_dir}")

    # Create results dataframe with both vals and rewards
    data = []
    for i, (x, f) in enumerate(zip(X, F)):
        row = {}

        # Design variables
        thicknesses = x[:env.max_layers]
        materials_idx = np.floor(x[env.max_layers:]).astype(int)

        for j in range(env.max_layers):
            row[f"thickness_{j}"] = thicknesses[j]
            row[f"material_{j}"] = materials_idx[j]


        state = CoatingState(
            max_layers=env.max_layers,
            n_materials=env.n_materials,
            air_material_index=env.air_material_index,
            substrate_material_index=env.substrate_material_index,
            materials=env.materials,
        )

        # Fill state with design
        air_found = False
        for k in range(env.max_layers):
            if air_found or materials_idx[k] == env.air_material_index:
                air_found = True
                state.set_layer(k, 0.0, env.air_material_index)
            else:
                state.set_layer(k, thicknesses[k], materials_idx[k])

        # Get base rewards and values from environment (normalised=True)
        normalised_rewards, vals = env.compute_reward(state, normalised=True)

        # Store values and normalised rewards
        for param in env.optimise_parameters:
            val = vals.get(param, 0.0)
            row[f"{param}_val"] = val
            row[f"{param}_reward"] = normalised_rewards.get(param, 0.0)

        data.append(row)

    df = pd.DataFrame(data)

    # Save combined CSV (for reference)
    combined_csv_path = save_dir / "pareto_front.csv"
    df.to_csv(combined_csv_path, index=False)

    # Save separate CSV files for values and rewards
    value_cols = [col for col in df.columns if col.endswith('_val')]
    design_cols = [col for col in df.columns if col.startswith('thickness_') or col.startswith('material_')]

    values_df = df[design_cols + value_cols].copy()
    # Rename _val columns to remove suffix for compatibility
    values_df.columns = [col.replace('_val', '') if col.endswith('_val') else col for col in values_df.columns]
    values_csv_path = save_dir / "pareto_front_values.csv"
    values_df.to_csv(values_csv_path, index=False)

    # Extract reward columns
    reward_cols = [col for col in df.columns if col.endswith('_reward')]
    rewards_df = df[design_cols + reward_cols].copy()
    rewards_csv_path = save_dir / "pareto_front_rewards.csv"
    rewards_df.to_csv(rewards_csv_path, index=False)

    # Plot Pareto fronts (both vals and rewards)
    if len(env.optimise_parameters) >= 2:
        plot_pareto_front(df, env.optimise_parameters, save_dir, plot_type="vals")
        plot_pareto_front(df, env.optimise_parameters, save_dir, plot_type="rewards")
        if verbose:
            print(f"  Saved Pareto front plots (vals + rewards)")

    # Plot a few sample designs
    n_samples = min(5, len(X))
    for i in range(n_samples):
        plot_coating_stack(
            X[i], env, materials, save_dir / f"stack_{i}.png"
        )


def plot_pareto_front(df: pd.DataFrame, objectives: list, save_dir: Path, plot_type: str = "vals"):
    """Plot Pareto front in either vals or rewards space.

    Args:
        df: DataFrame with both {param}_val and {param}_reward columns
        objectives: List of objective names
        save_dir: Directory to save plot
        plot_type: Either "vals" (physical values) or "rewards" (log-transformed rewards)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    obj1, obj2 = objectives[0], objectives[1]

    if plot_type == "vals":
        # Plot physical values
        x = df[f"{obj1}_val"].values
        y = df[f"{obj2}_val"].values

        # Handle reflectivity specially
        if obj1 == "reflectivity":
            x = 1 - x  # Plot as loss
            xlabel = "1 - Reflectivity"
        else:
            xlabel = obj1.replace("_", " ").title()

        if obj2 == "absorption":
            ylabel = "Absorption (ppm)"
        elif obj2 == "thermal_noise":
            ylabel = "Thermal Noise (m/√Hz)"
        else:
            ylabel = obj2.replace("_", " ").title()

        color = "red"
        filename = "pareto_front_vals.png"
        title_suffix = "Physical Values"

    elif plot_type == "rewards":
        # Plot rewards
        x = df[f"{obj1}_reward"].values
        y = df[f"{obj2}_reward"].values

        xlabel = f"{obj1.replace('_', ' ').title()} Reward"
        ylabel = f"{obj2.replace('_', ' ').title()} Reward"

        color = "blue"
        filename = "pareto_front_rewards.png"
        title_suffix = "Rewards"
    else:
        raise ValueError(f"plot_type must be 'vals' or 'rewards', got {plot_type}")

    ax.scatter(x, y, c=color, s=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Pareto Front - {title_suffix} ({len(df)} solutions)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_coating_stack(x: np.ndarray, env: CoatingEnvironment, materials: dict, save_path: Path):
    """Plot coating stack design."""
    fig, ax = plt.subplots(figsize=(8, 10))

    thicknesses = x[:env.max_layers] * 1e9  # Convert to nm
    material_indices = np.floor(x[env.max_layers:]).astype(int)

    # Material colors
    colors = {
        0: "lightgray",
        1: "steelblue",
        2: "coral",
        3: "mediumseagreen",
        4: "gold",
        5: "mediumpurple",
    }

    # Plot stack from bottom to top
    y_pos = 0
    for i, (thickness, mat_idx) in enumerate(zip(thicknesses, material_indices)):
        if thickness < 1e-3:  # Skip very thin layers (likely air)
            continue

        color = colors.get(mat_idx, "gray")
        mat_name = materials.get(mat_idx, {}).get("name", f"M{mat_idx}")

        ax.bar(
            0, thickness, bottom=y_pos, width=0.6,
            color=color, edgecolor="black", linewidth=0.5,
            label=mat_name if i == 0 or mat_idx not in material_indices[:i] else ""
        )
        y_pos += thickness

    ax.set_ylabel("Thickness (nm)")
    ax.set_title("Coating Stack Design")
    ax.set_xticks([])
    ax.set_xlim(-0.5, 0.5)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Genetic Algorithm on CoatOpt")
    parser.add_argument(
        "--generations", type=int, default=100, help="Number of generations"
    )
    parser.add_argument(
        "--population", type=int, default=100, help="Population size"
    )
    parser.add_argument(
        "--layers", type=int, default=20, help="Number of coating layers"
    )
    parser.add_argument(
        "--materials", type=str, default=None, help="Path to materials JSON"
    )
    parser.add_argument(
        "--save-dir", type=str, default="./genetic_output", help="Output directory"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="NSGA2",
        choices=["NSGA2", "NSGA3", "MOEAD"],
        help="Genetic algorithm to use",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--crossover-prob", type=float, default=0.9, help="Crossover probability"
    )
    parser.add_argument(
        "--mutation-prob", type=float, default=None, help="Mutation probability (default: 1/n_var)"
    )

    args = parser.parse_args()

    train_genetic(
        total_generations=args.generations,
        population_size=args.population,
        n_layers=args.layers,
        materials_path=args.materials,
        save_dir=args.save_dir,
        algorithm=args.algorithm,
        seed=args.seed,
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
    )
