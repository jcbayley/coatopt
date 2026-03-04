#!/usr/bin/env python3
"""
Genetic algorithms (NSGA-II, NSGA-III, MOEA/D) for multi-objective coating optimization.

Uses PyMOO to optimize coating designs with repair operators enforcing:
- No consecutive same materials (except air)
- All layers after first air must be air
- No air until min_layers_before_air reached

Config section: [nsga2]
  n_generations            = 100
  population_size          = 100
  algorithm                = NSGA2          # NSGA2, NSGA3, or MOEAD
  seed                     = 42
  crossover_probability    = 0.9
  crossover_eta            = 15.0
  mutation_probability     = None           # Default: 1/n_var
  mutation_eta             = 20.0
  min_layers_before_air    = 0              # Min layers before air allowed
"""
import os
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
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
from coatopt.utils.plotting import plot_coating_stack, plot_pareto_front
from coatopt.utils.utils import convert_pymoo_to_dataframes, load_materials


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
        material_upper = np.repeat(
            env.n_materials - 0.001, env.max_layers
        )  # Slightly less to avoid index error

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
        thicknesses = x[: self.env.max_layers]
        material_indices = np.floor(x[self.env.max_layers :]).astype(int)

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
    3. Air cannot be selected until min_layers_before_air is reached
    """

    def __init__(self, env: CoatingEnvironment, min_layers_before_air: int = 0):
        super().__init__()
        self.env = env
        self.min_layers_before_air = min_layers_before_air

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
        materials_continuous = x[self.env.max_layers :].copy()
        materials_idx = np.floor(materials_continuous).astype(int)

        # Fix consecutive same materials (excluding air)
        for j in range(1, len(materials_idx)):
            if (
                materials_idx[j] == materials_idx[j - 1]
                and materials_idx[j] != self.env.air_material_index
            ):
                # Change to a different random material
                available = [
                    m for m in range(self.env.n_materials) if m != materials_idx[j - 1]
                ]
                if available:
                    materials_idx[j] = np.random.choice(available)

        # Prevent air in first N layers
        if self.min_layers_before_air > 0:
            for j in range(min(self.min_layers_before_air, len(materials_idx))):
                if materials_idx[j] == self.env.air_material_index:
                    # Replace with random non-air material
                    available = [
                        m
                        for m in range(self.env.n_materials)
                        if m != self.env.air_material_index
                    ]
                    if available:
                        materials_idx[j] = np.random.choice(available)

        # Write back repaired material indices
        # Convert to continuous representation (add 0.5 so floor gives correct int)
        x[self.env.max_layers :] = materials_idx + 0.5

        return x


def train_genetic(config_path: str, save_dir: Optional[str] = None):
    """Train genetic algorithm on CoatOpt environment.

    Args:
        config_path: Path to config INI file
        save_dir: Directory to save results. If None, reads from config file.

    Returns:
        PyMOO result object
    """
    import configparser

    parser = configparser.ConfigParser()
    parser.read(config_path)

    # [General] section
    if save_dir is None:
        save_dir = parser.get("general", "save_dir")
    materials_path = parser.get("general", "materials_path")

    # [nsga2] section
    total_generations = parser.getint("nsga2", "n_generations")
    population_size = parser.getint("nsga2", "population_size")
    algorithm = parser.get("nsga2", "algorithm")
    seed = parser.getint("nsga2", "seed")
    crossover_prob = parser.getfloat("nsga2", "crossover_probability")
    crossover_eta = parser.getfloat("nsga2", "crossover_eta")
    mutation_prob = parser.getfloat("nsga2", "mutation_probability")
    mutation_eta = parser.getfloat("nsga2", "mutation_eta")
    min_layers_before_air = parser.getint("nsga2", "min_layers_before_air", fallback=0)
    verbose = True

    # [Data] section
    n_layers = parser.getint("data", "n_layers")

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
    repair = CoatingRepair(env, min_layers_before_air=min_layers_before_air)

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
    start_time = time.time()
    result = minimize(
        problem,
        algo,
        ("n_gen", total_generations),
        seed=seed,
        verbose=verbose,
    )
    end_time = time.time()

    if verbose:
        print(f"\nOptimization complete!")
        print(f"Pareto front size: {len(result.F)}")

    # Convert PyMOO results to standardized DataFrames
    designs_df, values_df, rewards_df = convert_pymoo_to_dataframes(result, env)

    # Optional: Plot Pareto fronts and sample designs
    if save_dir and len(env.optimise_parameters) >= 2:
        save_dir = Path(save_dir)
        # Create combined df for plotting
        plot_df = designs_df.copy()
        for col in values_df.columns:
            plot_df[f"{col}_val"] = values_df[col]
        for col in rewards_df.columns:
            plot_df[f"{col}_reward"] = rewards_df[col]

        plot_pareto_front(plot_df, env.optimise_parameters, save_dir, plot_type="vals")
        plot_pareto_front(
            plot_df, env.optimise_parameters, save_dir, plot_type="rewards"
        )
        if verbose:
            print(f"  Saved Pareto front plots")

        # Plot sample designs
        n_samples = min(5, len(result.X))
        for i in range(n_samples):
            x = result.X[i]
            thicknesses = x[: env.max_layers]
            materials_idx = np.floor(x[env.max_layers :]).astype(int)
            plot_coating_stack(
                thicknesses, materials_idx, materials, save_dir / f"stack_{i}.png"
            )

    return {
        "pareto_designs": designs_df,
        "pareto_values": values_df,
        "pareto_rewards": rewards_df,
        "model": None,
        "metadata": {
            "algorithm": algorithm,
            "total_generations": total_generations,
            "population_size": population_size,
            "crossover_prob": crossover_prob,
            "mutation_prob": mutation_prob,
            "seed": seed,
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Genetic Algorithm on CoatOpt")
    parser.add_argument(
        "--generations", type=int, default=100, help="Number of generations"
    )
    parser.add_argument("--population", type=int, default=100, help="Population size")
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
        "--mutation-prob",
        type=float,
        default=None,
        help="Mutation probability (default: 1/n_var)",
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
