#!/usr/bin/env python3
"""
Genetic algorithms (NSGA-II, NSGA-III, MOEA/D) for multi-objective coating optimization.

Uses PyMOO to optimize coating designs with repair operators enforcing:
- No consecutive same materials (except air)
- All layers after first air must be air
- No air until min_layers_before_air reached

Config section: [nsga2] or [moead]
  n_generations            = 100
  population_size          = 100            # NSGA2/NSGA3 only
  algorithm                = NSGA2          # NSGA2, NSGA3, or MOEAD
  seed                     = 42
  crossover_probability    = 0.9
  crossover_eta            = 15.0
  mutation_probability     =                # Blank/None: 1/n_var
  mutation_eta             = 20.0
  min_layers_before_air    = 0              # Min layers before air allowed
  n_partitions             = 12             # NSGA3/MOEAD reference directions
  n_neighbors              = 20             # MOEAD only
  prob_neighbor_mating     = 0.9            # MOEAD only
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from coatopt.environments.environment import CoatingEnvironment
from coatopt.environments.state import CoatingState
from coatopt.utils.configs import load_config
from coatopt.utils.metrics import compute_hypervolume
from coatopt.utils.plotting import plot_coating_stack, plot_pareto_front
from coatopt.utils.utils import convert_pymoo_to_dataframes, load_materials_from_parser


class HypervolumeHistory(Callback):
    """Record hypervolume against evaluation count as the search runs.

    Wall-clock is a poor axis for comparing an evolutionary search against an
    RL one, since they differ by an order of magnitude in designs evaluated per
    second. Evaluations put both on the same footing, so record them here in
    the same reward space and with the same reference point the RL trainer
    logs (F is the negated normalised reward, so -F is directly comparable).

    Sampled every `every` generations: hypervolume is quadratic in front size,
    and on a 1000-member front computing it every generation costs more than
    the search itself.
    """

    def __init__(self, n_objectives: int, every: int = 10):
        super().__init__()
        self.every = max(1, every)
        self.ref_point = np.zeros(n_objectives)
        self.rows = []

    def notify(self, algorithm):
        generation = algorithm.n_gen
        if generation != 1 and generation % self.every:
            return
        front = algorithm.opt.get("F") if algorithm.opt is not None else None
        if front is None or len(front) == 0:
            return
        try:
            hv = compute_hypervolume(
                -np.asarray(front, dtype=float), self.ref_point, maximize=True
            )
        except Exception:
            return
        self.rows.append(
            {
                "generation": int(generation),
                "evaluations": int(algorithm.evaluator.n_eval),
                "pareto.size": int(len(front)),
                "pareto.hypervolume": float(hv),
            }
        )


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
        """Repair a single individual to enforce all design constraints."""
        thicknesses = x[: self.env.max_layers].copy()
        materials_continuous = x[self.env.max_layers :].copy()
        materials_idx = np.floor(materials_continuous).astype(int)

        # Fix consecutive same materials (excluding air); never pick air as
        # replacement to avoid prematurely terminating the coating stack
        for j in range(1, len(materials_idx)):
            if (
                materials_idx[j] == materials_idx[j - 1]
                and materials_idx[j] != self.env.air_material_index
            ):
                available = [
                    m
                    for m in range(self.env.n_materials)
                    if m != materials_idx[j - 1] and m != self.env.air_material_index
                ]
                if not available:  # fallback: only exclude previous material
                    available = [
                        m
                        for m in range(self.env.n_materials)
                        if m != materials_idx[j - 1]
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

        x[: self.env.max_layers] = thicknesses
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

    # [nsga2] section, or [moead] for MOEA/D runs
    section = "moead" if parser.has_section("moead") else "nsga2"
    total_generations = parser.getint(section, "n_generations")
    # MOEA/D takes its population from the reference directions instead
    population_size = parser.getint(section, "population_size", fallback=0)
    algorithm = parser.get(section, "algorithm")
    seed = parser.getint(section, "seed")
    crossover_prob = parser.getfloat(section, "crossover_probability")
    crossover_eta = parser.getfloat(section, "crossover_eta")
    # Blank or "None" means the pymoo default of 1/n_var, applied below
    mutation_prob_raw = parser.get(section, "mutation_probability", fallback="")
    mutation_prob = (
        None
        if mutation_prob_raw.strip().lower() in ("", "none")
        else float(mutation_prob_raw)
    )
    mutation_eta = parser.getfloat(section, "mutation_eta")
    min_layers_before_air = parser.getint(section, "min_layers_before_air", fallback=0)
    verbose = True

    # [Data] section
    n_layers = parser.getint("data", "n_layers")

    # Setup
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load materials
    materials = load_materials_from_parser(parser, config_path)

    # Load config from file
    config = load_config(config_path)
    config.data.n_layers = n_layers

    # Reference directions for NSGA-III/MOEA/D; defaults are pymoo's own
    n_partitions = parser.getint(section, "n_partitions", fallback=12)
    n_neighbors = parser.getint(section, "n_neighbors", fallback=20)
    prob_neighbor_mating = parser.getfloat(
        section, "prob_neighbor_mating", fallback=0.9
    )

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
        ref_dirs = get_reference_directions(
            "uniform", len(env.optimise_parameters), n_partitions=n_partitions
        )
        # One subproblem per direction, so the directions set the population
        population_size = len(ref_dirs)
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

    # Run optimization. algorithm_runtime covers this alone; the Pareto and
    # stack plots below are excluded so the figure is comparable across runs.
    history = HypervolumeHistory(
        n_objectives=len(env.optimise_parameters),
        every=parser.getint(section, "hypervolume_freq", fallback=10),
    )
    opt_start = time.perf_counter()
    result = minimize(
        problem,
        algo,
        ("n_gen", total_generations),
        seed=seed,
        verbose=verbose,
        callback=history,
    )
    algorithm_runtime = time.perf_counter() - opt_start

    # Same filename and columns the RL trainer writes, so anything reading a
    # run directory gets one format regardless of which algorithm produced it.
    if save_dir and history.rows:
        import pandas as pd

        history_path = Path(save_dir)
        history_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history.rows).to_csv(
            history_path / "training_history.csv", index=False
        )
        if verbose:
            print(f"  Saved hypervolume history ({len(history.rows)} points)")

    if verbose:
        print("\nOptimization complete!")
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
            print("  Saved Pareto front plots")

        # Plot sample designs
        n_samples = min(5, len(result.X))
        for i in range(n_samples):
            x = result.X[i]
            thicknesses = x[: env.max_layers].copy()
            materials_idx = np.floor(x[env.max_layers :]).astype(int)
            # Apply air cascade (already done by repair, but be explicit for plots)
            air_found = False
            for k in range(env.max_layers):
                if air_found or materials_idx[k] == env.air_material_index:
                    air_found = True
                    materials_idx[k] = env.air_material_index
                    thicknesses[k] = 0.0
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
            "algorithm_runtime": round(algorithm_runtime, 2),
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
