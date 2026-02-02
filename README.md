# CoatOpt: Multi-Objective Coating Optimization

A test bed for comparing Reinforcement Learning and Evolutionary algorithms on multi-objective optimization problems for designing gravitational wave detector mirror coatings.

## Overview

CoatOpt provides a unified framework for testing and comparing different optimization algorithms on coating stack design:

**Algorithms Supported:**
- **RL Algorithms**: PPO, DQN, MORL (Stable-Baselines3)
- **Evolutionary Algorithms**: NSGA-II, MOEA/D

**Optimization Objectives:**
- **Reflectivity**: Maximize mirror reflectance (↑)
- **Absorption**: Minimize optical absorption (↓)
- **Thermal Noise**: Minimize Brownian thermal noise (↓)

All algorithms maintain Pareto fronts of non-dominated solutions for multi-objective optimization.

## Installation

CoatOpt uses **uv** as the package manager. Python 3.9+ is required.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up the project
cd coatopt_simple
uv sync
```

## Quick Start

### Running Experiments

All algorithms use a unified run script with config files:

```bash
# Run with config file
uv run python -m coatopt.run --config experiments/config_sb3_discrete.ini

# Or use the shorter form
uv run python -m coatopt.run --config experiments/config_genetic.ini
```

### Example Config

Create a config file (e.g., `my_experiment.ini`):

```ini
[General]
save_dir = ./runs
materials_path = src/coatopt/config/materials.json
run_name = test1
# Optional: experiment_name = 20layer-0.1-0.5 (auto-generated if not set)

[Data]
n_layers = 20
min_thickness = 0.1
max_thickness = 0.5
optimise_parameters = reflectivity, absorption
optimise_targets = {"reflectivity": 1.0, "absorption": 0.0}
objective_bounds = {"reflectivity": [0.9, 0.99999], "absorption": [1e-6, 100e-6]}
constraint_schedule = interleaved

[sb3_discrete]
total_timesteps = 100000
n_thickness_bins = 20
epochs_per_step = 200
steps_per_objective = 10
constraint_penalty = 10.0
max_entropy = 0.2
min_entropy = 0.01
mask_consecutive_materials = True
mask_air_until_min_layers = True
min_layers_before_air = 4
verbose = 1

[Algorithm]
# PPO hyperparameters
learning_rate = 3e-4
n_steps = 2048
batch_size = 64
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 0.5
net_arch_pi = [256, 256]
net_arch_vf = [256, 256]
```

### Directory Structure

The framework organizes results to mirror MLflow's experiment/run hierarchy:

```
runs/
  └── {experiment_name}/          
      ├── 20240202-sb3_discrete-test1/    # Run 1: PPO discrete
      ├── 20240202-genetic-baseline/      # Run 2: NSGA-II
      └── 20240203-moead-tuned/           # Run 3: MOEA/D
```

Each run directory contains:
- `config.ini` - Config file backup
- `pareto_front.csv` - Pareto-optimal solutions
- `*.png` - Training plots and visualizations

## MLflow Tracking

All runs are automatically tracked in MLflow:

```bash
# View results in MLflow UI
uv run mlflow ui

# Open browser to http://localhost:5000
```

**MLflow Organization:**
- **Experiment** = Problem definition (e.g., "20layer-0.1-0.5")
- **Run** = Algorithm attempt (e.g., "20240202-sb3_discrete-test1")


## Example: Comparing Algorithms

```bash
# Run PPO on 20-layer problem
uv run python -m coatopt.run --config experiments/config_sb3_discrete.ini

# Run NSGA-II on same problem (update save_dir to use same experiment name)
uv run python -m coatopt.run --config experiments/config_genetic.ini

# Compare in MLflow
uv run mlflow ui
```

### Algorithm-Specific Sections
See example configs in `experiments/` for algorithm-specific parameters.
