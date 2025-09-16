# CoatOpt: Multi-Objective Coating Optimization

A reinforcement learning framework for optimizing gravitational wave detector mirror coatings using PC-HPPO (Parameter Constrained Hybrid Proximal Policy Optimization).

## Overview

CoatOpt uses multi-objective reinforcement learning to design optimal coating stacks that simultaneously optimize:
- **Reflectivity**: Maximize mirror reflectance
- **Thermal Noise**: Minimize Brownian thermal noise
- **Absorption**: Minimize optical absorption

The algorithm maintains a Pareto front of non-dominated solutions, allowing exploration of trade-offs between competing objectives.

## Installation

CoatOpt uses **uv** as the package manager. Python 3.9+ is required.

### Option 1: Using uv (Recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up the project
cd coatopt
uv sync
```

### Option 2: Traditional Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -e .
```

## Quick Start

### Training a Model (CLI)

```bash
# Using uv (recommended)
uv run coatopt-train -c src/coatopt/config/default.ini --save-plots

# Or with activated environment
source .venv/bin/activate
coatopt-train -c src/coatopt/config/default.ini --save-plots
```

### Interactive GUI

```bash
# Launch GUI interface
uv run coatopt-ui
```

### Evaluation Only

```bash
# Run evaluation on trained model
uv run coatopt-train -c src/coatopt/config/default.ini --evaluate -n 1000 --save-plots
```

### Start Fresh Training

```bash
# Retrain from scratch (ignore existing checkpoints)
uv run coatopt-train -c src/coatopt/config/default.ini --retrain
```

## Configuration

Key parameters in `default.ini`:

**Data Section:**
- `n_layers`: Number of coating layers (default: 8)
- `optimise_parameters`: Objectives to optimize (reflectivity, thermal_noise, absorption)
- `min_thickness`/`max_thickness`: Layer thickness bounds

**Network Section:**
- `pre_network_type`: Feature extraction network (lstm, linear, attn)
- `hidden_size`: Network hidden dimensions
- `hyper_networks`: Enable hypernetwork architecture

**Training Section:**
- `n_iterations`: Training iterations (default: 8000)
- `lr_*_policy`: Learning rates for discrete/continuous policies
- `clip_ratio`: PPO clipping parameter

## Documentation

For detailed usage instructions and examples:
- **[Basic Usage Guide](docs/basic_usage.md)** - Installation and getting started
- **[Configuration Reference](docs/default_config.md)** - All configuration parameters
- **[Available Options](docs/available_options.md)** - Advanced features and options

## Outputs

Training generates:
- **Model checkpoints**: Neural network weights
- **Training metrics**: HDF5 files with training statistics
- **Evaluation results**: Complete analysis data
- **Plots**: Training progress and Pareto front visualizations

## Algorithm Details

PC-HPPO-OML uses:
- **Hierarchical action space**: Discrete material selection + continuous thickness
- **Multi-objective rewards**: Dynamic weight cycling and randomisation to explore Pareto front
- **Pareto front maintenance**: Non-dominated sorting to find pareto front
- **LSTM or attention pre-networks**: Sequential processing of coating layer information
- **PPO updates**: Stable policy gradient optimization with clipping

## Development

### Code Quality

This project includes pre-commit hooks for code formatting and linting:

```bash
# Install development dependencies
uv sync --extra dev

# Set up pre-commit hooks
uv run pre-commit install
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src --cov-report=html
```

## Requirements

Key dependencies automatically installed:
- **torch>=2.0.0**: Neural network training
- **numpy, scipy**: Numerical computation
- **tmm, tmm_fast**: Transfer Matrix Method for optics
- **matplotlib**: Visualization
- **pandas**: Data handling
- **pymoo**: Multi-objective optimization utilities
- **mlflow**: Experiment tracking
