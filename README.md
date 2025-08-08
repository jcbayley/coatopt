# CoatOpt: Multi-Objective Coating Optimization

A reinforcement learning framework for optimizing gravitational wave detector mirror coatings using PC-HPPO (Parameter Constrained Hybrid Proximal Policy Optimization).
Taken ideas from https://iopscience.iop.org/article/10.1088/2632-2153/abc327, https://arxiv.org/abs/1903.01344 and 
## Overview

CoatOpt uses multi-objective reinforcement learning to design optimal coating stacks that simultaneously optimize:
- **Reflectivity**: Maximize mirror reflectance
- **Thermal Noise**: Minimize Brownian thermal noise
- **Absorption**: Minimize optical absorption

The algorithm maintains a Pareto front of non-dominated solutions, allowing exploration of trade-offs between competing objectives.

## Installation

Create a conda environment (tested on Python 3.11):

```bash
conda create -n coatopt python=3.11
conda activate coatopt
```

Install the package:

```bash
pip install .
```

## Quick Start

### Training a Model cli

Train the PC-HPPO-OML algorithm for multi-objective Pareto optimization in the command line:

```bash
coatopt-cli -c src/coatopt/config/default.ini --save-plots
```

### Training a Model ui

Train the PC-HPPO-OML algorithm for multi-objective Pareto optimization in the user interface, then see UI instructions:

```bash
coatopt-ui 
```

### Testing/Evaluation

See the UI docs for the ui instructions, for the cli run :

```bash
coatopt-cli -c src/coatopt/config/default.ini --evaluate -n 1000 --save-plots
```

### Continue Training

Resume training is on by default, if you want to retrain or start new training run:

```bash
coatopt-cli -c src/coatopt/config/default.ini -retrain
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

## Outputs

Training generates:
- **HDF5 results**: Complete evaluation results for analysis
- **HDF5 evaluation**: evaluation results file
- **Plots**: plots showing training and pareto front metrics

## Algorithm Details

PC-HPPO-OML uses:
- **Hierarchical action space**: Discrete material selection + continuous thickness
- **Multi-objective rewards**: Dynamic weight cycling and randomisation to explore Pareto front
- **Pareto front maintenance**: Non-dominated sorting to find pareto front
- **LSTM or attention pre-networks**: Sequential processing of coating layer information
- **PPO updates**: Stable policy gradient optimization with clipping

## Requirements

Core dependencies:
- `torch`: Neural networks
- `numpy`, `scipy`: Numerical computation  
- `tmm`, `tmm_fast`: Transfer Matrix Method for optics
- `matplotlib`: Visualization
- `pandas`: Data handling
- `pymoo`: Multi-objective optimization utilities
