# CoatOpt Basic Usage Guide

This guide covers the basic usage of the CoatOpt package for multi-objective coating optimization using reinforcement learning.

## Installation

1. Create a conda environment (Python 3.11 recommended):
```bash
conda create -n coatopt python=3.11
conda activate coatopt
```

2. Install the package:
```bash
cd coatopt
pip install .
```

## Quick Start

### Command Line Interface (CLI)

The primary way to use CoatOpt is through the command line interface:

#### Training a New Model

```bash
coatopt-train -c src/coatopt/config/default.ini --save-plots
```

#### Continue Training from Checkpoint

By default, training continues from existing checkpoints. To explicitly continue:

```bash
coatopt-train -c config.ini
```

#### Start Fresh Training (Ignore Existing Data)

To start training from scratch, ignoring any existing checkpoints:

```bash
coatopt-train -c config.ini --retrain
```

#### Evaluation Only

To run evaluation on a trained model without additional training:

```bash
coatopt-train -c config.ini --evaluate -n 2000
```

#### Quiet Mode

To run with minimal output:

```bash
coatopt-train -c config.ini --quiet
```

### CLI Options

- `-c, --config`: Path to configuration file (required)
- `--retrain`: Start training from scratch (ignore existing checkpoints)
- `--evaluate`: Run evaluation only (no training)
- `-n, --n-samples`: Number of samples for evaluation (default: 1000)
- `--save-plots`: Save training plots and visualizations
- `-q, --quiet`: Minimal output (less verbose)
- `-v, --version`: Show version information

### Graphical User Interface (UI)

For interactive training and visualization:

```bash
coatopt-ui
```

The UI provides:
- Real-time training visualization
- Pareto front exploration
- Training control (start/stop/resume)

## Configuration Files

CoatOpt uses INI-style configuration files. The default configuration is located at `src/coatopt/config/default.ini`.

### Creating Custom Configurations

1. Copy the default configuration:
```bash
cp src/coatopt/config/default.ini my_config.ini
```

2. Edit the configuration file with your preferred settings

3. Run training with your custom configuration:
```bash
coatopt-train -c my_config.ini
```

## Output Structure

Training creates the following output structure in your specified root directory:

```
output_directory/
├── network_weights/          # Model weights
├── evaluation_outputs/       # Evaluation results
├── training_checkpoint.h5    # checkpoint file
├── training_metrics.h5       # Training statistics
└── evaluation_outputs.h5     # evaluation outputs
```

## Key Concepts

### Multi-Objective Optimization

CoatOpt simultaneously optimizes multiple objectives:
- **Reflectivity**: Maximize mirror reflectance
- **Thermal Noise**: Minimize Brownian thermal noise  
- **Absorption**: Minimize optical absorption
- **Total thickness**: Minimize total thickness

### Pareto Front

The algorithm maintains a Pareto front of non-dominated solutions, allowing exploration of trade-offs between competing objectives.

### PC-HPPO Algorithm

CoatOpt uses Parameter Constrained Hybrid Proximal Policy Optimization (PC-HPPO) with:
- Hybrid action space (discrete material + continuous thickness)
- Multi-objective reward functions
- LSTM or attention-based networks for sequential processing

## Common Workflows

### 1. Quick Evaluation

```bash
# Train a model quickly with default settings
coatopt-train -c src/coatopt/config/default.ini --save-plots

# Evaluate the trained model
coatopt-train -c src/coatopt/config/default.ini --evaluate -n 1000 --save-plots
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `batch_size` in config
2. **Slow Training**: Check if using GPU (`device = "cuda:0"`)
3. **Poor Convergence**: Adjust learning rates or increase `n_iterations`
4. **File Not Found**: Use absolute paths in configuration

### Getting Help

- Check configuration options in `available_options.md`
- Review default configuration in `default_config.md`
- Use `--help` flag: `coatopt-train --help`
