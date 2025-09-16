# CoatOpt Documentation

This directory contains comprehensive documentation for the CoatOpt multi-objective coating optimization package.

## Documentation Overview

### 📚 Documentation Files

| File | Description |
|------|-------------|
| [`basic_usage.md`](basic_usage.md) | **Start here!** Basic usage guide, CLI commands, and common workflows |
| [`default_config.md`](default_config.md) | Complete reference for all configuration parameters |
| [`available_options.md`](available_options.md) | All available reward functions, network types, and advanced options |
| [`default.ini`](default.ini) | Default configuration file with all parameters |

### 🚀 Quick Start

1. **New Users**: Start with [`basic_usage.md`](basic_usage.md) for installation and first steps
2. **Configuration**: Check [`default_config.md`](default_config.md) for parameter details
3. **Advanced Users**: See [`available_options.md`](available_options.md) for all available options

## Package Overview

CoatOpt is a reinforcement learning framework for optimizing gravitational wave detector mirror coatings using PC-HPPO (Parameter Constrained Hybrid Proximal Policy Optimization).

### Key Features

- **Multi-Objective Optimization**: Simultaneously optimize reflectivity, thermal noise, and absorption
- **Pareto Front Exploration**: Maintain and explore trade-offs between competing objectives
- **Advanced Neural Networks**: LSTM, attention, and hypernetwork architectures
- **Flexible Configuration**: Extensive customization through INI files
- **CLI and GUI**: Command-line and graphical user interfaces

### Algorithms

- **PC-HPPO**: Parameter Constrained Hybrid Proximal Policy Optimization
- **Hierarchical Actions**: Discrete material selection + continuous thickness control
- **Multi-Objective Rewards**: Dynamic weight cycling and various reward functions
- **Pareto Front Maintenance**: Non-dominated sorting for optimal solutions

## Common Use Cases

### 1. Basic Training
```bash
coatopt-train -c src/coatopt/config/default.ini --save-plots
```

### 2. Custom Configuration
```bash
# Copy and modify default config
cp src/coatopt/config/default.ini my_experiment.ini
# Edit my_experiment.ini as needed
coatopt-train -c my_experiment.ini --save-plots
```

### 3. Evaluation Only
```bash
coatopt-train -c config.ini --evaluate -n 2000 --save-plots
```

### 4. Interactive GUI
```bash
coatopt-ui
```

## Configuration Hierarchy

```
CoatOpt Configuration
├── [General] - Basic setup (directories, models)
├── [Data] - Environment and optimization settings
│   ├── Layer configuration (n_layers, thickness bounds)
│   ├── Optimization objectives (reflectivity, thermal_noise, absorption)
│   └── Reward functions (default, log_targets, hypervolume, etc.)
├── [Network] - Neural network architecture
│   ├── Pre-networks (lstm, linear, attn)
│   └── Policy networks (discrete, continuous, value)
├── [Training] - Algorithm parameters
│   ├── Learning rates and schedules
│   ├── PPO parameters (clip_ratio, gamma)
│   └── Multi-objective settings (weight cycling)
├── [MCMC] - MCMC comparison method
└── [Genetic] - Genetic algorithm comparison
```

## Available Options Summary

### Reward Functions
- `default`: Standard sigmoid-based rewards
- `log_targets`: Logarithmic minimization
- `normed_log_targets`: Normalized log rewards
- `hypervolume`: Hypervolume-based Pareto rewards
- `target`, `raw`, `area`: Specialized reward types

### Network Types
- `lstm`: LSTM sequential processing
- `linear`: Simple linear networks
- `attn`: Transformer attention networks

### Optimization Objectives
- `reflectivity`: Mirror reflectance
- `thermal_noise`: Brownian thermal noise
- `absorption`: Optical absorption
- `thickness`: Total coating thickness

## Output Structure

Training produces organized outputs:

```
output_directory/
├── states/                 # Training checkpoints
├── network_weights/        # Model weights
├── plots/                  # Training visualizations
├── evaluation/            # Evaluation results
├── pareto_front.h5        # Final Pareto front
├── all_points.h5          # All explored points
├── training_metrics.h5    # Training statistics
└── best_states.h5         # Best solutions
```

## Getting Help

### Documentation
- Read the appropriate `.md` files for your needs
- Check configuration examples in each file
- Review the default configuration file

### Command Line Help
```bash
coatopt-train --help
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `batch_size` in config |
| Slow training | Check GPU usage, increase `batch_size` |
| Poor convergence | Adjust learning rates, increase iterations |
| File not found | Use absolute paths in configuration |

### Performance Optimization

| Goal | Configuration Changes |
|------|---------------------|
| **Speed** | GPU device, larger batch size, linear networks |
| **Quality** | Attention networks, larger networks, more iterations |
| **Memory** | Smaller batch size, smaller networks, CPU device |

## Contributing

When contributing to documentation:

1. Keep examples practical and tested
2. Update all relevant files when adding features
3. Maintain consistent formatting and style
4. Include performance and usage guidance

## Version Information

This documentation is for CoatOpt v1.0.0. For the latest updates, check the main repository.
