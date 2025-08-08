# Available Options Reference

This document provides a comprehensive reference for all available options and parameters in CoatOpt.

## Reward Functions

CoatOpt supports multiple reward function types that can be specified using the `reward_function` parameter in the `[Data]` section.

### Available Reward Functions

| Function Name | Description | Use Case |
|---------------|-------------|----------|
| `"default"` | Standard sigmoid-based reward with log transformations | General optimization, balanced objectives |
| `"target"` | Target-based reward function | When specific target values are known |
| `"raw"` | Raw objective values without transformation | Direct optimization of metrics |
| `"log_targets"` | Logarithmic minimization with target scaling | Optimization with wide dynamic ranges |
| `"hypervolume"` | Hypervolume-based reward for Pareto optimization | Multi-objective optimization focus |
| `"normed_log_targets"` | Normalized logarithmic targets with environment scaling | Balanced multi-objective optimization |
| `"normed_log"` | Normalized logarithmic rewards | Standardized objective scaling |
| `"area"` | Area-based Pareto front reward | Pareto front expansion |

### Reward Function Details

#### `"default"` - Standard Reward Function
- Uses sigmoid functions to shape rewards around targets
- Applies log transformations to handle wide value ranges
- Good for general-purpose optimization
- Balanced treatment of all objectives

#### `"log_targets"` - Log Minimization
- Focuses on minimizing objectives with logarithmic scaling
- Suitable for objectives with exponential behavior
- Good for thermal noise and absorption optimization

#### `"normed_log_targets"` - Normalized Log Targets  
- Normalizes objectives using environment statistics
- Prevents any single objective from dominating
- Ideal for balanced multi-objective optimization

#### `"hypervolume"` - Hypervolume Reward
- Rewards solutions that increase Pareto front hypervolume  
- Encourages diverse Pareto-optimal solutions
- Best for pure multi-objective optimization

## Network Architectures

CoatOpt supports different neural network architectures specified by the `pre_network_type` parameter.

### Pre-Network Types

| Type | Description | Strengths | Best For |
|------|-------------|-----------|----------|
| `"lstm"` | LSTM-based sequential processing | Handles variable-length sequences, memory | Sequential coating design |
| `"linear"` | Simple linear transformation | Fast, lightweight | Simple problems, debugging |
| `"attn"` | Transformer attention mechanism | Parallel processing, long-range dependencies | Complex coating interactions |

### Pre-Network Configuration

#### LSTM Networks (`"lstm"`)
```ini
[Network]
pre_network_type = "lstm"
hidden_size = 32        # LSTM hidden state size
n_pre_layers = 2        # Number of LSTM layers
```

#### Linear Networks (`"linear"`)
```ini
[Network]
pre_network_type = "linear"
hidden_size = 32        # Linear layer size
n_pre_layers = 2        # Number of linear layers
```

#### Attention Networks (`"attn"`)
```ini
[Network]
pre_network_type = "attn"
hidden_size = 32        # Embedding dimension
n_pre_layers = 2        # Number of transformer layers
```

## Optimization Parameters

### Available Optimization Objectives

CoatOpt can optimize multiple objectives simultaneously:

| Objective | Parameter Name | Description | Units |
|-----------|----------------|-------------|-------|
| Reflectivity | `"reflectivity"` | Mirror reflectance | Fraction (0-1) |
| Thermal Noise | `"thermal_noise"` | Brownian thermal noise | m/√Hz |
| Absorption | `"absorption"` | Optical absorption | Fraction |
| Thickness | `"thickness"` | Total coating thickness | Meters |

### Configuration Example

```ini
[Data]
optimise_parameters = ["reflectivity", "thermal_noise", "absorption"]
optimise_targets = {
    "reflectivity": 0.999999,
    "thermal_noise": 1e-21,
    "absorption": 0.001
}
```

## Model Types

### Available Model Types

Currently, CoatOpt supports:

| Model Type | Description | Algorithm |
|------------|-------------|-----------|
| `"hppo"` | Hybrid Proximal Policy Optimization | PC-HPPO with hierarchical actions |

Future versions may include additional algorithms.

## Device Options

### Supported Devices

| Device | Description | Requirements |
|--------|-------------|--------------|
| `"cuda:0"` | First GPU | CUDA-compatible GPU + PyTorch with CUDA |
| `"cuda:1"` | Second GPU | Multiple GPUs available |
| `"cpu"` | CPU only | Works on any system (slower) |

## Optimizer Options

### Available Optimizers

| Optimizer | Description | Characteristics |
|-----------|-------------|----------------|
| `"adam"` | Adam optimizer | Adaptive learning rates, good default |
| `"sgd"` | Stochastic Gradient Descent | Simple, requires careful learning rate tuning |
| `"rmsprop"` | RMSprop optimizer | Good for recurrent networks |

## Weight Cycling Methods

For multi-objective optimization, CoatOpt supports different weight cycling strategies:

| Method | Description | Use Case |
|--------|-------------|----------|
| `"smooth"` | Smooth transitions between objectives | Balanced exploration |
| `"random"` | Random objective weights | Maximum diversity |
| `"fixed"` | Fixed weight combinations | Specific trade-off preferences |

## Materials Configuration

CoatOpt can use different materials databases:

| Materials File | Description |
|----------------|-------------|
| `"default"` | Default materials (Ti, Ta, SiO2, etc.) |
| `"custom"` | User-defined materials file |
| `"ligo"` | LIGO-specific materials |

### Custom Materials File Format

Materials are defined in JSON format:
```json
{
    "material_name": {
        "n": 2.3,        // Refractive index
        "k": 0.0001,     // Extinction coefficient  
        "Y": 1.4e11,     // Young's modulus
        "sigma": 0.23,   // Poisson ratio
        "phi": 2.3e-4    // Loss angle
    }
}
```

## Advanced Options

### Hypernetwork Architecture

Enable hypernetworks for parameter-efficient training:

```ini
[Network]
hyper_networks = True
include_material_in_policy = True
```

### Entropy Scheduling

Control exploration vs exploitation over training:

```ini
[Training]
entropy_beta_start = 1.0      # High exploration initially
entropy_beta_end = 0.001      # Low exploration at end
entropy_beta_decay_length = 5000  # Decay over 5000 iterations
```

### Learning Rate Scheduling

Implement learning rate decay:

```ini
[Training]
scheduler_start = 1000        # Start decay at iteration 1000
lr_step = 2000               # Reduce LR every 2000 steps
lr_min = 1e-6                # Minimum learning rate
```

## Configuration Validation

CoatOpt validates configurations at startup and will report errors for:

- Invalid reward function names
- Incompatible network configurations  
- Missing required parameters
- Invalid file paths
- Unsupported device configurations

## Performance Tuning

### For Speed
- Use GPU: `device = "cuda:0"`
- Increase batch size: `batch_size = 512`
- Use linear networks: `pre_network_type = "linear"`
- Reduce network sizes

### For Quality  
- Use attention networks: `pre_network_type = "attn"`
- Increase network sizes
- More training iterations: `n_iterations = 15000`
- Smaller learning rates

### For Memory Efficiency
- Reduce batch size: `batch_size = 128`
- Smaller networks: `hidden_size = 16`
- Use CPU: `device = "cpu"`

## Experimental Features

Some features are experimental and may not be fully stable:

- Hypernetwork architectures
- Advanced entropy scheduling
- Custom reward function combinations
- Multi-GPU training

Use these features with caution and validate results carefully.

## Related Documentation

- See `basic_usage.md` for usage examples
- See `default_config.md` for detailed parameter descriptions
- Check the main README for algorithm background