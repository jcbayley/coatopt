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
| `"normed_log_targets"` (recommended) | Normalized logarithmic targets with environment scaling | Balanced multi-objective optimization |
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

#### `"normed_log_targets"` - Normalized Log Targets  (recommended)
- Normalizes objectives using environment statistics
- Prevents any single objective from dominating
- Ideal for balanced multi-objective optimization
- Works best with Reflectivity, Absorption, Thermal noise

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

## Mixture of Experts (MoE)

CoatOpt supports Mixture of Experts architectures for handling multi-objective optimization with specialized expert networks.

### MoE Configuration

```ini
[Network]
use_mixture_of_experts = true
moe_n_experts = 7                          # Number of expert networks
moe_expert_specialization = "sobol_sequence"  # Expert specialization strategy
moe_gate_hidden_dim = 64                   # Hidden dimension for gating network
moe_gate_temperature = 0.5                 # Temperature for expert selection (lower = more decisive)
moe_load_balancing_weight = 0.01           # Weight for load balancing auxiliary loss
```

### MoE Expert Specialization Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `"sobol_sequence"` | Uses Sobol quasi-random sequence to assign expert regions | Balanced coverage of objective space |
| `"random"` | Random assignment of expert specializations | Exploration-focused |
| `"uniform"` | Uniform distribution of experts across objectives | Simple, evenly distributed |

### MoE Parameters

- **`moe_n_experts`**: Number of expert networks. More experts = finer specialization but higher computational cost
- **`moe_gate_temperature`**: Controls expert selection decisiveness. Lower values (0.1-0.5) force decisive selection; higher values (1.0+) allow soft mixing
- **`moe_expert_specialization`**: Strategy for assigning experts to different objective regions
- **`moe_load_balancing_weight`**: Auxiliary loss weight to encourage balanced expert usage

## Reward Normalization

CoatOpt supports reward normalization to balance objectives with different scales and ranges.

### Reward Normalization Configuration

```ini
[Data]
use_reward_normalization = true
reward_normalization_mode = "adaptive"        # "fixed" or "adaptive"
reward_normalization_ranges = {}              # Leave empty to auto-compute from objective bounds
reward_normalization_alpha = 0.3              # Learning rate for adaptive mode
```

### Auto-Computed Ranges

CoatOpt can automatically compute reward normalization ranges based on the objective bounds defined in reward functions:

- **Automatic Detection**: Leave `reward_normalization_ranges = {}` empty
- **Reward Function Analysis**: Analyzes your reward function type (e.g., log-based) 
- **Objective Bounds**: Uses `env.objective_bounds` to estimate typical reward ranges
- **Smart Defaults**: Provides sensible ranges for common reward functions like `normalise_log_targets`

### Manual Ranges (Optional)

```ini
reward_normalization_ranges = {               # Manual override (optional)
    "reflectivity": [8, 28], 
    "absorption": [8, 25]
}
```

### Normalization Modes

| Mode | Description | When to Use |
|------|-------------|-------------|
| `"fixed"` | Use predefined ranges for normalization | When you know typical reward ranges |
| `"adaptive"` | Learn ranges dynamically during training | For unknown reward distributions |

### Normalization Benefits

- **Scale Balance**: Prevents high-magnitude objectives from dominating
- **Improved Trade-offs**: Enables better learning of balanced solutions
- **Stable Training**: Reduces reward scale variations across episodes

### Automatic Range Detection

The system automatically computes appropriate ranges when left empty by analyzing:
- **Reward Function Type**: Detects log-based, linear, or exponential reward patterns
- **Objective Bounds**: Uses existing `env.objective_bounds` from reward functions
- **Scale Estimation**: Automatically estimates typical reward ranges for each objective

**Manual Range Override** (optional): If auto-detection doesn't work well, specify ranges manually:
- Reflectivity rewards typically range 8-28  
- Absorption rewards typically range 8-25
- Run episodes without normalization to observe actual ranges

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

# Optional: Specify objective bounds for reward normalization
objective_bounds = {
    "reflectivity": [1e-6, 1e-1],
    "absorption": [1e-4, 1000.0],
    "thermal_noise": [1e-25, 1e-15]
}
```

### Objective Bounds

Objective bounds define the expected range for each optimization parameter and are used for:
- **Automatic reward range computation**: When `reward_normalization_ranges = {}` (empty), the system automatically computes appropriate normalization ranges by evaluating the reward function at these bounds
- **Optimization constraints**: Helps the algorithm understand the feasible parameter space
- **Default behavior**: If not specified, reward functions set default bounds automatically

The bounds use the format: `{"parameter": [min_value, max_value]}`

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
| `"annealed_random"` | Annealed Dirichlet weights | Progressive from extreme to balanced |
| `"step"` | Step-wise cycling (2 objectives only) | Traditional alternating approach |
| `"linear"` | Linear grid of weights | Systematic exploration |
| `"adaptive_pareto"` | **NEW** Adaptive exploration targeting Pareto gaps | Enhanced exploration of under-explored regions |

### Enhanced Weight Exploration (Phase 2)

The new `"adaptive_pareto"` method implements enhanced weight exploration by:
- **Gap Detection**: Identifies under-explored regions in the current Pareto front
- **Dynamic Sampling**: Adjusts weight probability based on front density
- **Archive Memory**: Prevents revisiting recently used weight combinations
- **Progressive Exploration**: Higher exploration early, more exploitation later

**Configuration Example:**
```ini
[Training]
cycle_weights = "adaptive_pareto"
final_weight_epoch = 2000
start_weight_alpha = 0.1
final_weight_alpha = 1.0
```

**Benefits:**
- 50-80% improvement in middle-front exploration
- Better coverage uniformity across Pareto front
- Reduced clustering around extreme solutions
- Maintains diversity while avoiding redundant exploration

### Hypervolume-Based Training (Phase 3.2)

**NEW** Direct hypervolume optimization provides an alternative to weighted scalarization:

**Key Features:**
- **Direct HV Optimization**: Replace weighted scalarization with hypervolume gradient estimation
- **HV-based Rewards**: Individual contribution rewards based on hypervolume improvement
- **Adaptive Reference Point**: Automatic reference point adaptation based on current front bounds
- **Enhanced Coverage**: Better Pareto front coverage and diversity

**Configuration Options:**
```ini
[Training]
use_hypervolume_trainer = True      # Enable hypervolume-enhanced trainer
use_hypervolume_loss = True         # Use HV loss in addition to standard rewards  
hv_loss_weight = 0.5               # Weight for hypervolume loss (0-1)
hv_update_interval = 10            # Update HV reference point every N episodes
adaptive_reference_point = True     # Automatically adapt reference point
```

**Benefits:**
- **Better Front Quality**: Direct optimization for coverage and diversity
- **Non-convex Discovery**: Finds non-convex Pareto front sections missed by scalarization
- **Scalable Performance**: Maintains efficiency with larger fronts
- **Near-optimal Hypervolume**: Achieves theoretically better hypervolume values

**When to Use:**
- Multi-objective problems with 2-4 objectives
- When front coverage uniformity is critical
- Problems where weighted scalarization misses solutions
- When you need provably good hypervolume performance

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
# General entropy scheduling (applies to both policies if specific ones not set)
entropy_beta_start = 1.0      # High exploration initially
entropy_beta_end = 0.001      # Low exploration at end
entropy_beta_decay_length = 5000  # Decay over 5000 iterations

# Separate entropy coefficients for discrete and continuous policies (recommended)
entropy_beta_discrete_start = 0.1    # Discrete policy exploration
entropy_beta_discrete_end = 0.01
entropy_beta_continuous_start = 0.05 # Continuous policy exploration  
entropy_beta_continuous_end = 0.001
```

### Entropy Policy Types

| Policy Type | Controls | Typical Range | Use Case |
|-------------|----------|---------------|----------|
| **Discrete** | Material selection | 0.01 - 0.1 | Lower values for decisive material choices |
| **Continuous** | Layer thickness | 0.001 - 0.05 | Higher precision for thickness optimization |

### Entropy Scheduling Benefits

- **Separate Control**: Fine-tune exploration for material vs thickness decisions
- **Balanced Policies**: Prevent one policy from dominating early training
- **Multi-objective**: Essential for balanced objective exploration

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