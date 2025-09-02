# Default Configuration Reference

This document describes all configuration parameters available in the CoatOpt default configuration file (`default.ini`).

## Configuration Sections

### [General]

General settings for the optimization pipeline.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `root_dir` | `"./"` | Root directory for output files |
| `data_dir` | `"./data"` | Directory for data files |
| `load_model` | `False` | Whether to load a pre-trained model |
| `load_model_path` | `"root"` | Path to pre-trained model |
| `materials_file` | `"default"` | Materials database file to use |
| `continue_training` | `False` | Continue training from checkpoint |

### [Data]

Data and environment configuration parameters.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_layers` | `8` | Number of coating layers to optimize |
| `min_thickness` | `1e-10` | Minimum layer thickness (meters) |
| `max_thickness` | `300e-9` | Maximum layer thickness (meters) |
| `use_observation` | `True` | Use environment observations |
| `reward_shape` | `"none"` | Reward shaping method |
| `thermal_reward_shape` | `"none"` | Thermal noise reward shaping |
| `use_intermediate_reward` | `False` | Use rewards during episode |
| `ignore_air_option` | `False` | Ignore air layers in optimization |
| `ignore_substrate_option` | `False` | Ignore substrate in optimization |

#### Optimization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optimise_parameters` | `["reflectivity", "thermal_noise", "absorption"]` | Objectives to optimize |
| `optimise_targets` | `{"reflectivity":1.0, "thermal_noise":1e-21, "absorption":0.01}` | Target values for optimization |
| `design_criteria` | `{"reflectivity":0.99999, "thermal_noise":5.394e-21, "absorption":0.01}` | Design criteria thresholds |

#### Reward Function Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_ligo_reward` | `False` | Use LIGO-specific reward function |
| `include_random_rare_state` | `False` | Include rare state exploration |
| `reward_function` | `"default"` | Reward function type (see Available Options) |

### [Network]

Neural network architecture configuration.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | `"hppo"` | Model type (currently only "hppo" supported) |
| `hyper_networks` | `False` | Use hypernetwork architecture |
| `include_layer_number` | `True` | Include layer number in state |
| `include_material_in_policy` | `False` | Include material info in policy |

#### Pre-Network Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pre_network_type` | `"lstm"` | Pre-network type ("lstm", "linear", "attn") |
| `hidden_size` | `32` | Hidden layer size for pre-network |
| `n_pre_layers` | `2` | Number of pre-network layers |

#### Policy Network Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_value_layers` | `2` | Number of value network layers |
| `n_continuous_layers` | `2` | Number of continuous policy layers |
| `n_discrete_layers` | `2` | Number of discrete policy layers |
| `continuous_hidden_size` | `16` | Hidden size for continuous policy |
| `discrete_hidden_size` | `16` | Hidden size for discrete policy |
| `value_hidden_size` | `16` | Hidden size for value network |

### [Training]

Training algorithm parameters.

#### Basic Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_iterations` | `8000` | Total training iterations |
| `n_episodes_per_update` | `256` | Episodes collected per update |
| `n_epochs_per_update` | `5` | Epochs per policy update |
| `batch_size` | `256` | Batch size for training |
| `device` | `"cuda:0"` | Device for training ("cuda:0", "cpu") |
| `model_save_interval` | `10` | Save model every N iterations |

#### Learning Rates

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr_discrete_policy` | `0.0001` | Learning rate for discrete policy |
| `lr_continuous_policy` | `0.0001` | Learning rate for continuous policy |
| `lr_value` | `0.001` | Learning rate for value network |

#### PPO Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip_ratio` | `0.01` | PPO clipping ratio |
| `gamma` | `0.999` | Discount factor |
| `optimiser` | `"adam"` | Optimizer type |

#### Entropy Regularization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `entropy_beta_start` | `1.0` | Initial entropy coefficient |
| `entropy_beta_end` | `0.001` | Final entropy coefficient |
| `entropy_beta_decay_length` | `None` | Decay length for entropy |
| `entropy_beta_decay_start` | `0` | When to start entropy decay |
| `entropy_beta_use_restarts` | `False` | Whether to restart entropy decay like LR scheduler |

#### Learning Rate Scheduling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scheduler_start` | `0` | When to start LR scheduling |
| `scheduler_end` | `-1` | When to end LR scheduling (-1 = never) |
| `lr_step` | `5000` | Steps between LR reductions |
| `lr_min` | `1e-5` | Minimum learning rate |
| `t_mult` | `2` | LR schedule multiplier |

#### Multi-Objective Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_init_solutions` | `1000` | Initial Pareto front solutions |
| `final_weight_epoch` | `1` | Final weight cycling epoch |
| `start_weight_alpha` | `1.0` | Starting weight alpha |
| `final_weight_alpha` | `1.0` | Final weight alpha |
| `cycle_weights` | `"smooth"` | Weight cycling method |
| `n_weight_cycles` | `1` | Number of weight cycles |
| `weight_network_save` | `False` | Save weight networks |

### [MCMC]

MCMC algorithm parameters (for comparison methods).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_walkers` | `32` | Number of MCMC walkers |
| `n_steps` | `1000` | Number of MCMC steps |

### [Genetic]

Genetic algorithm parameters (for comparison methods).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_states` | `100` | Population size |
| `states_fraction_keep` | `0.1` | Fraction of population to keep |
| `thickness_sigma` | `1e-9` | Mutation standard deviation |
| `num_iterations` | `1000` | Number of generations |

## Configuration Examples

### High-Performance Training

For faster training with more resources:

```ini
[Training]
n_iterations = 10000
batch_size = 512
n_episodes_per_update = 512
lr_discrete_policy = 0.001
lr_continuous_policy = 0.001
device = "cuda:0"
```

### Conservative Training

For stable but slower training:

```ini
[Training]
n_iterations = 15000
batch_size = 128
n_episodes_per_update = 128
lr_discrete_policy = 0.00005
lr_continuous_policy = 0.00005
clip_ratio = 0.005
```

### Large Network

For complex problems requiring larger networks:

```ini
[Network]
hidden_size = 64
n_pre_layers = 3
continuous_hidden_size = 32
discrete_hidden_size = 32
value_hidden_size = 32
```

### Few-Layer Optimization

For optimizing fewer layers:

```ini
[Data]
n_layers = 4
max_thickness = 200e-9
```

## Tips for Configuration

1. **GPU Memory**: Reduce `batch_size` if running out of GPU memory
2. **Training Speed**: Increase `n_episodes_per_update` and `batch_size` for faster training
3. **Stability**: Decrease learning rates if training is unstable
4. **Exploration**: Increase `entropy_beta_start` for more exploration
5. **Fine-tuning**: Use smaller learning rates and longer training for fine-tuning

## Related Documentation

- See `available_options.md` for all available reward functions and network types
- See `basic_usage.md` for usage examples
- Check the main README for algorithm details