"""
Structured configuration management for coating optimization.
Provides type-safe configuration objects to replace repetitive config.get() calls.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from .load_config import CoatingConfigParser


@dataclass
class GeneralConfig:
    """General configuration parameters."""
    root_dir: str
    data_dir: str
    load_model: bool
    load_model_path: str
    materials_file: str
    continue_training: bool


@dataclass
class DataConfig:
    """Data and environment configuration parameters."""
    n_layers: int
    min_thickness: float
    max_thickness: float
    use_observation: bool
    reward_shape: str
    thermal_reward_shape: str
    reflectivity_reward_shape: str
    absorption_reward_shape: str
    use_intermediate_reward: bool
    ignore_air_option: bool
    ignore_substrate_option: bool
    optimise_parameters: List[str]
    optimise_targets: Dict[str, float]
    optimise_weight_ranges: Dict[str, List[float]]
    use_ligo_reward: bool
    include_random_rare_state: bool
    use_optical_thickness: bool
    combine: str
    reward_func: str


@dataclass
class NetworkConfig:
    """Neural network architecture configuration."""
    model_type: str
    hyper_networks: bool
    include_layer_number: bool
    include_material_in_policy: bool
    pre_network_type: str
    hidden_size: int
    n_pre_layers: int
    n_continuous_layers: int
    n_discrete_layers: int
    n_value_layers: int
    discrete_hidden_size: int
    continuous_hidden_size: int
    value_hidden_size: int


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""
    n_iterations: int
    lr_discrete_policy: float
    lr_continuous_policy: float
    lr_value: float
    n_episodes_per_update: int
    n_epochs_per_update: int
    clip_ratio: float
    gamma: float
    batch_size: int
    optimiser: str
    device: str
    model_save_interval: int
    
    # Entropy scheduling
    entropy_beta_start: float
    entropy_beta_end: float
    entropy_beta_decay_length: Optional[int]
    entropy_beta_decay_start: int
    
    # Learning rate scheduling
    scheduler_start: int
    scheduler_end: int
    lr_step: int
    lr_min: float
    
    # Pareto optimization
    n_init_solutions: int
    final_weight_epoch: int
    start_weight_alpha: float
    final_weight_alpha: float
    cycle_weights: bool
    n_weight_cycles: int
    weight_network_save: bool


@dataclass
class CoatingOptimizationConfig:
    """Complete configuration for coating optimization."""
    general: GeneralConfig
    data: DataConfig
    network: NetworkConfig
    training: TrainingConfig
    
    @classmethod
    def from_config_parser(cls, config: CoatingConfigParser) -> 'CoatingOptimizationConfig':
        """Create structured config from ConfigParser."""
        
        general = GeneralConfig(
            root_dir=config.get("General", "root_dir"),
            data_dir=config.get("General", "data_dir"),
            load_model=config.get("General", "load_model"),
            load_model_path=config.get("General", "load_model_path"),
            materials_file=config.get("General", "materials_file"),
            continue_training=config.get("General", "continue_training")
        )
        
        data = DataConfig(
            n_layers=config.get("Data", "n_layers"),
            min_thickness=config.get("Data", "min_thickness"),
            max_thickness=config.get("Data", "max_thickness"),
            use_observation=config.get("Data", "use_observation"),
            reward_shape=config.get("Data", "reward_shape"),
            thermal_reward_shape=config.get("Data", "thermal_reward_shape"),
            reflectivity_reward_shape=config.get("Data", "reflectivity_reward_shape", fallback="none"),
            absorption_reward_shape=config.get("Data", "absorption_reward_shape", fallback="none"),
            use_intermediate_reward=config.get("Data", "use_intermediate_reward"),
            ignore_air_option=config.get("Data", "ignore_air_option"),
            ignore_substrate_option=config.get("Data", "ignore_substrate_option"),
            optimise_parameters=config.get("Data", "optimise_parameters"),
            optimise_targets=config.get("Data", "optimise_targets"),
            optimise_weight_ranges=config.get("Data", "optimise_weight_ranges", fallback={}),
            use_ligo_reward=config.get("Data", "use_ligo_reward"),
            include_random_rare_state=config.get("Data", "include_random_rare_state"),
            use_optical_thickness=config.get("Data", "use_optical_thickness", fallback=True),
            combine=config.get("Data", "combine", fallback="logproduct"),
            reward_func=config.get("Data", "reward_func")
        )
        
        network = NetworkConfig(
            model_type=config.get("Network", "model_type"),
            hyper_networks=config.get("Network", "hyper_networks"),
            include_layer_number=config.get("Network", "include_layer_number"),
            include_material_in_policy=config.get("Network", "include_material_in_policy"),
            pre_network_type=config.get("Network", "pre_network_type"),
            hidden_size=config.get("Network", "hidden_size"),
            n_pre_layers=config.get("Network", "n_pre_layers"),
            n_continuous_layers=config.get("Network", "n_continuous_layers", fallback=2),
            n_discrete_layers=config.get("Network", "n_discrete_layers", fallback=2),
            n_value_layers=config.get("Network", "n_value_layers", fallback=2),
            discrete_hidden_size=config.get("Network", "discrete_hidden_size", fallback=16),
            continuous_hidden_size=config.get("Network", "continuous_hidden_size", fallback=16),
            value_hidden_size=config.get("Network", "value_hidden_size", fallback=16)
        )
        
        training = TrainingConfig(
            n_iterations=config.get("Training", "n_iterations"),
            lr_discrete_policy=config.get("Training", "lr_discrete_policy"),
            lr_continuous_policy=config.get("Training", "lr_continuous_policy"),
            lr_value=config.get("Training", "lr_value"),
            n_episodes_per_update=config.get("Training", "n_episodes_per_update"),
            n_epochs_per_update=config.get("Training", "n_epochs_per_update"),
            clip_ratio=config.get("Training", "clip_ratio"),
            gamma=config.get("Training", "gamma"),
            batch_size=config.get("Training", "batch_size"),
            optimiser=config.get("Training", "optimiser"),
            device=config.get("Training", "device"),
            model_save_interval=config.get("Training", "model_save_interval"),
            entropy_beta_start=config.get("Training", "entropy_beta_start"),
            entropy_beta_end=config.get("Training", "entropy_beta_end"),
            entropy_beta_decay_length=config.get("Training", "entropy_beta_decay_length"),
            entropy_beta_decay_start=config.get("Training", "entropy_beta_decay_start"),
            scheduler_start=config.get("Training", "scheduler_start"),
            scheduler_end=config.get("Training", "scheduler_end"),
            lr_step=config.get("Training", "lr_step"),
            lr_min=config.get("Training", "lr_min"),
            n_init_solutions=config.get("Training", "n_init_solutions"),
            final_weight_epoch=config.get("Training", "final_weight_epoch"),
            start_weight_alpha=config.get("Training", "start_weight_alpha"),
            final_weight_alpha=config.get("Training", "final_weight_alpha"),
            cycle_weights=config.get("Training", "cycle_weights", fallback=False),
            n_weight_cycles=config.get("Training", "n_weight_cycles"),
            weight_network_save=config.get("Training", "weight_network_save")
        )
        
        return cls(
            general=general,
            data=data,
            network=network,
            training=training
        )