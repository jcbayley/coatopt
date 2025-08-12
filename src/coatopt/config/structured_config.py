"""
Structured configuration management for coating optimisation.
Provides type-safe configuration objects to replace repetitive config.get() calls.
"""
from dataclasses import dataclass, field, fields, MISSING
from typing import Dict, List, Any, Optional, get_type_hints
from .load_config import CoatingConfigParser


@dataclass
class BaseConfig:
    """Base configuration class with automatic field mapping from INI sections."""
    
    @classmethod
    def from_config_section(cls, config: CoatingConfigParser, section_name: str, 
                          strict=True, **defaults):
        """Automatically create config from INI section using dataclass field names.
        
        Args:
            config: Configuration parser
            section_name: INI section name
            strict: If True, raises error for unknown config parameters (catches typos)
            **defaults: Default values for optional fields
        """
        kwargs = {}
        
        # Get type hints for proper type conversion
        type_hints = get_type_hints(cls)
        
        # Get all expected field names from the dataclass
        expected_fields = {field_info.name for field_info in fields(cls)}
        
        if strict:
            # Get all actual field names from the config section
            try:
                actual_fields = set(config.options(section_name))
            except:
                actual_fields = set()
            
            # Check for typos/unknown fields in config
            unknown_fields = actual_fields - expected_fields
            if unknown_fields:
                # Create helpful error message with suggestions
                error_msg = f"Unknown configuration parameters in section '[{section_name}]': {sorted(unknown_fields)}.\n"
                error_msg += f"Valid parameters are: {sorted(expected_fields)}\n"
                
                # Try to suggest corrections for potential typos
                suggestions = []
                for unknown in unknown_fields:
                    close_matches = [field for field in expected_fields 
                                   if abs(len(field) - len(unknown)) <= 2 and
                                   sum(c1 != c2 for c1, c2 in zip(field, unknown)) <= 2]
                    if close_matches:
                        suggestions.append(f"'{unknown}' -> did you mean '{close_matches[0]}'?")
                
                if suggestions:
                    error_msg += "Possible corrections:\n" + "\n".join(f"  {s}" for s in suggestions)
                
                raise ValueError(error_msg)
        
        for field_info in fields(cls):
            field_name = field_info.name
            field_type = type_hints.get(field_name, str)
            
            # Check if there's a default provided
            default_value = defaults.get(field_name, field_info.default if field_info.default is not MISSING else None)
            
            try:
                # Get value from config, using fallback if provided
                if default_value is not None:
                    value = config.get(section_name, field_name, fallback=default_value)
                else:
                    value = config.get(section_name, field_name)
                
                kwargs[field_name] = value
                
            except Exception as e:
                if default_value is not None:
                    kwargs[field_name] = default_value
                else:
                    raise ValueError(f"Required field '{field_name}' not found in section '{section_name}' and no default provided: {e}")
        
        return cls(**kwargs)


@dataclass
class GeneralConfig(BaseConfig):
    """General configuration parameters."""
    root_dir: str
    data_dir: str
    load_model: bool
    load_model_path: str
    materials_file: str
    continue_training: bool


@dataclass
class DataConfig(BaseConfig):
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
    design_criteria: Dict[str, float] 
    use_ligo_reward: bool
    include_random_rare_state: bool
    use_optical_thickness: bool
    combine: str
    reward_function: str


@dataclass
class NetworkConfig(BaseConfig):
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
    buffer_size: int = 10000  # Default buffer size for replay memory


@dataclass
class TrainingConfig(BaseConfig):
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
    t_mult: float
    
    # Pareto optimisation
    n_init_solutions: int
    final_weight_epoch: int
    start_weight_alpha: float
    final_weight_alpha: float
    cycle_weights: bool
    n_weight_cycles: int
    weight_network_save: bool


@dataclass
class GeneticConfig(BaseConfig):
    """Genetic algorithm configuration parameters."""
    algorithm: str  # NSGA2, NSGA3, MOEAD
    population_size: int
    n_generations: int
    crossover_probability: float
    crossover_eta: float
    mutation_probability: float
    mutation_eta: float
    eliminate_duplicates: bool
    n_neighbors: Optional[int] = None  # For MOEAD
    prob_neighbor_mating: Optional[float] = None  # For MOEAD
    n_partitions: Optional[int] = None  # For NSGA3/MOEAD reference directions
    seed: int = 1234
    thickness_sigma: float = 1e-4


@dataclass
class CoatingOptimisationConfig:
    """Complete configuration for coating optimisation."""
    general: GeneralConfig
    data: DataConfig
    network: Optional[NetworkConfig] = None
    training: Optional[TrainingConfig] = None
    genetic: Optional[GeneticConfig] = None
    
    @classmethod
    def from_config_parser(cls, config: CoatingConfigParser) -> 'CoatingOptimisationConfig':
        """Create structured config from ConfigParser using automatic field mapping."""
        
        # Define defaults for optional fields
        data_defaults = {
            'reflectivity_reward_shape': 'none',
            'absorption_reward_shape': 'none',
            'optimise_weight_ranges': {},
            'use_optical_thickness': True,
            'combine': 'logproduct'
        }
        
        network_defaults = {
            'n_continuous_layers': 2,
            'n_discrete_layers': 2,
            'n_value_layers': 2,
            'discrete_hidden_size': 16,
            'continuous_hidden_size': 16,
            'value_hidden_size': 16
        }
        
        training_defaults = {
            'cycle_weights': False
        }
        
        genetic_defaults = {
            'eliminate_duplicates': True,
            'seed': 10,
            'thickness_sigma': 1e-4
        }
        
        # Create config objects automatically
        general = GeneralConfig.from_config_section(config, "General")
        data = DataConfig.from_config_section(config, "Data", **data_defaults)
        
        # Optional sections
        network = None
        training = None
        genetic = None
        
        if config.has_section("Network"):
            network = NetworkConfig.from_config_section(config, "Network", **network_defaults)
        
        if config.has_section("Training"):
            training = TrainingConfig.from_config_section(config, "Training", **training_defaults)
        
        if config.has_section("Genetic"):
            genetic = GeneticConfig.from_config_section(config, "Genetic", **genetic_defaults)
        
        return cls(
            general=general,
            data=data,
            network=network,
            training=training,
            genetic=genetic
        )