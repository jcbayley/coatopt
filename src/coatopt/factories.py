"""
Factory functions for creating environment and agent objects.
Simplifies object creation and reduces parameter passing complexity.

Main Entry Points:
- create_environment(): Creates appropriate environment based on model_type
- setup_training_pipeline(): Complete pipeline setup for any algorithm type
- get_supported_model_types(): List available model types and descriptions

Supported Algorithm Types:
- hppo: Single-objective RL with PC-HPPO agent
- multiobjective/pareto: Multi-objective RL with Pareto optimization  
- genetic: Genetic algorithm optimization (NSGA-II, NSGA-III, MOEA/D)
"""
from typing import Dict, Any, Tuple, Optional, Union
from coatopt.algorithms import hppo
from coatopt.algorithms.genetic_algorithms.genetic_moo import GeneticTrainer
from coatopt.environments.hppo_environment import HPPOEnvironment
from coatopt.environments.multiobjective_environment import MultiObjectiveEnvironment
from coatopt.environments.genetic_environment import GeneticCoatingStack
from coatopt.config.structured_config import CoatingOptimisationConfig
import os



def create_environment(config: CoatingOptimisationConfig, materials: Dict[int, Dict[str, Any]]) -> Union[HPPOEnvironment, MultiObjectiveEnvironment, GeneticCoatingStack]:
    """
    Create appropriate environment based on configuration.
    
    Maps model types to environments:
    - "hppo" -> HPPOEnvironment (single-objective RL)
    - "multiobjective" or "pareto" -> MultiObjectiveEnvironment (multi-objective RL with Pareto fronts)  
    - "genetic" -> GeneticCoatingStack (genetic algorithm optimization)
    
    Args:
        config: Structured configuration object
        materials: Materials dictionary
        
    Returns:
        Environment instance of the appropriate type
        
    Raises:
        ValueError: If model_type is not supported or network config is missing
    """
    if config.network is None:
        raise ValueError("Network configuration is required to determine model type")
    
    model_type = config.network.model_type.lower()
    
    if model_type == "hppo":
        return create_hppo_environment(config, materials)
    elif model_type == "hppo_multiobjective":
        return create_multiobjective_environment(config, materials)
    elif model_type == "genetic":
        return create_genetic_environment(config, materials)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: hppo, multiobjective, genetic")


def create_hppo_environment(config: CoatingOptimisationConfig, materials: Dict[int, Dict[str, Any]]) -> HPPOEnvironment:
    """
    Create HPPOEnvironment from structured configuration.
    
    Args:
        config: Structured configuration object
        materials: Materials dictionary
        
    Returns:
        Configured HPPOEnvironment
    """
    env = HPPOEnvironment(
        max_layers=config.data.n_layers,
        min_thickness=config.data.min_thickness,
        max_thickness=config.data.max_thickness,
        materials=materials,
        opt_init=False,
        use_intermediate_reward=config.data.use_intermediate_reward,
        ignore_air_option=config.data.ignore_air_option,
        ignore_substrate_option=config.data.ignore_substrate_option,
        optimise_parameters=config.data.optimise_parameters,
        optimise_targets=config.data.optimise_targets,
        design_criteria=config.data.design_criteria,
        use_optical_thickness=config.data.use_optical_thickness,
        combine=config.data.combine,
        optimise_weight_ranges=config.data.optimise_weight_ranges,
        reward_function=config.data.reward_function,
    )
    return env


def create_multiobjective_environment(config: CoatingOptimisationConfig, materials: Dict[int, Dict[str, Any]]) -> MultiObjectiveEnvironment:
    """
    Create MultiObjectiveEnvironment (Pareto) from structured configuration.
    
    Args:
        config: Structured configuration object
        materials: Materials dictionary
        
    Returns:
        Configured MultiObjectiveEnvironment
    """
    env = MultiObjectiveEnvironment(
        max_layers=config.data.n_layers,
        min_thickness=config.data.min_thickness,
        max_thickness=config.data.max_thickness,
        materials=materials,
        opt_init=False,
        use_intermediate_reward=config.data.use_intermediate_reward,
        ignore_air_option=config.data.ignore_air_option,
        ignore_substrate_option=config.data.ignore_substrate_option,
        optimise_parameters=config.data.optimise_parameters,
        optimise_targets=config.data.optimise_targets,
        design_criteria=config.data.design_criteria,
        use_optical_thickness=config.data.use_optical_thickness,
        combine=config.data.combine,
        optimise_weight_ranges=config.data.optimise_weight_ranges,
        reward_function=config.data.reward_function,
        final_weight_epoch=config.training.final_weight_epoch if config.training else 1,
        start_weight_alpha=config.training.start_weight_alpha if config.training else 1.0,
        final_weight_alpha=config.training.final_weight_alpha if config.training else 1.0,
        cycle_weights=config.training.cycle_weights if config.training else False,
        n_weight_cycles=config.training.n_weight_cycles if config.training else 2,
    )
    return env


def create_pareto_environment(config: CoatingOptimisationConfig, materials: Dict[int, Dict[str, Any]]) -> MultiObjectiveEnvironment:
    """
    Create MultiObjectiveEnvironment (backward compatibility name).
    
    Args:
        config: Structured configuration object
        materials: Materials dictionary
        
    Returns:
        Configured MultiObjectiveEnvironment
    """
    return create_multiobjective_environment(config, materials)


def create_pc_hppo_agent(config: CoatingOptimisationConfig, env: Union[HPPOEnvironment, MultiObjectiveEnvironment]) -> hppo.PCHPPO:
    """
    Create PC-HPPO agent from structured configuration.
    
    Args:
        config: Structured configuration object
        env: Environment object (needed for input/output dimensions)
        
    Returns:
        Configured PC-HPPO agent
    """
    # Determine input size based on observation vs state
    input_size = env.obs_space_shape if config.data.use_observation else env.state_space_shape
    print(f"Creating PC-HPPO agent with input size: {input_size}")
    
    agent = hppo.PCHPPO(
        input_size,
        env.n_materials,
        1,  # output_size
        hidden_size=config.network.hidden_size,
        lr_discrete_policy=config.training.lr_discrete_policy,
        lr_continuous_policy=config.training.lr_continuous_policy,
        lr_value=config.training.lr_value,
        lr_step=config.training.lr_step,
        lr_min=config.training.lr_min,
        lower_bound=0,
        upper_bound=1,
        n_updates=config.training.n_episodes_per_update,
        beta=config.training.entropy_beta_start,
        clip_ratio=config.training.clip_ratio,
        gamma=config.training.gamma,
        include_layer_number=config.network.include_layer_number,
        include_material_in_policy=config.network.include_material_in_policy,
        pre_type=config.network.pre_network_type,
        n_heads=2,  # Fixed parameter - could be configurable if needed
        n_pre_layers=config.network.n_pre_layers,
        optimiser=config.training.optimiser,
        n_continuous_layers=config.network.n_continuous_layers,
        n_discrete_layers=config.network.n_discrete_layers,
        n_value_layers=config.network.n_value_layers,
        discrete_hidden_size=config.network.discrete_hidden_size,
        continuous_hidden_size=config.network.continuous_hidden_size,
        value_hidden_size=config.network.value_hidden_size,
        substrate_material_index=env.substrate_material_index,
        air_material_index=env.air_material_index,
        ignore_air_option=config.data.ignore_air_option,
        ignore_substrate_option=config.data.ignore_substrate_option,
        num_objectives=len(config.data.optimise_parameters),
        entropy_beta_start=config.training.entropy_beta_start,
        entropy_beta_end=config.training.entropy_beta_end,
        entropy_beta_decay_length=config.training.entropy_beta_decay_length,
        entropy_beta_discrete_start=config.training.entropy_beta_discrete_start,
        entropy_beta_discrete_end=config.training.entropy_beta_discrete_end,
        entropy_beta_continuous_start=config.training.entropy_beta_continuous_start,
        entropy_beta_continuous_end=config.training.entropy_beta_continuous_end,
        hyper_networks=config.network.hyper_networks,
    )
    return agent


def create_trainer(config: CoatingOptimisationConfig, agent: hppo.PCHPPO, env: Union[HPPOEnvironment, MultiObjectiveEnvironment], continue_training: bool = False) -> hppo.HPPOTrainer:
    """
    Create HPPO trainer from structured configuration.
    
    Args:
        config: Structured configuration object
        agent: PC-HPPO agent
        env: Environment object
        continue_training: Whether to continue from existing checkpoint
        
    Returns:
        Configured HPPO trainer
    """
    trainer = hppo.HPPOTrainer(
        agent=agent,
        env=env,
        n_iterations=config.training.n_iterations,
        n_layers=config.data.n_layers,
        root_dir=config.general.root_dir,
        use_obs=config.data.use_observation,
        entropy_beta_start=config.training.entropy_beta_start,
        entropy_beta_end=config.training.entropy_beta_end,
        entropy_beta_decay_length=config.training.entropy_beta_decay_length,
        entropy_beta_decay_start=config.training.entropy_beta_decay_start,
        n_epochs_per_update=config.training.n_epochs_per_update,
        scheduler_start=config.training.scheduler_start,
        scheduler_end=config.training.scheduler_end,
        continue_training=continue_training,
        weight_network_save=config.training.weight_network_save,
    )
    return trainer


def load_model_if_needed(agent: hppo.PCHPPO, config: CoatingOptimisationConfig, continue_training: bool) -> None:
    """
    Load pre-trained model weights if specified in configuration.
    
    Args:
        agent: PC-HPPO agent to load weights into
        config: Configuration object
        continue_training: Whether continuing training
    """
    if config.general.load_model or continue_training:
        if config.general.load_model_path == "root" or continue_training:
            agent.load_networks(os.path.join(config.general.root_dir, hppo.HPPOConstants.NETWORK_WEIGHTS_DIR))
        else:
            agent.load_networks(config.general.load_model_path)
        print(f"Loaded model from: {config.general.load_model_path if config.general.load_model_path != 'root' else config.general.root_dir}")


def setup_optimisation_pipeline(config: CoatingOptimisationConfig, materials: Dict[int, Dict[str, Any]], continue_training: bool = False, init_pareto_front: bool = True) -> Tuple[Union[HPPOEnvironment, MultiObjectiveEnvironment], hppo.PCHPPO, hppo.HPPOTrainer]:
    """
    Complete setup of the optimisation pipeline.
    
    Args:
        config: Structured configuration object
        materials: Materials dictionary
        continue_training: Whether to continue from checkpoint
        init_pareto_front: Whether to initialize Pareto front (only applies to multiobjective environments)
        
    Returns:
        Tuple of (environment, agent, trainer)
    """
    print("Setting up environment...")
    env = create_environment(config, materials)
    
    print("Creating agent...")
    agent = create_pc_hppo_agent(config, env)
    
    print("Loading model if needed...")
    load_model_if_needed(agent, config, continue_training)
    
    print("Setting up trainer...")
    trainer = create_trainer(config, agent, env, continue_training)
    
    # Only initialize Pareto front for multiobjective environments
    if init_pareto_front and isinstance(env, MultiObjectiveEnvironment):
        print("Initializing Pareto front...")
        trainer.init_pareto_front(n_solutions=config.training.n_init_solutions)
    
    print("Pipeline setup complete.")
    return env, agent, trainer


def create_genetic_environment(config: CoatingOptimisationConfig, materials: Dict[int, Dict[str, Any]]) -> GeneticCoatingStack:
    """
    Create GeneticCoatingStack environment from structured configuration.
    
    Args:
        config: Structured configuration object
        materials: Materials dictionary
        
    Returns:
        Configured GeneticCoatingStack environment
    """
    env = GeneticCoatingStack(
        max_layers=config.data.n_layers,
        min_thickness=config.data.min_thickness,
        max_thickness=config.data.max_thickness,
        materials=materials,
        opt_init=False,
        use_intermediate_reward=config.data.use_intermediate_reward,
        reflectivity_reward_shape=config.data.reflectivity_reward_shape,
        thermal_reward_shape=config.data.thermal_reward_shape,
        absorption_reward_shape=config.data.absorption_reward_shape,
        ignore_air_option=config.data.ignore_air_option,
        ignore_substrate_option=config.data.ignore_substrate_option,
        use_ligo_reward=config.data.use_ligo_reward,
        optimise_parameters=config.data.optimise_parameters,
        optimise_targets=config.data.optimise_targets,
        include_random_rare_state=config.data.include_random_rare_state,
        use_optical_thickness=config.data.use_optical_thickness,
        thickness_sigma=config.genetic.thickness_sigma,
        combine=config.data.combine,
        reward_function=config.data.reward_function,
    )
    return env


def create_genetic_trainer(config: CoatingOptimisationConfig, env: GeneticCoatingStack) -> GeneticTrainer:
    """
    Create genetic algorithm trainer from structured configuration.
    
    Args:
        config: Structured configuration object
        env: Genetic environment object
        
    Returns:
        Configured genetic trainer
    """
    trainer = GeneticTrainer(
        environment=env,
        config=config.genetic,
        output_dir=config.general.root_dir
    )
    return trainer


def setup_genetic_optimisation_pipeline(config: CoatingOptimisationConfig, materials: Dict[int, Dict[str, Any]]) -> Tuple[GeneticCoatingStack, GeneticTrainer]:
    """
    Complete setup of the genetic optimisation pipeline.
    
    Args:
        config: Structured configuration object
        materials: Materials dictionary
        
    Returns:
        Tuple of (environment, trainer)
    """
    print("Setting up genetic environment...")
    env = create_genetic_environment(config, materials)
    
    print("Creating genetic trainer...")
    trainer = create_genetic_trainer(config, env)
    
    print("Genetic pipeline setup complete.")
    return env, trainer


def setup_training_pipeline(config: CoatingOptimisationConfig, materials: Dict[int, Dict[str, Any]], continue_training: bool = False, init_pareto_front: bool = True):
    """
    Setup the appropriate training pipeline based on model type.
    
    Args:
        config: Structured configuration object
        materials: Materials dictionary
        continue_training: Whether to continue from checkpoint
        init_pareto_front: Whether to initialize Pareto front (only applies to multiobjective)
        
    Returns:
        Tuple appropriate for the selected algorithm type
    """
    # Validate configuration first
    validate_model_config(config)
    
    model_type = config.network.model_type.lower()
    
    if model_type == "genetic":
        return setup_genetic_optimisation_pipeline(config, materials)
    elif model_type in ["hppo", "multiobjective", "pareto"]:
        return setup_optimisation_pipeline(config, materials, continue_training, init_pareto_front)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: hppo, multiobjective, genetic")


def get_supported_model_types() -> Dict[str, str]:
    """
    Get dictionary of supported model types and their descriptions.
    
    Returns:
        Dictionary mapping model type to description
    """
    return {
        "hppo": "Single-objective reinforcement learning with PC-HPPO agent",
        "multiobjective": "Multi-objective reinforcement learning with Pareto fronts", 
        "pareto": "Multi-objective reinforcement learning with Pareto fronts (alias for multiobjective)",
        "genetic": "Genetic algorithm optimization (NSGA-II, NSGA-III, MOEA/D)"
    }


def validate_model_config(config: CoatingOptimisationConfig) -> None:
    """
    Validate that the model configuration is consistent and complete.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid or incomplete
    """
    if config.network is None:
        raise ValueError("Network configuration is required")
    
    model_type = config.network.model_type.lower()
    
    if model_type not in get_supported_model_types():
        supported = list(get_supported_model_types().keys())
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: {supported}")
    
    # Check algorithm-specific requirements
    if model_type == "genetic":
        if config.genetic is None:
            raise ValueError("Genetic configuration section is required for genetic algorithms")
    
    if model_type in ["hppo", "multiobjective", "pareto"]:
        if config.training is None:
            raise ValueError("Training configuration section is required for RL algorithms")