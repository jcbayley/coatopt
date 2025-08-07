"""
Factory functions for creating environment and agent objects.
Simplifies object creation and reduces parameter passing complexity.
"""
from typing import Dict, Any
from coatopt.algorithms import pc_hppo_oml
from coatopt.environments.thermal_noise_environment_pareto import ParetoCoatingStack
from coatopt.config.structured_config import CoatingoptimisationConfig
from typing import Tuple




def create_pareto_environment(config: CoatingoptimisationConfig, materials: Dict[int, Dict[str, Any]]) -> ParetoCoatingStack:
    """
    Create ParetoCoatingStack environment from structured configuration.
    
    Args:
        config: Structured configuration object
        materials: Materials dictionary
        
    Returns:
        Configured ParetoCoatingStack environment
    """
    env = ParetoCoatingStack(
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
        design_criteria=config.data.design_criteria,
        include_random_rare_state=config.data.include_random_rare_state,
        use_optical_thickness=config.data.use_optical_thickness,
        combine=config.data.combine,
        optimise_weight_ranges=config.data.optimise_weight_ranges,
        reward_function=config.data.reward_function,
        final_weight_epoch=config.training.final_weight_epoch,
        start_weight_alpha=config.training.start_weight_alpha,
        final_weight_alpha=config.training.final_weight_alpha,
        cycle_weights=config.training.cycle_weights,
        n_weight_cycles=config.training.n_weight_cycles,
    )
    return env


def create_pc_hppo_agent(config: CoatingoptimisationConfig, env: ParetoCoatingStack) -> pc_hppo_oml.PCHPPO:
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
    
    agent = pc_hppo_oml.PCHPPO(
        input_size,
        env.n_materials,
        1,  # output_size
        hidden_size=config.network.hidden_size,
        disc_lr_policy=config.training.lr_discrete_policy,
        cont_lr_policy=config.training.lr_continuous_policy,
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
        beta_start=config.training.entropy_beta_start,
        beta_end=config.training.entropy_beta_end,
        beta_decay_length=config.training.entropy_beta_decay_length,
        hyper_networks=config.network.hyper_networks,
    )
    return agent


def create_trainer(config: CoatingoptimisationConfig, agent: pc_hppo_oml.PCHPPO, env: ParetoCoatingStack, continue_training: bool = False) -> pc_hppo_oml.HPPOTrainer:
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
    trainer = pc_hppo_oml.HPPOTrainer(
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


def load_model_if_needed(agent: pc_hppo_oml.PCHPPO, config: CoatingoptimisationConfig, continue_training: bool) -> None:
    """
    Load pre-trained model weights if specified in configuration.
    
    Args:
        agent: PC-HPPO agent to load weights into
        config: Configuration object
        continue_training: Whether continuing training
    """
    if config.general.load_model or continue_training:
        if config.general.load_model_path == "root" or continue_training:
            agent.load_networks(config.general.root_dir)
        else:
            agent.load_networks(config.general.load_model_path)
        print(f"Loaded model from: {config.general.load_model_path if config.general.load_model_path != 'root' else config.general.root_dir}")


def setup_optimisation_pipeline(config: CoatingoptimisationConfig, materials: Dict[int, Dict[str, Any]], continue_training: bool = False, init_pareto_front: bool = True) -> Tuple[ParetoCoatingStack, pc_hppo_oml.PCHPPO, pc_hppo_oml.HPPOTrainer]:
    """
    Complete setup of the optimisation pipeline.
    
    Args:
        config: Structured configuration object
        materials: Materials dictionary
        continue_training: Whether to continue from checkpoint
        
    Returns:
        Tuple of (environment, agent, trainer)
    """
    print("Setting up environment...")
    env = create_pareto_environment(config, materials)
    
    print("Creating agent...")
    agent = create_pc_hppo_agent(config, env)
    
    print("Loading model if needed...")
    load_model_if_needed(agent, config, continue_training)
    
    print("Setting up trainer...")
    trainer = create_trainer(config, agent, env, continue_training)
    if init_pareto_front:
        print("Initializing Pareto front...")
        trainer.init_pareto_front(n_solutions=config.training.n_init_solutions)
    
    print("Pipeline setup complete.")
    return env, agent, trainer