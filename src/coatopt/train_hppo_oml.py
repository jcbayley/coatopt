from coatopt.algorithms import hppo_oml
from coatopt.environments import CoatingStack
from coatopt.config import read_config, read_materials
from coatopt.train_coating_hppo import training_loop
import os
import argparse

if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=False, default="none")
    parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    config = read_config(os.path.abspath(args.config)) 


    materials = read_materials(os.path.join(config.get("General", "materials_file")))


    env = CoatingStack(
        config.get("Data", "n_layers"), 
        config.get("Data", "min_thickness"),
        config.get("Data", "max_thickness"),
        materials,
        opt_init=False,
        use_intermediate_reward = config.get("Data", "use_intermediate_reward"),
        reward_shape=config.get("Data", "reward_shape"),
        ignore_air_option=config.get("Data", "ignore_air_option"),
        use_ligo_reward=config.get("Data", "use_ligo_reward"),
        use_ligo_thermal_noise=config.get("Data", "use_ligo_thermal_noise"),
    )


    device = "cpu"

    insize = env.obs_space_shape if config.get("Data", "use_observation")==True else env.state_space_shape

    agent = hppo_oml.HPPO(
            insize,
            env.n_materials, 
            1,
            hidden_size=config.get("Network", "hidden_size"),
            disc_lr_policy=config.get("Training", "lr_discrete_policy"),
            cont_lr_policy=config.get("Training", "lr_continuous_policy"),
            lr_value=config.get("Training", "lr_value"),
            lr_step=config.get("Training", "lr_step"),
            lr_min=config.get("Training", "lr_min"),
            lower_bound=0,
            upper_bound=1,
            n_updates=config.get("Training", "n_episodes_per_update"),
            beta=config.get("Training", "entropy_beta_start"),
            clip_ratio=config.get("Training", "clip_ratio"),
            gamma=config.get("Training", "gamma"),
            include_layer_number=config.get("Network", "include_layer_number"),
            include_material_in_policy=config.get("Network", "include_material_in_policy"),
            pre_type=config.get("Network", "pre_network_type"),
            n_heads=2,
            n_pre_layers=config.get("Network", "n_pre_layers"),
            optimiser=config.get("Training", "optimiser"),
            n_continuous_layers=config.get("Network", "n_continuous_layers"),
            n_discrete_layers=config.get("Network", "n_discrete_layers"),
            n_value_layers=config.get("Network", "n_value_layers"),
            discrete_hidden_size=config.get("Network", "discrete_hidden_size"),
            continuous_hidden_size=config.get("Network", "continuous_hidden_size"),
            value_hidden_size=config.get("Network", "value_hidden_size"),
            substrate_material_index=env.substrate_material_index

            )
 
    
    if config.get("General", "load_model"):
        if config.get("General", "load_model_path") == "root":
            agent.load_networks(config.get("General", "root_dir"))
        else:
            agent.load_networks(config.get("General", "load_model_path"))


    optimal_state = env.get_optimal_state()
    optimal_value = env.compute_state_value(optimal_state)
    fig, ax = env.plot_stack(optimal_state)
    ax.set_title(f" opt val: {optimal_value}")
    fig.savefig(os.path.join(config.get("General", "root_dir"),  f"opt_state.png"))

    optimal_state_r = env.get_optimal_state(reverse=True)
    optimal_value_r = env.compute_state_value(optimal_state_r)
    fig, ax = env.plot_stack(optimal_state_r)
    ax.set_title(f" opt val: {optimal_value_r}")
    fig.savefig(os.path.join(config.get("General", "root_dir"),  f"opt_state_reverse.png"))
    

    trainer = hppo_oml.HPPOTrainer(
        agent, 
        env, 
        config.get("Training", "n_iterations"), 
        config.get("Data", "n_layers"),  
        root_dir=config.get("General", "root_dir"),
        use_obs=config.get("Data", "use_observation"),
        beta_start=config.get("Training", "entropy_beta_start"),
        beta_end=config.get("Training", "entropy_beta_end"),
        beta_decay_length=config.get("Training", "entropy_beta_decay_length"),
        beta_decay_start=config.get("Training", "entropy_beta_decay_start"),
    )
    
    trainer.train()
    
    """
    rewards, max_train_state = training_loop(
        config.get("General", "root_dir"),
        env, 
        agent, 
        max_episodes=config.get("Training", "n_iterations"),
        n_ep_train=config.get("Training", "n_epochs_per_update"),
        max_layers=config.get("Data", "n_layers"),
        useobs=config.get("Data", "use_observation"),
        beta_start=config.get("Training", "entropy_beta_start"),
        beta_end=config.get("Training", "entropy_beta_end"),
        beta_decay_length=config.get("Training", "entropy_beta_decay_length"),
        beta_decay_start=config.get("Training", "entropy_beta_decay_start"),
        upper_bound=config.get("Data", "min_thickness"),
        lower_bound=config.get("Data", "max_thickness"),
        save_interval=config.get("Training", "model_save_interval")
        )
    """