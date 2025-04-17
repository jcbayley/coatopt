from coatopt.algorithms import pc_hppo_oml
from coatopt.environments import CoatingStack, coating_utils
from coatopt.config import read_config, read_materials
from coatopt.train_coating_hppo import training_loop
import os
import argparse
import numpy as np
import h5py

if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=False, default="none")
    parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--continue-training', action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    config = read_config(os.path.abspath(args.config)) 


    materials = read_materials(os.path.join(config.get("General", "materials_file")))

    continue_training = args.continue_training or config.get("General", "continue_training")

    env = CoatingStack(
        config.get("Data", "n_layers"), 
        config.get("Data", "min_thickness"),
        config.get("Data", "max_thickness"),
        materials,
        opt_init=False,
        use_intermediate_reward = config.get("Data", "use_intermediate_reward"),
        reflectivity_reward_shape=config.get("Data", "reflectivity_reward_shape"),
        thermal_reward_shape=config.get("Data", "thermal_reward_shape"),
        absorption_reward_shape=config.get("Data", "absorption_reward_shape"),                
        ignore_air_option=config.get("Data", "ignore_air_option"),
        ignore_substrate_option=config.get("Data", "ignore_substrate_option"),
        use_ligo_reward=config.get("Data", "use_ligo_reward"),
        optimise_parameters=config.get("Data", "optimise_parameters"),
        optimise_targets=config.get("Data", "optimise_targets"),
        include_random_rare_state=config.get("Data", "include_random_rare_state"),
        use_optical_thickness=config.get("Data", "use_optical_thickness"),
        combine= config.get("Data", "combine"),
        optimise_weight_ranges=config.get("Data", "optimise_weight_ranges"),
    )


    device = "cpu"

    insize = env.obs_space_shape if config.get("Data", "use_observation")==True else env.state_space_shape

    agent = pc_hppo_oml.PCHPPO(
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
            substrate_material_index=env.substrate_material_index,
            air_material_index=env.air_material_index,
            ignore_air_option=config.get("Data", "ignore_air_option"),
            ignore_substrate_option=config.get("Data", "ignore_substrate_option"),
            num_objectives=len(config.get("Data", "optimise_targets"))
            )
 
    
    if config.get("General", "load_model") or continue_training:
        if config.get("General", "load_model_path") == "root" or continue_training:
            agent.load_networks(config.get("General", "root_dir"))
        else:
            agent.load_networks(config.get("General", "load_model_path"))


    def convert_state_to_physical(state, env):
        for l_ind in range(len(state)):
            state[l_ind,0] = coating_utils.optical_to_physical(state[l_ind,0], env.light_wavelength, env.materials[np.argmax(state[l_ind][1:])]['n'])
        return state
    
    optimal_state = env.get_optimal_state(inds_alternate=[1,2])
    optimal_value = env.compute_state_value(optimal_state, return_separate=True)
    if env.use_optical_thickness:
        optimal_stat = convert_state_to_physical(optimal_state, env)
    fig, ax = env.plot_stack(optimal_state)
    ax.set_title(f" opt val: {optimal_value}")
    fig.savefig(os.path.join(config.get("General", "root_dir"),  f"opt_state.png"))

    optimal_state_r = env.get_optimal_state(inds_alternate=[2,1])
    optimal_value_r = env.compute_state_value(optimal_state_r, return_separate=True)
    if env.use_optical_thickness:
        optimal_state_r = convert_state_to_physical(optimal_state_r, env)
    fig, ax = env.plot_stack(optimal_state_r)
    ax.set_title(f" opt val: {optimal_value_r}")
    fig.savefig(os.path.join(config.get("General", "root_dir"),  f"opt_state_reverse.png"))

    optimal_state_2 = env.get_optimal_state(inds_alternate=[1,3])
    optimal_value_2 = env.compute_state_value(optimal_state_2, return_separate=True)
    if env.use_optical_thickness:
        optimal_state_2 = convert_state_to_physical(optimal_state_2, env)
    fig, ax = env.plot_stack(optimal_state_2)
    ax.set_title(f" opt val: {optimal_value_2}")
    fig.savefig(os.path.join(config.get("General", "root_dir"),  f"opt_state_2.png"))

    optimal_state_r2 = env.get_optimal_state(inds_alternate=[3,1])
    optimal_value_r2 = env.compute_state_value(optimal_state_r2, return_separate=True)
    if env.use_optical_thickness:
        optimal_state_r2 = convert_state_to_physical(optimal_state_r2, env)
    fig, ax = env.plot_stack(optimal_state_r2)
    ax.set_title(f" opt val: {optimal_value_r2}")
    fig.savefig(os.path.join(config.get("General", "root_dir"),  f"opt_state_reverse_2.png"))
    
    optimal_state_2m = env.get_optimal_state_2mat(inds_alternate=[1,3,2])
    optimal_value_2m = env.compute_state_value(optimal_state_2m, return_separate=True)
    if env.use_optical_thickness:
        optimal_state_2m = convert_state_to_physical(optimal_state_2m, env)
    fig, ax = env.plot_stack(optimal_state_2m)
    ax.set_title(f" opt val: {optimal_value_2m}")
    fig.savefig(os.path.join(config.get("General", "root_dir"),  f"opt_state_2mat.png"))

    optimal_state_2rm = env.get_optimal_state_2mat(inds_alternate=[1,2,3])
    optimal_value_2rm = env.compute_state_value(optimal_state_2rm, return_separate=True)
    if env.use_optical_thickness:
        optimal_state_2rm = convert_state_to_physical(optimal_state_2rm, env)
    fig, ax = env.plot_stack(optimal_state_2rm)
    ax.set_title(f" opt val: {optimal_value_2rm}")
    fig.savefig(os.path.join(config.get("General", "root_dir"),  f"opt_state_2mat_reverse.png"))

    trainer = pc_hppo_oml.HPPOTrainer(
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
        scheduler_start=config.get("Training", "scheduler_start"),
        scheduler_end=config.get("Training", "scheduler_end"),
        continue_training=continue_training
    )
    
    if args.train:
        trainer.train()

    if args.test:
        sampled_states, sampled_rewards, sampled_weights, sampled_vals = trainer.generate_solutions(1000)

        rewards = {}
        for key in sampled_rewards[0].keys():
            rewards[f"{key}_rewards"] = np.array([sampled_rewards[i][key] for i in range(len(sampled_rewards))])

        for key in sampled_vals[0].keys():
            rewards[f"{key}_vals"] = np.array([sampled_vals[i][key] for i in range(len(sampled_vals))])

        with h5py.File("sampled_outputs.h5", "w") as f:
            f.create_dataset("states", data=sampled_states)
            f.create_dataset("weights", data=sampled_weights)
            for key in rewards.keys():
                f.create_dataset(key, data=rewards[key])
    
