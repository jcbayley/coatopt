import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
#from deepqn_cart import plotLearning
import copy 
from coatopt.environments.thermal_noise_environment_genetic import GeneticCoatingStack
from coatopt.config import read_config, read_materials
from coatopt.src.coatopt.algorithms.genetic_algorithms.genetic_algorithm import StatePool
from coatopt.environments import coating_utils
import argparse
#import plotting




if __name__ == '__main__':
    #env = gym.make('CartPole-v1')
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=False, default="none")
    parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    config = read_config(os.path.abspath(args.config)) 


    materials = read_materials(os.path.join(config.get("General", "materials_file")))

    env = GeneticCoatingStack(
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
        thickness_sigma=config.get("Genetic", "thickness_sigma"),
        combine= config.get("Data", "combine"),
    )

    if not os.path.isdir(config.get("General", "root_dir")):
        os.makedirs(config.get("General", "root_dir"))

    
    num_iterations = config.get("Genetic", "num_iterations")
    statepool = StatePool(
        env, 
        n_states=config.get("Genetic", "n_states"), 
        states_fraction_keep = config.get("Genetic", "states_fraction_keep"),
    )

    filename = os.path.join(config.get("General", "root_dir"),'coating.png')
    scores = []
    max_scores = []
    eps_history = []
    n_steps = 0
    final_state = None
    if not os.path.isdir(os.path.join(config.get("General", "root_dir"), "stackplots")):
        os.makedirs(os.path.join(config.get("General", "root_dir"), "stackplots"))
    
    n_mean_calc = 100
    best_state = None
    best_state_value = -np.inf
    for i in range(num_iterations):
        #statepool.fraction_keep_states = min(0.91 - 3*(i/num_iterations), 0.05)
        sort_state_values, top_state = statepool.evolve_step()
        env.thickness_sigma = env.thickness_sigma * 1.0#0.99
        score = np.mean(sort_state_values[:,1])
        scores.append(score)
        max_scores.append(sort_state_values[0,1])
        with open(os.path.join(config.get("General", "root_dir"), "max_scores.txt"), 'w') as f:
            np.savetxt(f, max_scores)

        if sort_state_values[0,1] > best_state_value:
            best_state_value = sort_state_values[0,1]
            best_state = top_state
            with open(os.path.join(config.get("General", "root_dir"), "best_state.txt"), 'w') as f:
                f.write(str(best_state_value))
                f.write(str(best_state))

        if i % 10 == 0:
            print('episode: ', i,'score %.5f ' % score)

            if best_state is not None:
                fig, ax = plt.subplots()
                ax.plot(scores[:])
                ax.set_ylabel("mean batch score")
                ax.set_xlabel("iteration")
                fig.savefig(os.path.join(config.get("General", "root_dir"), "scores.png"))

                fig, ax = plt.subplots()
                ax.plot(max_scores[:])
                ax.set_ylabel("max_score")
                ax.set_xlabel("iteration")
                fig.savefig(os.path.join(config.get("General", "root_dir"), "max_scores.png"))

                
                fig, ax = env.plot_stack(top_state)
                fig.savefig(os.path.join(config.get("General", "root_dir"), "stackplots", f"it{i}.png"))

                fig, ax = env.plot_stack(best_state)
                fig.savefig(os.path.join(config.get("General", "root_dir"),  f"best_state.png"))

                if i > 100:
                    fig, ax = plt.subplots()
                    ax.plot(scores[-100:])
                    fig.savefig(os.path.join(config.get("General", "root_dir"), "scores_zoom.png"))

                    fig, ax = plt.subplots()
                    ax.plot(max_scores[-100:])
                    fig.savefig(os.path.join(config.get("General", "root_dir"), "max_scores_zoom.png"))

    sorted_state_values = statepool.order_states()
    top_states = statepool.current_states[sorted_state_values[:10, 0].astype(int)]
    #print(top_states)
    #print(sorted_state_values[:10])

    top_state = top_states[0]

    print("-----------------------------")
    print(top_state)
    top_score = env.compute_reward(top_state)
    print(top_score)

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

    #print(top_state_value)


