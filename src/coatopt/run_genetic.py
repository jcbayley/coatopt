import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import time
import matplotlib.pyplot as plt
#from deepqn_cart import plotLearning
import copy 
from coatopt.environments.thermal_noise_environment_genetic import GeneticCoatingStack
from coatopt.config import read_config, read_materials
from coatopt.networks.genetic_algorithm import StatePool
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
        thickness_sigma=config.get("Genetic", "thickness_sigma"),
        opt_init=False)

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
    for i in range(num_iterations):
        #statepool.fraction_keep_states = min(0.91 - 3*(i/num_iterations), 0.05)
        sort_state_values, top_state = statepool.evolve_step()
        env.thickness_sigma = env.thickness_sigma * 1.0#0.99
        score = np.mean(sort_state_values[:,1])
        scores.append(score)
        max_scores.append(sort_state_values[0,1])
        if i % 10 == 0:
            print('episode: ', i,'score %.5f ' % score)

            fig, ax = plt.subplots()
            ax.plot(scores[:])
            fig.savefig(os.path.join(config.get("General", "root_dir"), "scores.png"))

            fig, ax = plt.subplots()
            ax.plot(np.log(np.abs(max_scores))[:])
            fig.savefig(os.path.join(config.get("General", "root_dir"), "max_scores.png"))

            
            fig, ax = env.plot_stack(top_state)
            fig.savefig(os.path.join(config.get("General", "root_dir"), "stackplots", f"it{i}.png"))

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

    opt_state = env.get_optimal_state()
    opt_score = env.compute_reward(opt_state)
    print("--------------")
    #print(opt_state)
    print(f"opt state score: {opt_score}")
    #plotting.plot_coating(top_state, os.path.join(root_dir, "coating.png"))

    #print(top_state_value)


