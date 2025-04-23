import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
#from deepqn_cart import plotLearning
import copy 
from coatopt.environments.thermal_noise_environment_genetic import GeneticCoatingStack
from coatopt.config import read_config, read_materials
from coatopt.algorithms.genetic_algorithm import StatePool
from coatopt.environments import coating_utils
from coatopt.tools import plotting
import argparse
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableGA
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.population import Population



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

    class CoatingMoo(ElementwiseProblem):

        def __init__(self, environment, **kwargs):

            self.env = environment
            self.n_var = self.env.max_layers * 2
            n_obj=2
            thick_lower = np.repeat(self.env.min_thickness, self.env.max_layers)
            thick_upper = np.repeat(self.env.max_thickness, self.env.max_layers)
            material_lower = np.repeat(0, self.env.max_layers)
            material_upper = np.repeat(self.env.n_materials-1, self.env.max_layers)
            xl=np.concatenate((thick_lower, material_lower))
            xu=np.concatenate((thick_upper, material_upper))
            """
            vars = {}

            for i in range(self.env.max_layers):
                vars.update({
                    f"layer_{i}": Real(
                        bounds=(self.env.min_thickness,self.env.max_thickness),
                    ),
                    f"layer_{i}_material": Integer(
                        bounds=(0,self.env.n_materials-1),
                    ),
                })
            super().__init__(vars=vars, n_obj=1, **kwargs)
            """
            super().__init__(n_var=self.n_var, n_obj=n_obj, xl=xl, xu=xu, **kwargs)

        def make_state_from_vars(self, vars):
            state = np.zeros((self.env.max_layers, self.env.n_materials+1))
            layer_thickness = vars[:self.env.max_layers]
            materials_inds = np.floor(vars[self.env.max_layers:]).astype(int)  
            for i in range(self.env.max_layers):
                #state[i,0] = vars[f"layer_{i}"]
                #state[i,vars[f"layer_{i}_material"]+1] = 1
                state[i,0] = layer_thickness[i]
                state[i,materials_inds[i]+2] = 1
            return state

        def _evaluate(self, X, out, *args, **kwargs):

            state = self.make_state_from_vars(X)
            total_reward, vals, rewards = self.env.compute_reward(state)
            #out["F"] = -rewards["reflectivity"]
            out["F"] = np.column_stack([-rewards["reflectivity"], -rewards["absorption"]])

    coating_problem = CoatingMoo(env)
    
    algorithm = NSGA2(pop_size=2000, )
    #algorithm = GA(
    #    pop_size=1000,
    #    eliminate_duplicates=True,)
    
    #algorithm = MixedVariableGA(pop_size=50, survival=RankAndCrowdingSurvival())
    #algorithm = MixedVariableGA(pop_size=50)


    res = minimize(coating_problem,
                algorithm,
                ('n_gen', 200),
                seed=10,
                save_history=True,
                verbose=True)
    
    all_pop = Population()

    for algorithm in res.history:
        all_pop = Population.merge(all_pop, algorithm.off)

    df = pd.DataFrame(all_pop.get("X"), columns=[f"X{i+1}" for i in range(coating_problem.n_var)])

    df.to_csv(os.path.join(config.get("General", "root_dir"), "population_data.csv"), index=False)

    fig, ax = plt.subplots()
    ax.scatter(1 - 10**(-res.F[:, 0]), 10**(-res.F[:, 1] - 10), s=10, c="red", alpha=0.5)
    #ax.plot(1 - 10**(-res.F[:]), c="red", alpha=0.5)
    ax.set_xlabel("Reflectivity")
    ax.set_ylabel("Absorption")
    fig.savefig(os.path.join(config.get("General", "root_dir"),  f"pareto_front.png"))

    topstacks = []
    if len(res.X.shape) < 2:
        res.X = [res.X, ]

    for i,t_x in enumerate(res.X):
        state = coating_problem.make_state_from_vars(t_x)
        total_reward, vals, rewards = env.compute_reward(state)
        #vals = {"reflectivity": 1 - 10**(-res.F[i, 0]), "absorption": 10**(-res.F[i, 1] - 10), "thermal_noise": 0, "thickness": 0}
        #vals = {"reflectivity": vals["reflectivity"], "absorption": vals["absorption"], "thermal_noise": vals["thermal_noise"], "thickness": 0}
        #rewards = {"total_reward":total_reward}
        fig, ax = plotting.plot_stack(state, env.materials, rewards=rewards, vals=vals)
        fig.savefig(os.path.join(config.get("General", "root_dir"),  f"stack_{i}.png"))
        if i > 10:
            break
    
    plot = Scatter()
    plot.add(coating_problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, facecolor="none", edgecolor="red")
    plot.show()

    """
    filename = os.path.join(config.get("General", "root_dir"),'coating.png')
    scores = []
    max_scores = []
    eps_history = []
    n_steps = 0
    final_state = None
    if not os.path.isdir(os.path.join(config.get("General", "root_dir"), "stackplots")):
        os.makedirs(os.path.join(config.get("General", "root_dir"), "stackplots"))
    
    #print(top_states)
    #print(sorted_state_values[:10])

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
    """

