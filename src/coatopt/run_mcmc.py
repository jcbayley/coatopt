import emcee
from coatopt.environments.thermal_noise_environment_mcmc import MCMCCoatingStack
from coatopt.config import read_config, read_materials
import corner
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=False, default="none")
    parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    config = read_config(os.path.abspath(args.config)) 


    materials = read_materials(os.path.join(config.get("General", "materials_file")))


    env = MCMCCoatingStack(
        config.get("Data", "n_layers"), 
        config.get("Data", "min_thickness"),
        config.get("Data", "max_thickness"),
        materials,
        opt_init=False)
    


    nwalkers = config.get("MCMC", "n_walkers")
    nsteps = config.get("MCMC", "n_steps")

    initial_params = []
    for i in range(nwalkers):
        t_initial_state = env.sample_state_space()
        t_initial_params = env.convert_state_to_params(t_initial_state) 
        t_initial_params += np.random.normal(0, 1e-4, np.shape(t_initial_params))
        initial_params.append(t_initial_params)


    # Set up the sampler.
    ndim = len(initial_params[0])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, env.log_probability)

    sampler.run_mcmc(np.array(initial_params), nsteps)

    flat_samples = sampler.get_chain(discard=2000, thin=1, flat=True)

    fig = corner.corner(flat_samples)
    fig.savefig(os.path.join(config.get("General", "root_dir"), f'cornerplot.png'))

    # Convert params to state for each sample
    converted_states = [env.convert_params_to_state(params) for params in flat_samples]
    print(converted_states[0])

    # Compute the value of each state
    state_values = [env.compute_state_value(state) for state in converted_states]

    # Get the indices of the top 10 state values
    top_10_indices = sorted(range(len(state_values)), key=lambda i: state_values[i], reverse=True)[:10]

    # Histogram the state values
    fig, ax = plt.subplots()
    ax.hist(state_values, bins=10)
    ax.set_xlabel('State Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of State Values')

    fig.savefig(os.path.join(config.get("General", "root_dir"), 'state_values.png'))

    # Plot the state of the top 10
    top_10_indices = sorted(range(len(state_values)), key=lambda i: state_values[i], reverse=True)[:10]
    top_10_states = [converted_states[i] for i in top_10_indices]
    top_10_values = [state_values[i] for i in top_10_indices]
    for i, state in enumerate(top_10_states):
        fig, ax = env.plot_stack(state)
        ax.set_title(f'Top 10 State, value:{top_10_values[i]}')
        fig.savefig(os.path.join(config.get("General", "root_dir"), f'top_10_state_{i}.png'))


    opt_state = env.get_optimal_state()
    opt_value = env.compute_state_value(opt_state)
    fig, ax = env.plot_stack(opt_state)
    ax.set_title(f'Optimal State, value:{opt_value}')
    fig.savefig(os.path.join(config.get("General", "root_dir"), 'optimal_state.png'))