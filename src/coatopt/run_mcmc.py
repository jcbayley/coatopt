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
        thickness_sigma=config.get("MCMC", "thickness_sigma"),
    )


    nwalkers = config.get("MCMC", "n_walkers")
    nsteps = config.get("MCMC", "n_steps")

    initial_params = []
    for i in range(nwalkers):
        t_initial_state = env.sample_state_space()
        t_initial_params = env.convert_state_to_params(t_initial_state) 
        t_initial_params += np.random.normal(0, 0.1, np.shape(t_initial_params))
        initial_params.append(t_initial_params)


    # Set up the sampler.
    ndim = len(initial_params[0])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, env.log_probability)
    if not os.path.isdir(os.path.join(config.get("General", "root_dir"), "states")):
        os.makedirs(os.path.join(config.get("General", "root_dir"), "states"))

    for sample in sampler.sample(initial_params, iterations=nsteps, progress=True):
        if sampler.iteration % (nsteps//20) == 0:
            print(f"Step {sampler.iteration} of {nsteps}")
            ind = np.random.randint(len(sample[0]))
            next_state = env.convert_params_to_state(sample[0][ind])
            state_reward, state_vals, state_rewards = env.compute_reward(next_state)
            fig, ax = env.plot_stack(next_state)
            ax.set_title(f'{state_reward}, R:{state_vals["reflectivity"]}, T:{state_vals["thermal_noise"]}, A:{state_vals["absorption"]}')
            fig.savefig(os.path.join(config.get("General", "root_dir"), "states", f'state_{sampler.iteration}.png'))
    #sampler.run_mcmc(np.array(initial_params), nsteps)

    flat_samples = sampler.get_chain(discard=config.get("MCMC", "n_burnin"), thin=1, flat=True)

    fig = corner.corner(flat_samples)
    fig.savefig(os.path.join(config.get("General", "root_dir"), f'cornerplot.png'))

    # Convert params to state for each sample
    converted_states = [env.convert_params_to_state(params) for params in flat_samples]
    print(converted_states[0])

    # Compute the value of each state
    state_values = [env.compute_reward(state) for state in converted_states]

    # Get the indices of the top 10 state values
    top_10_indices = sorted(range(len(state_values)), key=lambda i: state_values[i][0], reverse=True)[:10]

    # Histogram the state values
    fig, ax = plt.subplots()
    ax.hist(state_values, bins=10)
    ax.set_xlabel('State Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of State Values')

    fig.savefig(os.path.join(config.get("General", "root_dir"), 'state_values.png'))

    # Plot the state of the top 10
    top_10_indices = sorted(range(len(state_values)), key=lambda i: state_values[i][0], reverse=True)[:10]
    top_10_states = [converted_states[i] for i in top_10_indices]
    top_10_values = [state_values[i][0] for i in top_10_indices]
    for i, state in enumerate(top_10_states):
        fig, ax = env.plot_stack(state)
        ax.set_title(f'Top 10 State, value:{top_10_values[i]}')
        fig.savefig(os.path.join(config.get("General", "root_dir"), f'top_10_state_{i}.png'))


    opt_state = env.get_optimal_state()
    opt_value = env.compute_state_value(opt_state)
    fig, ax = env.plot_stack(opt_state)
    ax.set_title(f'Optimal State, value:{opt_value}')
    fig.savefig(os.path.join(config.get("General", "root_dir"), 'optimal_state.png'))