
import argparse
from coatopt.config import read_config, read_materials
import os
from coatopt.environments.thermal_noise_environment_discrete import DiscreteCoatingStack
from coatopt.algorithms.dqn import DQNAgent, DQNTrainer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=False, default="none")
    parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    config = read_config(os.path.abspath(args.config)) 


    materials = read_materials(os.path.join(config.get("General", "materials_file")))

    thickness_options = [0.1, 0.5, 1.0, 1.4, 1.9]

    env = DiscreteCoatingStack(
        config.get("Data", "n_layers"), 
        config.get("Data", "min_thickness"),
        config.get("Data", "max_thickness"),
        materials,
        thickness_options=thickness_options,
        opt_init=False,
        use_intermediate_reward = config.get("Data", "use_intermediate_reward"),
        reward_shape=config.get("Data", "reward_shape"),
        ignore_air_option=config.get("Data", "ignore_air_option"),
        use_ligo_reward=config.get("Data", "use_ligo_reward"),
        use_ligo_thermal_noise=config.get("Data", "use_ligo_thermal_noise"),
    )

    agent = DQNAgent(
            env.obs_space_shape, 
            env.n_materials*len(thickness_options), 
            epsilon_start=1.0,
            epsilon_decay=0.999,
            epsilon_min=0.01,
            buffer_size=10000,
            batch_size=512,
            hidden_size=64,
            learning_rate=1e-3,
            gamma=0.999,
    )


    trainer = DQNTrainer(
        agent, 
        env, 
        config.get("Training", "n_iterations"), 
        config.get("Data", "n_layers"), 
        exploration_decay=0.999, 
        root_dir=config.get("General", "root_dir"),
        batch_size=1024)
    
    optimal_state = env.get_optimal_state()
    optimal_value = env.compute_state_value(optimal_state)
    fig, ax = env.plot_stack(optimal_state)
    ax.set_title(f" opt val: {optimal_value}")
    fig.savefig(os.path.join(config.get("General", "root_dir"),  f"opt_state.png"))

    trainer.train()



