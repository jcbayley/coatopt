
import argparse
from coatopt.config import read_config, read_materials
import os
from coatopt.environments.thermal_noise_environment import CoatingStack
from coatopt.algorithms.pdqn2 import PDQNAgent, PDQNTrainer
import numpy as np

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
    )

    agent = PDQNAgent(
            np.prod(env.obs_space_shape), 
            env.n_materials,
            epsilon_initial=1.0,
            epsilon_final=0.01,
            epsilon_decay=5000,
            batch_size=256,
            gamma=0.999,
            lr_critic=0.001,
            lr_actor=0.001,
            lr_alpha=0.001,
            tau_critic=0.005,
            tau_actor=0.005,
            critic_hidden_layers=(32, 32, 32),
            actor_hidden_layers=(32, 32, 32),
            min_thickness=config.get("Data", "min_thickness"),
            max_thickness=config.get("Data", "max_thickness"),
    )

    trainer = PDQNTrainer(
        agent, 
        env, 
        config.get("Training", "n_iterations"), 
        config.get("Data", "n_layers"), 
        exploration_decay=0.999, 
        root_dir=config.get("General", "root_dir"),
        batch_size=1024)

    trainer.train()



