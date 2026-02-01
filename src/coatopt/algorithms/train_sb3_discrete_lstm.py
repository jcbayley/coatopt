#!/usr/bin/env python3
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from coatopt.algorithms.train_sb3_discrete import (
    CoatOptDiscreteGymWrapper,
    DiscreteActionPlottingCallback,
)
from coatopt.utils.configs import load_config
from coatopt.utils.utils import load_materials, evaluate_model
from coatopt.utils.callbacks import EntropyAnnealingCallback

class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """Custom LSTM feature extractor for sequential coating layer processing.

    This processes the flattened observation (max_layers * features_per_layer)
    by reshaping it back into a sequence and running it through an LSTM.

    Args:
        observation_space: Gym observation space
        lstm_hidden_size: Hidden size of LSTM
        lstm_num_layers: Number of LSTM layers
        features_dim: Output dimension 
        max_layers: Maximum number of coating layers
        features_per_layer: Number of features per layer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        features_dim: int = 128,
        max_layers: int = 20,
        features_per_layer: int = None,
    ):
        # Features dim is what we output to the policy/value networks
        super().__init__(observation_space, features_dim)

        self.max_layers = max_layers

        # Calculate features per layer from observation space
        obs_size = observation_space.shape[0]
        if features_per_layer is None:
            features_per_layer = obs_size // max_layers
        self.features_per_layer = features_per_layer

        # LSTM processes sequences of shape (batch, seq_len, input_size)
        self.lstm = nn.LSTM(
            input_size=features_per_layer,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        # Project LSTM output to desired feature dimension
        self.linear = nn.Linear(lstm_hidden_size, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Process observations through LSTM.

        Args:
            observations: Shape (batch_size, max_layers * features_per_layer)

        Returns:
            features: Shape (batch_size, features_dim)
        """
        batch_size = observations.shape[0]

        # Reshape flat observations back to sequence
        # (batch, max_layers * features_per_layer) -> (batch, max_layers, features_per_layer)
        sequences = observations.view(batch_size, self.max_layers, self.features_per_layer)

        # Run through LSTM
        # lstm_out shape: (batch, max_layers, lstm_hidden_size)
        # h_n shape: (num_layers, batch, lstm_hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(sequences)

        # Use the last timestep output
        last_output = lstm_out[:, -1, :]  # (batch, lstm_hidden_size)

        # Project to features_dim
        features = th.relu(self.linear(last_output))  # (batch, features_dim)

        return features


def train(config_path: str):
    """Train MaskablePPO with LSTM feature extractor and ACTION MASKING.

    Args:
        config_path: Path to config INI file

    Returns:
        Trained MaskablePPO model with LSTM
    """
    import configparser

    parser = configparser.ConfigParser()
    parser.read(config_path)

    # [General] section
    save_dir = parser.get('General', 'save_dir')
    materials_path = parser.get('General', 'materials_path')

    # [sb3_discrete_lstm] section
    section = 'sb3_discrete_lstm'
    total_timesteps = parser.getint(section, 'total_timesteps')
    n_thickness_bins = parser.getint(section, 'n_thickness_bins')
    verbose = parser.getint(section, 'verbose')
    target_reflectivity = parser.getfloat(section, 'target_reflectivity')
    target_absorption = parser.getfloat(section, 'target_absorption')
    mask_consecutive_materials = parser.getboolean(section, 'mask_consecutive_materials')
    mask_air_until_min_layers = parser.getboolean(section, 'mask_air_until_min_layers')
    min_layers_before_air = parser.getint(section, 'min_layers_before_air')
    epochs_per_step = parser.getint(section, 'epochs_per_step')
    steps_per_objective = parser.getint(section, 'steps_per_objective')
    tensorboard_log = parser.get(section, 'tensorboard_log')

    # LSTM-specific settings (with defaults)
    lstm_hidden_size = parser.getint(section, 'lstm_hidden_size', fallback=128)
    lstm_num_layers = parser.getint(section, 'lstm_num_layers', fallback=2)
    lstm_features_dim = parser.getint(section, 'lstm_features_dim', fallback=128)

    # [Data] section
    n_layers = parser.getint('Data', 'n_layers')
    constraint_schedule = parser.get('Data', 'constraint_schedule', fallback='interleaved').strip('"').strip("'")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if materials_path is None:
        materials_path = Path(__file__).parent / "config" / "materials.json"

    materials = load_materials(str(materials_path))
    print(f"Loaded {len(materials)} materials from {materials_path}")

    # Load config
    config = load_config(config_path)
    print(f"Loaded config from {config_path}")
    config.data.n_layers = n_layers

    # Create environment (reuse from train_sb3_discrete)
    env = CoatOptDiscreteGymWrapper(
        config,
        materials,
        n_thickness_bins=n_thickness_bins,
        constraint_penalty=10.0,
        target_constraint_bounds={
            "reflectivity": target_reflectivity,
            "absorption": target_absorption,
        },
        mask_consecutive_materials=mask_consecutive_materials,
        mask_air_until_min_layers=mask_air_until_min_layers,
        min_layers_before_air=min_layers_before_air,
        epochs_per_step=epochs_per_step,
        steps_per_objective=steps_per_objective,
        constraint_schedule=constraint_schedule,
    )


    tb_log = None

    # Calculate features per layer
    n_features_per_layer = 1 + env.env.n_materials + 2  # thickness + materials_onehot + n + k

    # Custom policy with LSTM feature extractor
    policy_kwargs = dict(
        features_extractor_class=LSTMFeatureExtractor,
        features_extractor_kwargs=dict(
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            features_dim=lstm_features_dim,
            max_layers=env.env.max_layers,
            features_per_layer=n_features_per_layer,
        ),
        net_arch=dict(
            pi=[64, 64],  # Policy network after LSTM
            vf=[64, 64],  # Value network after LSTM
        ),
    )


    # Create MaskablePPO with LSTM feature extractor
    model = MaskablePPO(
        "MlpPolicy",  # Use MlpPolicy with custom feature extractor
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.15,  # Initial value (will be updated by callback)
        verbose=0,
        tensorboard_log=tb_log,
    )

    # Create entropy annealing callback 
    entropy_callback = EntropyAnnealingCallback(
        max_ent=0.15,  # High exploration at start of each cycle
        min_ent=0.01,  # Low exploration at end of each cycle
        epochs_per_step=epochs_per_step,  # Reset annealing every N episodes
        verbose=0,
    )

    # Set up callbacks
    plotting_callback = DiscreteActionPlottingCallback(
        env=env,
        plot_freq=500,
        design_plot_freq=50,
        save_dir=str(save_dir),
        n_best_designs=5,
        materials=materials,
        verbose=verbose,
    )

    # Train
    # Combine callbacks
    from stable_baselines3.common.callbacks import CallbackList
    callbacks = CallbackList([entropy_callback, plotting_callback])
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    # Save model
    model_path = save_dir / "coatopt_ppo_discrete_lstm"
    model.save(str(model_path))
    print(f"\nModel saved to {model_path}")

    # Final evaluation
    evaluate_model(model, env, n_episodes=10, use_action_masks=True)

    # Save Pareto front
    plotting_callback.save_pareto_front_to_csv("pareto_front_lstm.csv")

    return model



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train SB3 MaskablePPO with LSTM on CoatOpt"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config INI file"
    )

    args = parser.parse_args()
    train(config_path=args.config)
