"""Shared utility functions for CoatOpt experiments."""

import json
import math
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def load_materials(path: str) -> dict:
    """Load materials from JSON, converting string keys to int.

    Args:
        path: Path to materials JSON file

    Returns:
        Dictionary mapping material indices (int) to material properties (dict)
    """
    with open(path) as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def evaluate_model(model, env, n_episodes: int = 10, use_action_masks: bool = False):
    """Evaluate trained model.

    Args:
        model: Trained SB3 model
        env: Gymnasium environment to evaluate on
        n_episodes: Number of evaluation episodes
        use_action_masks: Whether to use action masking (for MaskablePPO)

    Returns:
        None (prints evaluation results)
    """
    rewards = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        while not done:
            if use_action_masks and hasattr(env, 'action_masks'):
                # MaskablePPO with action masking
                action_masks = env.action_masks()
                action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            else:
                # Standard PPO
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = done or truncated

        rewards.append(episode_reward)
        vals = info.get("vals", {})
        print(
            f"  Episode {ep + 1}: reward={episode_reward:.4f}, "
            f"steps={steps}, vals={vals}"
        )


