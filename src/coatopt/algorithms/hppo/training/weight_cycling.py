"""
Weight cycling utilities for multi-objective optimization.
Simple functions for handling weight sampling and cycling strategies.
"""
import numpy as np


def annealed_dirichlet_weights(n_objectives, epoch, total_epochs, base_alpha=0.05, final_alpha=1.0, num_samples=1):
    """
    Sample preference weights from an annealed Dirichlet distribution.
    
    Args:
        n_objectives: Number of objectives
        epoch: Current training epoch
        total_epochs: Total number of epochs
        base_alpha: Initial concentration (low -> extreme weights)
        final_alpha: Final concentration (higher -> uniform weights)
        num_samples: Number of weight vectors to sample
    
    Returns:
        Array of sampled weight vectors summing to 1
    """
    # Annealing factor: linear schedule
    progress = np.min([epoch / total_epochs, 1])
    alpha = base_alpha + (final_alpha - base_alpha) * progress

    # Dirichlet concentration vector
    concentration = [alpha] * n_objectives
    
    weights = np.random.dirichlet(concentration, size=num_samples)
    
    # Replace rows with NaN or Inf with random valid weights
    mask = np.isnan(weights).any(axis=1) | np.isinf(weights).any(axis=1)
    if mask.any():
        n_invalid = mask.sum()
        # Generate random valid weights for invalid entries
        replacement_weights = np.random.dirichlet([1.0] * n_objectives, size=n_invalid)
        weights[mask] = replacement_weights
        
    return weights


def smooth_cycle_weights(n_objectives, t, T_cycle, T_hold, total_steps, 
                        start_weight_alpha=1.0, final_weight_alpha=1.0, random_anneal=True):
    """
    Generate smooth cyclic weights for N objectives.
    
    Args:
        n_objectives: Number of objectives
        t: Current time step
        T_cycle: Total steps for one full cycle
        T_hold: Number of steps to hold each one-hot vector
        total_steps: Total steps in training
        start_weight_alpha: Starting alpha for annealing
        final_weight_alpha: Final alpha for annealing
        random_anneal: Whether to use random annealing after cycles
        
    Returns:
        Weight vector for current time step
    """
    weights = np.zeros(n_objectives)
    phase_steps = T_cycle // n_objectives
    T_transition = phase_steps - T_hold

    if t < total_steps:
        cycle_pos = t % T_cycle
        class_idx = (cycle_pos // phase_steps) % n_objectives
        pos_in_phase = cycle_pos % phase_steps

        if pos_in_phase < T_hold:
            # Hold one-hot
            weights[class_idx] = 1.0
        else:
            # Transition between class_idx and next class
            next_idx = (class_idx + 1) % n_objectives
            alpha = (pos_in_phase - T_hold) / T_transition
            weights[class_idx] = 1.0 - alpha
            weights[next_idx] = alpha

    if t >= total_steps:
        if random_anneal:
            weights = annealed_dirichlet_weights(
                n_objectives, total_steps, total_steps, 
                base_alpha=start_weight_alpha, 
                final_alpha=final_weight_alpha, 
                num_samples=1
            )[0]
        else:
            weights = np.ones(n_objectives) / n_objectives

    return weights


def sample_reward_weights(n_objectives, cycle_weights, epoch=None, num_samples=1, 
                         final_weight_epoch=1, start_weight_alpha=1.0, 
                         final_weight_alpha=1.0, n_weight_cycles=2):
    """
    Sample weights for the reward function based on cycling strategy.
    
    Args:
        n_objectives: Number of objectives
        cycle_weights: Type of weight cycling ("step", "smooth", "annealed_random", etc.)
        epoch: Current epoch (used for scheduling)
        num_samples: Number of weight samples to generate
        final_weight_epoch: Final epoch for weight scheduling
        start_weight_alpha: Starting alpha for Dirichlet distribution
        final_weight_alpha: Final alpha for Dirichlet distribution
        n_weight_cycles: Number of weight cycles
        
    Returns:
        Weight vector for reward function
    """
    
    if cycle_weights == "step":
        if n_objectives != 2:
            import warnings
            warnings.warn(f"'step' cycle_weights is designed for 2 objectives but {n_objectives} were provided. "
                         f"Consider using 'smooth', 'annealed_random', or 'random' for multi-objective problems.")
            # Fallback to random weights for non-2-objective cases
            weights = np.random.dirichlet(alpha=np.ones(n_objectives))
        elif epoch is not None:
            # Original 2-objective behavior
            progress = np.min([epoch / final_weight_epoch, 1])
            if progress < 0.25:
                weights = np.array([1, 0])
            elif progress < 0.5:
                weights = np.array([0, 1])
            elif progress < 0.75:
                weights = np.array([1, 0])
            elif progress < 1.0:
                weights = np.array([0, 1])
            else:
                weights = annealed_dirichlet_weights(
                    n_objectives, epoch, 2 * final_weight_epoch, 
                    base_alpha=start_weight_alpha, 
                    final_alpha=final_weight_alpha, 
                    num_samples=1
                )[0]
        else:
            # Default uniform weights when no epoch provided
            weights = np.ones(n_objectives) / n_objectives
            
    elif cycle_weights == "smooth":
        T_hold = int(0.75 * final_weight_epoch / (n_objectives * n_weight_cycles))
        T_cycle = final_weight_epoch // n_weight_cycles
        weights = smooth_cycle_weights(
            n_objectives, epoch, T_cycle=T_cycle, T_hold=T_hold, 
            total_steps=final_weight_epoch, start_weight_alpha=start_weight_alpha,
            final_weight_alpha=final_weight_alpha
        )
        
    elif cycle_weights == "annealed_random":
        if epoch is not None:
            weights = annealed_dirichlet_weights(
                n_objectives, epoch, final_weight_epoch, 
                base_alpha=start_weight_alpha, 
                final_alpha=final_weight_alpha, 
                num_samples=1
            )[0]
        else:
            weights = np.random.dirichlet(alpha=np.ones(n_objectives))
            
    elif cycle_weights == "random":
        weights = np.random.dirichlet(alpha=np.ones(n_objectives))
        
    elif cycle_weights == "linear":
        # Create an N-dimensional grid of weights
        steps = 5  # Number of steps per dimension
        grid_axes = [np.linspace(0, 1, steps) for _ in range(n_objectives)]
        mesh = np.meshgrid(*grid_axes)
        flat = [m.flatten() for m in mesh]
        weight_grid = np.stack(flat, axis=-1)
        
        # Only keep weights that sum to 1 (within tolerance)
        weight_grid = weight_grid[np.isclose(weight_grid.sum(axis=1), 1.0)]
        
        # Iterate through the grid
        if len(weight_grid) > 0:
            index = (epoch // n_weight_cycles) % len(weight_grid)
            weights = weight_grid[index]
        else:
            weights = np.ones(n_objectives) / n_objectives
        
    else:
        raise ValueError(f"Unknown cycle_weights type: {cycle_weights}")
    
    return weights
