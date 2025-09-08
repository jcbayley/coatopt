import numpy as np
from typing import Optional, List, Tuple, Dict
import warnings


class ObjectiveRangeTracker:
    """
    Tracks objective ranges during individual exploration phase for later constraint application.
    """
    
    def __init__(self, n_objectives: int):
        self.n_objectives = n_objectives
        self.objective_ranges = {}  # Dict[objective_idx] -> {'min': float, 'max': float, 'values': List[float]}
        self.is_tracking = True
        
    def update(self, objective_values: Dict[str, float], objective_names: List[str], epoch: int, 
               individual_phase_episodes: int):
        """
        Update objective ranges during individual exploration phase.
        
        Args:
            objective_values: Current objective values (e.g., {'reflectivity': 0.999, 'thermal_noise': 1e-21})
            objective_names: List of objective names in order
            epoch: Current epoch
            individual_phase_episodes: Total episodes for individual exploration
        """
        # Only track during individual exploration phase
        if epoch >= individual_phase_episodes:
            self.is_tracking = False
            return
            
        # Determine which objective is currently being optimized
        episodes_per_objective = individual_phase_episodes // self.n_objectives
        current_objective_idx = (epoch // episodes_per_objective) % self.n_objectives
        
        if current_objective_idx < len(objective_names):
            current_objective_name = objective_names[current_objective_idx]
            
            if current_objective_name in objective_values:
                value = objective_values[current_objective_name]
                
                # Initialize if first time seeing this objective
                if current_objective_idx not in self.objective_ranges:
                    self.objective_ranges[current_objective_idx] = {
                        'min': value,
                        'max': value,
                        'values': [],
                        'name': current_objective_name
                    }
                
                # Update range
                range_info = self.objective_ranges[current_objective_idx]
                range_info['min'] = min(range_info['min'], value)
                range_info['max'] = max(range_info['max'], value)
                range_info['values'].append(value)
    
    def get_constraint_bounds(self, objective_idx: int, constraint_fraction: float = 1.0) -> Optional[Tuple[float, float]]:
        """
        Get constraint bounds for an objective based on learned ranges.
        
        Args:
            objective_idx: Index of the objective
            constraint_fraction: Fraction of the learned range to use as constraint bounds (1.0 = full range)
            
        Returns:
            Tuple of (min_bound, max_bound) or None if objective not tracked
        """
        if objective_idx not in self.objective_ranges:
            return None
            
        range_info = self.objective_ranges[objective_idx]
        range_span = range_info['max'] - range_info['min']
        
        # Add some buffer and use fraction of range
        buffer = 0.05 * range_span  # 5% buffer
        constraint_span = constraint_fraction * range_span
        
        min_bound = range_info['min'] - buffer
        max_bound = range_info['min'] + constraint_span + buffer
        
        return (min_bound, max_bound)
    
    def get_all_ranges(self) -> Dict[int, Dict[str, float]]:
        """Get all learned ranges for debugging/logging."""
        return {idx: {'min': info['min'], 'max': info['max'], 'name': info['name']} 
                for idx, info in self.objective_ranges.items()}


def get_objective_exploration_phase(epoch: int, n_objectives: int, episodes_per_objective: int = 100) -> dict:
    """
    Determine if we're in individual objective exploration phase.
    
    Args:
        epoch: Current episode
        n_objectives: Number of objectives  
        episodes_per_objective: How many episodes to spend on each objective alone
        
    Returns:
        dict with 'phase' ('individual' or 'combination') and 'target_objective' if individual
    """
    total_individual_episodes = n_objectives * episodes_per_objective
    
    if epoch < total_individual_episodes:
        # We're in individual exploration phase
        target_objective = (epoch // episodes_per_objective) % n_objectives
        return {'phase': 'individual', 'target_objective': target_objective}
    else:
        # We're in combination exploration phase
        return {'phase': 'combination'}


def get_preference_constrained_phase(epoch: int, n_objectives: int, episodes_per_objective: int = 1000, 
                                   episodes_per_constrained_phase: int = 2000) -> dict:
    """
    Determine preference constrained training phase.
    
    Phase 1: Learn each objective individually for n_objectives * episodes_per_objective episodes
    Phase 2: For each objective pair, constrain one while optimizing the other
    
    Args:
        epoch: Current episode
        n_objectives: Number of objectives
        episodes_per_objective: Episodes to spend learning each objective individually
        episodes_per_constrained_phase: Episodes to spend on each constrained optimization pair
        
    Returns:
        dict with phase info including constraints and active objective
    """
    total_individual_episodes = n_objectives * episodes_per_objective
    
    if epoch < total_individual_episodes:
        # Phase 1: Individual objective exploration
        target_objective = (epoch // episodes_per_objective) % n_objectives
        return {
            'phase': 'individual',
            'target_objective': target_objective,
            'constrained_objectives': [],
            'active_objective': target_objective
        }
    else:
        # Phase 2: Constrained optimization - each objective gets optimized while others are constrained
        epoch_in_phase2 = epoch - total_individual_episodes
        
        # Calculate which objective is being optimized and which are constrained
        active_objective = (epoch_in_phase2 // episodes_per_constrained_phase) % n_objectives
        constrained_objectives = [i for i in range(n_objectives) if i != active_objective]
        
        return {
            'phase': 'constrained',
            'target_objective': None,
            'constrained_objectives': constrained_objectives,
            'active_objective': active_objective,
            'epoch_in_constrained_phase': epoch_in_phase2 % episodes_per_constrained_phase
        }


def analyze_objective_space_coverage(all_rewards: Optional[List[np.ndarray]], n_objectives: int) -> Dict[str, np.ndarray]:
    """
    Analyze the coverage of the objective space to identify under-explored regions.
    
    Args:
        all_rewards: All reward vectors seen so far
        n_objectives: Number of objectives
        
    Returns:
        Dictionary with analysis results including objective ranges, densities, etc.
    """
    if not all_rewards or len(all_rewards) == 0:
        return {
            'objective_ranges': np.array([[0, 1] for _ in range(n_objectives)]),
            'objective_densities': np.ones(n_objectives),
            'under_explored_regions': np.ones(n_objectives) / n_objectives
        }
    
    rewards_array = np.array(all_rewards)
    
    # Calculate objective ranges and normalize
    obj_mins = np.min(rewards_array, axis=0)
    obj_maxs = np.max(rewards_array, axis=0)
    obj_ranges = obj_maxs - obj_mins
    
    # Avoid division by zero
    obj_ranges = np.where(obj_ranges == 0, 1, obj_ranges)
    normalized_rewards = (rewards_array - obj_mins) / obj_ranges
    
    # Analyze density in different regions for each objective
    n_bins = 5  # Divide each objective into 5 regions
    objective_densities = np.zeros((n_objectives, n_bins))
    
    for obj_idx in range(n_objectives):
        obj_values = normalized_rewards[:, obj_idx]
        hist, _ = np.histogram(obj_values, bins=n_bins, range=(0, 1))
        objective_densities[obj_idx] = hist
    
    # Find under-explored regions (regions with low density)
    total_samples = len(rewards_array)
    expected_per_bin = total_samples / (n_bins * n_objectives)
    
    under_explored_weights = np.zeros(n_objectives)
    for obj_idx in range(n_objectives):
        # Find the least explored bin for this objective
        min_density_bin = np.argmin(objective_densities[obj_idx])
        exploration_deficit = max(0, expected_per_bin - objective_densities[obj_idx][min_density_bin])
        
        # Weight this objective based on exploration deficit
        under_explored_weights[obj_idx] = 1 + exploration_deficit / expected_per_bin
    
    # Normalize weights
    under_explored_weights = under_explored_weights / np.sum(under_explored_weights)
    
    return {
        'objective_ranges': np.column_stack([obj_mins, obj_maxs]),
        'objective_densities': objective_densities,
        'under_explored_regions': under_explored_weights,
        'normalized_rewards': normalized_rewards
    }


def adaptive_weight_distribution(
    n_objectives: int, 
    pareto_front: Optional[np.ndarray] = None,
    exploration_factor: float = 0.3,
    archive_memory: Optional[List[np.ndarray]] = None,
    all_rewards: Optional[List[np.ndarray]] = None,
    epoch: int = 0
) -> np.ndarray:
    """
    Generate adaptive weight distribution based on objective space coverage analysis.
    
    This improved approach:
    1. Starts with random exploration to discover the space structure
    2. Gradually focuses on under-explored objective regions
    3. Uses actual reward distributions rather than trying to map Pareto points to weights
    
    Args:
        n_objectives: Number of objectives
        pareto_front: Current Pareto front points (used for basic stats)
        exploration_factor: Factor controlling exploration vs exploitation (0-1)
        archive_memory: List of recently used weight vectors to avoid
        all_rewards: All reward vectors seen so far (key for coverage analysis)
        epoch: Current training epoch for progressive exploration
        
    Returns:
        Weight vector targeting under-explored regions
    """
    # Early exploration phase: use random exploration with some corner emphasis
    if all_rewards is None or len(all_rewards) < 50 or exploration_factor > 0.7:
        return _early_exploration_weights(n_objectives, archive_memory, exploration_factor)
    
    # Analyze objective space coverage
    coverage_analysis = analyze_objective_space_coverage(all_rewards, n_objectives)
    under_explored_regions = coverage_analysis['under_explored_regions']
    
    # Decide exploration strategy based on training progress
    strategy_prob = np.random.rand()
    
    if strategy_prob < exploration_factor:
        # Focus on under-explored objective regions
        weight = _generate_targeted_weights(under_explored_regions, n_objectives)
    elif strategy_prob < exploration_factor + 0.3:
        # Random exploration with bias towards under-explored regions  
        # Use under-explored regions as Dirichlet alpha parameters
        alpha = 1.0 + 2.0 * under_explored_regions  # Bias towards under-explored
        weight = np.random.dirichlet(alpha)
    else:
        # Pure random exploration
        weight = np.random.dirichlet(np.ones(n_objectives))
    
    # Apply archive-based diversity if available
    if archive_memory and len(archive_memory) > 0:
        weight = _apply_archive_diversity(weight, archive_memory, n_objectives)
    
    return weight


def _early_exploration_weights(n_objectives: int, archive_memory: Optional[List[np.ndarray]], 
                             exploration_factor: float) -> np.ndarray:
    """Generate weights for early exploration phase."""
    # Balance between corner exploration and random exploration
    if np.random.rand() < 0.4:  # 40% corner weights for extreme solutions
        corner_idx = np.random.randint(n_objectives)
        weight = np.zeros(n_objectives)
        weight[corner_idx] = 0.8 + 0.2 * np.random.rand()  # Slightly noisy corners
        remaining = 1 - weight[corner_idx]
        other_indices = [i for i in range(n_objectives) if i != corner_idx]
        if other_indices:
            other_weights = np.random.dirichlet(np.ones(len(other_indices)))
            for i, idx in enumerate(other_indices):
                weight[idx] = remaining * other_weights[i]
        return weight
    else:  # 60% random exploration
        alpha = 0.5 + exploration_factor  # More diverse when exploration_factor is high
        weight = np.random.dirichlet(alpha * np.ones(n_objectives))
        
        # Apply archive diversity if available
        if archive_memory and len(archive_memory) > 0:
            weight = _apply_archive_diversity(weight, archive_memory, n_objectives)
        
        return weight


def _generate_targeted_weights(under_explored_regions: np.ndarray, n_objectives: int) -> np.ndarray:
    """Generate weights that target under-explored objective regions."""
    # Create weights that emphasize under-explored objectives
    base_weight = under_explored_regions.copy()
    
    # Add some randomness but keep the bias
    noise_strength = 0.3
    noise = np.random.normal(0, noise_strength, n_objectives)
    weight = base_weight + noise
    
    # Ensure positive weights
    weight = np.maximum(weight, 0.01)  # Minimum weight to avoid zeros
    
    # Normalize
    weight = weight / np.sum(weight)
    
    return weight


def _apply_archive_diversity(weight: np.ndarray, archive_memory: List[np.ndarray], 
                           n_objectives: int, max_attempts: int = 10) -> np.ndarray:
    """Apply diversity constraint to avoid recently used weights."""
    min_distance_threshold = 0.15  # Minimum L2 distance from archived weights
    
    for attempt in range(max_attempts):
        # Check distance from all archived weights
        min_distance = min([np.linalg.norm(weight - archived) for archived in archive_memory])
        
        if min_distance > min_distance_threshold:
            return weight  # Good diversity, use this weight
        
        # Too close to archived weights, modify it
        # Add noise proportional to how close we are
        noise_strength = max(0.1, (min_distance_threshold - min_distance) * 2)
        noise = np.random.normal(0, noise_strength, n_objectives)
        weight = weight + noise
        
        # Ensure positive and normalize
        weight = np.maximum(weight, 0.01)
        weight = weight / np.sum(weight)
    
    # If we couldn't find a diverse weight after max_attempts, return it anyway
    return weight


class WeightArchive:
    """
    Archive for tracking recently used weight vectors to promote diversity.
    """
    
    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self.weights: List[np.ndarray] = []
        self.timestamps: List[int] = []
        self.current_time = 0
    
    def add_weight(self, weight: np.ndarray):
        """Add a weight vector to the archive."""
        self.weights.append(weight.copy())
        self.timestamps.append(self.current_time)
        self.current_time += 1
        
        # Remove old weights if archive is full
        if len(self.weights) > self.max_size:
            self.weights.pop(0)
            self.timestamps.pop(0)
    
    def get_recent_weights(self, n_recent: int = 10) -> List[np.ndarray]:
        """Get the most recently used weight vectors."""
        return self.weights[-n_recent:] if len(self.weights) >= n_recent else self.weights.copy()
    
    def clear(self):
        """Clear the archive."""
        self.weights.clear()
        self.timestamps.clear()
        self.current_time = 0


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
                        start_weight_alpha=1.0, final_weight_alpha=1.0, random_anneal=True, transfer_fraction=0.25):
    """
    Generate smooth cyclic weights for N objectives.
    
    Args:
        n_objectives: Number of objectives
        t: Current time step
        T_cycle: Total steps for one full cycle
        T_hold: Number of steps to hold each one-hot vector (overridden by transfer_fraction)
        transfer_fraction: Fraction of cycle spent transitioning between objectives (default 0.25)
        total_steps: Total steps in training
        start_weight_alpha: Starting alpha for annealing
        final_weight_alpha: Final alpha for annealing
        random_anneal: Whether to use random annealing after cycles
        
    Returns:
        Weight vector for current time step
    """
    weights = np.zeros(n_objectives)
    phase_steps = T_cycle // n_objectives
    T_transition = int(transfer_fraction * phase_steps)
    T_hold = phase_steps - T_transition

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
                         final_weight_alpha=1.0, n_weight_cycles=2, 
                         pareto_front=None, weight_archive=None, all_rewards=None,
                         transfer_fraction=0.75, env=None):
    """
    Sample weights for the reward function based on cycling strategy.
    
    Args:
        n_objectives: Number of objectives
        cycle_weights: Type of weight cycling:
            - "step": Alternating step weights (2 objectives only)
            - "smooth": Smooth cycling between objectives
            - "annealed_random": Random weights with annealing
            - "adaptive_pareto": Adaptive based on Pareto front gaps
            - "individual_then_adaptive": First explore each objective individually, then adaptive combinations
            - "preference_constrained": Learn each objective individually, then constrained optimization
            - "random": Pure random weights
            - "linear": Linear grid cycling
        epoch: Current epoch (used for scheduling)
        num_samples: Number of weight samples to generate
        final_weight_epoch: Final epoch for weight scheduling
        start_weight_alpha: Starting alpha for Dirichlet distribution
        final_weight_alpha: Final alpha for Dirichlet distribution
        n_weight_cycles: Number of weight cycles
        pareto_front: Current Pareto front for adaptive strategies (optional)
        weight_archive: Archive of recently used weights for diversity (optional)
        all_rewards: All reward vectors seen so far, used for objective space coverage analysis (optional)
        transfer_fraction: Fraction of cycle spent at extremes (for smooth cycling)
        env: Environment object (for getting preference constraints configuration)
        
    Returns:
        Weight vector for reward function
    """
    
    if cycle_weights == "individual_then_adaptive":
        # Two-phase strategy: explore each objective individually, then learn combinations
        phase_info = get_objective_exploration_phase(epoch or 0, n_objectives, episodes_per_objective=1000)
        
        if phase_info['phase'] == 'individual':
            # Phase 1: Focus 100% on one objective at a time
            weights = np.zeros(n_objectives)
            weights[phase_info['target_objective']] = 1.0
        else:
            # Phase 2: Now that we know each objective's potential, use adaptive exploration
            if epoch is not None and final_weight_epoch > 0:
                # Adjust progress calculation to account for individual exploration phase
                individual_phase_episodes = n_objectives * 100
                remaining_epochs = final_weight_epoch - individual_phase_episodes
                if remaining_epochs > 0:
                    progress = min((epoch - individual_phase_episodes) / remaining_epochs, 1.0)
                else:
                    progress = 1.0
                exploration_factor = 0.8 * (1 - progress) + 0.2
            else:
                exploration_factor = 0.5
                
            recent_weights = None
            if weight_archive is not None:
                recent_weights = weight_archive.get_recent_weights()
            
            weights = adaptive_weight_distribution(
                n_objectives=n_objectives,
                pareto_front=pareto_front,
                exploration_factor=exploration_factor,
                archive_memory=recent_weights,
                all_rewards=all_rewards,
                epoch=epoch or 0
            )
            
            if weight_archive is not None:
                weight_archive.add_weight(weights)
    
    elif cycle_weights == "preference_constrained":
        # Two-phase strategy: learn each objective individually, then constrained optimization
        # Get parameters from environment if available
        if env is not None:
            episodes_per_objective = getattr(env, 'preference_constraint_episodes_per_objective', 1000)
            episodes_per_constrained_phase = getattr(env, 'preference_constraint_episodes_per_phase', 2000)
        else:
            episodes_per_objective = 1000
            episodes_per_constrained_phase = 2000
            
        phase_info = get_preference_constrained_phase(
            epoch or 0, n_objectives, 
            episodes_per_objective=episodes_per_objective,
            episodes_per_constrained_phase=episodes_per_constrained_phase
        )
        
        if phase_info['phase'] == 'individual':
            # Phase 1: Focus 100% on one objective at a time
            weights = np.zeros(n_objectives)
            weights[phase_info['target_objective']] = 1.0
        else:
            # Phase 2: Constrained optimization - focus on active objective
            weights = np.zeros(n_objectives)
            weights[phase_info['active_objective']] = 1.0
            
            # Note: Constraints will be handled in the reward system
            # The weight vector just indicates which objective to optimize
    
    elif cycle_weights == "step":
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
                         final_weight_epoch=1, start_weight_alpha=1.0, 
                         final_weight_alpha=1.0, n_weight_cycles=2, 
                         pareto_front=None, weight_archive=None, all_rewards=None,
                         transfer_fraction=0.25)
        else:
            # Default uniform weights when no epoch provided
            weights = np.ones(n_objectives) / n_objectives
            
    elif cycle_weights == "smooth":
        T_cycle = final_weight_epoch // n_weight_cycles
        phase_steps = T_cycle // n_objectives
        T_hold = int(1-transfer_fraction * phase_steps)
        weights = smooth_cycle_weights(
            n_objectives, epoch, T_cycle=T_cycle, T_hold=T_hold, 
            total_steps=final_weight_epoch, start_weight_alpha=start_weight_alpha,
            final_weight_alpha=final_weight_alpha,
            transfer_fraction=transfer_fraction
        )
        
    elif cycle_weights == "annealed_random":
        if epoch is not None:
            weights = annealed_dirichlet_weights(
                n_objectives, epoch, final_weight_epoch, 
                base_alpha=start_weight_alpha, 
                final_alpha=final_weight_alpha, 
                num_samples=1)
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
        T_cycle = final_weight_epoch // n_weight_cycles
        phase_steps = T_cycle // n_objectives
        T_transition = int(transfer_fraction * phase_steps)
        T_hold = phase_steps - T_transition
        weights = smooth_cycle_weights(
            n_objectives, epoch, T_cycle=T_cycle, T_hold=T_hold, 
            total_steps=final_weight_epoch, start_weight_alpha=start_weight_alpha,
            final_weight_alpha=final_weight_alpha,
            transfer_fraction=transfer_fraction
        )

    
    elif cycle_weights == "adaptive_pareto":
        # Improved adaptive weight sampling based on objective space coverage
        # Get exploration factor based on training progress
        if epoch is not None and final_weight_epoch > 0:
            progress = min(epoch / final_weight_epoch, 1.0)
            # Start with high exploration, gradually reduce
            exploration_factor = 0.8 * (1 - progress) + 0.2  # Range: 0.8 -> 0.2
        else:
            exploration_factor = 0.5  # Default middle exploration
        
        # Get recent weights for diversity
        recent_weights = None
        if weight_archive is not None:
            recent_weights = weight_archive.get_recent_weights()
        
        # Generate adaptive weight using improved approach
        weights = adaptive_weight_distribution(
            n_objectives=n_objectives,
            pareto_front=pareto_front,
            exploration_factor=exploration_factor,
            archive_memory=recent_weights,
            all_rewards=all_rewards,  # Key improvement: pass all rewards for coverage analysis
            epoch=epoch or 0
        )
        
        # Add to archive for future diversity
        if weight_archive is not None:
            weight_archive.add_weight(weights)
        
    else:
        raise ValueError(f"Unknown cycle_weights type: {cycle_weights}")
    
    return weights
