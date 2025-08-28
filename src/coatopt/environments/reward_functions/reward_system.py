"""
Plugin-based reward system that automatically discovers reward functions.
Makes it easy to add new reward functions without modifying selectors.
"""
import importlib
import inspect
from typing import Dict, Callable, Any, List, Optional
import numpy as np
from collections import deque


class RewardRegistry:
    """Registry for reward functions using automatic discovery."""
    
    def __init__(self):
        self._functions = {}
        self._discover_reward_functions()
    
    def _discover_reward_functions(self):
        """Automatically discover all reward functions."""
        try:
            # Import the reward function module
            from . import coating_reward_function as reward_module
            
            # Get all functions that start with 'reward_function'
            for name, func in inspect.getmembers(reward_module, inspect.isfunction):
                if name.startswith('reward_function'):
                    # Use the part after 'reward_function_' as the key
                    key = name.replace('reward_function_', '') or 'default'
                    self._functions[key] = func
                    
        except ImportError:
            print("Warning: Could not import coating_reward_function module")
    
    def register(self, name: str, func: Callable):
        """Manually register a reward function."""
        self._functions[name] = func
    
    def get_function(self, name: str) -> Callable:
        """Get a reward function by name."""
        if name not in self._functions:
            raise ValueError(f"Reward function '{name}' not found. Available: {list(self._functions.keys())}")
        return self._functions[name]
    
    def list_functions(self) -> list:
        """List all available reward functions."""
        return list(self._functions.keys())


class RewardCalculator:
    """Simplified reward calculator with automatic function discovery and optional reward normalization."""
    
    def __init__(self, reward_type="default", optimise_parameters=None, optimise_targets=None, 
                 use_reward_normalization=False, reward_normalization_mode="fixed",
                 reward_normalization_ranges=None, reward_normalization_alpha=0.1,
                 reward_history_size=1000, **kwargs):
        """
        Initialize with basic parameters.
        
        Args:
            reward_type: Name of reward function to use
            optimise_parameters: Parameters to optimize
            optimise_targets: Target values
            use_reward_normalization: Whether to normalize individual rewards before weighting
            reward_normalization_mode: "fixed" (use provided ranges) or "adaptive" (learn from history)
            reward_normalization_ranges: Dict mapping parameter names to [min, max] ranges for normalization
            reward_normalization_alpha: Learning rate for adaptive range updates
            reward_history_size: Number of recent rewards to keep for adaptive normalization
            **kwargs: Additional parameters passed to reward function
        """
        self.registry = RewardRegistry()
        self.reward_type = reward_type
        self.optimise_parameters = optimise_parameters or ["reflectivity", "thermal_noise", "absorption", "thickness"]
        self.optimise_targets = optimise_targets or {
            "reflectivity": 0.99999,
            "thermal_noise": 5.394480540642821e-21,
            "absorption": 0.01,
            "thickness": 0.1
        }
        self.kwargs = kwargs
        
        # Reward normalization setup
        self.use_reward_normalization = use_reward_normalization
        self.reward_normalization_mode = reward_normalization_mode
        self.reward_normalization_ranges = reward_normalization_ranges or {}
        self.reward_normalization_alpha = reward_normalization_alpha
        
        # History tracking for adaptive normalization
        self.reward_history_size = reward_history_size
        self.reward_history = {param: deque(maxlen=reward_history_size) for param in self.optimise_parameters}
        self.adaptive_ranges = {param: {"min": float('inf'), "max": float('-inf')} for param in self.optimise_parameters}
        
        # Get the reward function once at initialization
        self.reward_function = self.registry.get_function(self.reward_type)
    
    def _compute_reward_ranges_from_objective_bounds(self, env) -> Dict[str, List[float]]:
        """
        Compute expected reward ranges by evaluating the reward function at objective bounds.
        
        Args:
            env: Environment instance with objective_bounds attribute
            
        Returns:
            Dict mapping parameter names to [min_reward, max_reward] ranges
        """
        if not hasattr(env, 'objective_bounds'):
            return {}
            
        bounds = env.objective_bounds
        if not bounds:
            return {}
            
        ranges = {}
        
        # Dummy values for parameters not being tested
        dummy_values = {
            'reflectivity': 0.999,
            'absorption': 1e-6,
            'thermal_noise': 1e-20,
            'thickness': 5.0,
        }
        
        # For each parameter we want to optimize
        for param in self.optimise_parameters:
            if param not in bounds:
                continue
                
            min_bound, max_bound = bounds[param]
            reward_values = []
            
            # Test at min and max bounds only (rewards are monotonic)
            for test_value in [min_bound, max_bound]:
                # Set up test parameters
                test_params = dummy_values.copy()
                test_params[param] = test_value
                
                try:
                    # Call the reward function
                    _, _, rewards = self.reward_function(
                        reflectivity=test_params['reflectivity'],
                        thermal_noise=test_params['thermal_noise'], 
                        total_thickness=test_params['thickness'],
                        absorption=test_params['absorption'],
                        optimise_parameters=[param],
                        optimise_targets={param: self.optimise_targets.get(param, test_value)},
                        weights={param: 1.0},
                        env=env,
                        **self.kwargs
                    )
                    
                    if param in rewards:
                        reward_values.append(rewards[param])
                        
                except Exception as e:
                    print(f"Warning: Could not evaluate reward function for {param} at {test_value}: {e}")
                    continue
            
            # Set range based on min/max rewards
            if reward_values:
                ranges[param] = [min(reward_values), max(reward_values)]
                
        return ranges
    
    def _normalize_rewards(self, individual_rewards: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize individual rewards to [0, 1] range.
        
        Args:
            individual_rewards: Dict mapping parameter names to their individual reward values
            
        Returns:
            Dict mapping parameter names to normalized rewards [0, 1]
        """
        if not self.use_reward_normalization:
            return individual_rewards
            
        normalized_rewards = {}
        
        for param, reward in individual_rewards.items():
            if param not in self.optimise_parameters:
                normalized_rewards[param] = reward
                continue
                
            # Update history for adaptive mode
            self.reward_history[param].append(reward)
            
            # Get normalization range
            if self.reward_normalization_mode == "fixed" and param in self.reward_normalization_ranges:
                min_val, max_val = self.reward_normalization_ranges[param]
            elif self.reward_normalization_mode == "adaptive" and len(self.reward_history[param]) > 10:
                # Use moving average for adaptive range
                history = list(self.reward_history[param])
                current_min, current_max = min(history), max(history)
                
                # Update adaptive range with exponential moving average
                if self.adaptive_ranges[param]["min"] == float('inf'):
                    self.adaptive_ranges[param]["min"] = current_min
                    self.adaptive_ranges[param]["max"] = current_max
                else:
                    self.adaptive_ranges[param]["min"] = (1 - self.reward_normalization_alpha) * self.adaptive_ranges[param]["min"] + self.reward_normalization_alpha * current_min
                    self.adaptive_ranges[param]["max"] = (1 - self.reward_normalization_alpha) * self.adaptive_ranges[param]["max"] + self.reward_normalization_alpha * current_max
                
                min_val = self.adaptive_ranges[param]["min"]
                max_val = self.adaptive_ranges[param]["max"]
            else:
                # No normalization available yet
                normalized_rewards[param] = reward
                continue
            
            # Normalize to [0, 1]
            if max_val > min_val:
                normalized_rewards[param] = (reward - min_val) / (max_val - min_val)
            else:
                normalized_rewards[param] = reward
                
        return normalized_rewards

    def calculate(self, reflectivity, thermal_noise, thickness, absorption, env=None, weights=None, **extra_kwargs):
        """Calculate reward using the selected function with optional normalization."""
        # Merge initialization kwargs with call-time kwargs
        call_kwargs = {**self.kwargs, **extra_kwargs}
        
        # Always try to pass env if available - most functions can use it
        if env is not None:
            call_kwargs['env'] = env
            
            # Auto-compute reward ranges from objective bounds if not provided
            if self.use_reward_normalization and not self.reward_normalization_ranges:
                computed_ranges = self._compute_reward_ranges_from_objective_bounds(env)
                if computed_ranges:
                    self.reward_normalization_ranges = computed_ranges
                    print(f"Auto-computed reward normalization ranges: {computed_ranges}")

        # Get the original reward calculation
        total_reward, vals, rewards = self.reward_function(
            reflectivity=reflectivity,
            thermal_noise=thermal_noise,
            total_thickness=thickness,
            absorption=absorption,
            optimise_parameters=self.optimise_parameters,
            optimise_targets=self.optimise_targets,
            weights=weights,
            **call_kwargs
        )
        
        # Apply normalization if enabled and weights are provided
        if self.use_reward_normalization and weights is not None:
            # Extract individual rewards from the rewards dict
            individual_rewards = {param: rewards.get(param, 0.0) for param in self.optimise_parameters}
            
            # Normalize individual rewards
            normalized_rewards = self._normalize_rewards(individual_rewards)
            
            # Recompute total reward using normalized values and weights
            total_reward = sum(weights.get(param, 0.0) * normalized_rewards.get(param, 0.0) 
                             for param in self.optimise_parameters)
            
            # Update rewards dict with normalized values (optional - for debugging/logging)
            for param in self.optimise_parameters:
                rewards[f"{param}_normalized"] = normalized_rewards[param]
        
        return total_reward, vals, rewards


# Decorator for easy reward function registration
def reward_function_plugin(name: str = None):
    """Decorator to register reward functions."""
    def decorator(func):
        registry = RewardRegistry()
        plugin_name = name or func.__name__.replace('reward_function_', '') or 'default'
        registry.register(plugin_name, func)
        return func
    return decorator


# Example of how to add new reward functions easily:
@reward_function_plugin("example_new")
def reward_function_example_new(reflectivity, thermal_noise, total_thickness, absorption, 
                               optimise_parameters, optimise_targets, env=None, **kwargs):
    """
    Example of how to add a new reward function.
    Just define it with this decorator and it's automatically available.
    All reward functions should accept env as a parameter to access environment state.
    """
    # Simple example: negative sum of all parameters
    return -(reflectivity + thermal_noise + total_thickness + absorption)
