"""
Plugin-based reward system that automatically discovers reward functions.
Makes it easy to add new reward functions without modifying selectors.
"""
import importlib
import inspect
from typing import Dict, Callable, Any
import numpy as np


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
    """Simplified reward calculator with automatic function discovery."""
    
    def __init__(self, reward_type="default", optimise_parameters=None, optimise_targets=None, **kwargs):
        """
        Initialize with basic parameters.
        
        Args:
            reward_type: Name of reward function to use
            optimise_parameters: Parameters to optimize
            optimise_targets: Target values
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
        
        # Get the reward function once at initialization
        self.reward_function = self.registry.get_function(self.reward_type)
    
    def calculate(self, reflectivity, thermal_noise, thickness, absorption, weights=None, **extra_kwargs):
        """Calculate reward using the selected function."""
        # Merge initialization kwargs with call-time kwargs
        call_kwargs = {**self.kwargs, **extra_kwargs}
        

        return self.reward_function(
                reflectivity=reflectivity,
                thermal_noise=thermal_noise,
                total_thickness=thickness,
                absorption=absorption,
                optimise_parameters=self.optimise_parameters,
                optimise_targets=self.optimise_targets,
                weights=weights,
                **call_kwargs
            )


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
                               optimise_parameters, optimise_targets, **kwargs):
    """
    Example of how to add a new reward function.
    Just define it with this decorator and it's automatically available.
    """
    # Simple example: negative sum of all parameters
    return -(reflectivity + thermal_noise + total_thickness + absorption)
