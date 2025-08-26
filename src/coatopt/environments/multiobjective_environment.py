"""
Pareto optimization coating environment.
Extends the base CoatingStack with multi-objective optimization capabilities.
"""
from typing import Optional, TYPE_CHECKING
from .hppo_environment import HPPOEnvironment
import numpy as np
import copy

from ..config.structured_config import CoatingOptimisationConfig
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

class MultiObjectiveEnvironment(HPPOEnvironment):
    """
    Coating environment with Pareto multi-objective optimization.
    This extends the base functionality to support multi-objective optimization.
    """
    
    def __init__(self, config: Optional[CoatingOptimisationConfig] = None, **kwargs):
        """Initialize Pareto environment."""
        super().__init__(config, **kwargs)
        
        # Enable multi-objective optimization
        self.multi_objective = True
        self.pareto_objectives = ["reflectivity", "thermal_noise", "absorption"]

    def setup_multiobjective_specific_attributes(self, **kwargs):
        """Setup multi-objective specific attributes."""
        # Enable multi-objective optimization
        self.multi_objective = True
        self.pareto_objectives = ["reflectivity", "thermal_noise", "absorption"]
        self.pareto_front = []
        self.all_points = []
        self.all_vals = []
    
    def compute_pareto_front(self, points):        
        """
        Compute the Pareto front from a set of points.

        Parameters:
            points (numpy.ndarray): Array of points (shape: [n_points, n_dimensions]).

        Returns:
            numpy.ndarray: Pareto front points.
        """
        # Perform non-dominated sorting
        nds = NonDominatedSorting()
        fronts = nds.do(points) 
    
        # Extract the Pareto front (the first front)
        pareto_front = points[fronts[0]]
    
        return pareto_front

    def update_pareto_front(self, pareto_front, new_point):
        """
        Update the Pareto front with a new point and check if it updates the front.

        Parameters:
            pareto_front (numpy.ndarray or list): Current Pareto front points (shape: [n_points, n_dimensions]).
            new_point (numpy.ndarray): New point to be added (shape: [n_dimensions]).

        Returns:
            numpy.ndarray: Updated Pareto front.
            bool: Whether the Pareto front was updated or not.
        """
        # Handle empty Pareto front case
        if len(pareto_front) == 0 or (isinstance(pareto_front, np.ndarray) and pareto_front.size == 0):
            # Warning when Pareto front is empty - this might indicate missing data from previous training
            print("WARNING: Pareto front is empty. This may occur when:")
            print("  - Starting fresh training (normal)")
            print("  - Continuing training but previous Pareto front wasn't loaded (check save/load logic)")
            
            # If pareto front is empty, the new point becomes the first point
            new_point_array = np.array([new_point]) if new_point.ndim == 1 else new_point
            updated_pareto_front = self.compute_pareto_front(new_point_array)
            return updated_pareto_front, True
        
        # Convert pareto_front to numpy array if it's a list
        if isinstance(pareto_front, list):
            pareto_front = np.array(pareto_front)
        
        # Ensure new_point is properly shaped (1D -> 2D)
        if new_point.ndim == 1:
            new_point = new_point.reshape(1, -1)
        
        # Combine the current Pareto front with the new point
        combined_points = np.vstack([pareto_front, new_point])

        updated_pareto_front = self.compute_pareto_front(combined_points)

        # Check if the Pareto front was updated
        pareto_updated = not np.array_equal(updated_pareto_front, pareto_front)

        return updated_pareto_front, pareto_updated
    
    def compute_reward(self, new_state, max_value=0.0, target_reflectivity=1.0, objective_weights=None):
        """reward is the improvement of the state over the previous one

        Args:
            state (_type_): _description_
            action (_type_): _description_
        """

        new_reflectivity, new_thermal_noise, new_E_integrated, new_total_thickness = self.compute_state_value(new_state, return_separate=True)
        
        if objective_weights is not None:
            weights = {
                key:objective_weights[i] for i,key in enumerate(self.optimise_parameters)
            }
        else:
            weights=None

        total_reward, vals, rewards = self.reward_calculator.calculate(
                reflectivity=new_reflectivity,
                thermal_noise=new_thermal_noise,
                thickness=new_total_thickness,
                absorption=new_E_integrated,
                weights=weights
            )
        
        new_point = np.zeros((len(self.optimise_parameters),))

        i = 0
        for key in self.optimise_parameters:
            new_point[i] = rewards[key]
            i += 1

        updated_pareto_front, front_updated = self.update_pareto_front(copy.copy(self.pareto_front), copy.copy(new_point))

        rewards["updated_pareto_front"] = updated_pareto_front
        rewards["front_updated"] = front_updated

        return total_reward, vals, rewards
    
    def step(self, action, max_state=0, verbose=False, state=None, layer_index=None, always_return_value=False, objective_weights=None):
        """action[0] - material, action[1] - thickness"""
        
        # Initialize state and layer index
        if state is None:
            state = self.current_state
        else:
            self.current_state = state

        if layer_index is None:
            layer_index = self.current_index
        else:
            self.current_index = layer_index

        # Extract action parameters
        material = action[0]
        thickness = action[1]
        
        # Ensure state is numpy array or float, not tensor
        state = self.check_numpy_cpu(state)
        material = int(self.check_numpy_cpu(material))
        thickness = self.check_numpy_cpu(thickness)

        new_state = self.update_state(np.copy(state), thickness, material)
        full_action = None
        # Initialize default values
        neg_reward = -1000
        reward = neg_reward
        terminated = False
        finished = False
        
        rewards = {
            "reflectivity": 0, "thermal_noise": 0, "thickness": 0, 
            "absorption": 0, "total_reward": 0
        }
        vals = {
            "reflectivity": 0, "thermal_noise": 0, "thickness": 0, "absorption": 0
        }

        # Update current state
        self.current_state = new_state

        # Check termination conditions
        if self.min_thickness > thickness or thickness > self.max_thickness or not np.isfinite(thickness):
            print("out of thickness bounds")
        elif self.current_index == self.max_layers-1 or material == self.air_material_index:
            # Episode finished
            finished = True
            reward, vals, rewards = self.compute_reward(new_state, max_state, objective_weights=objective_weights)
        elif self.use_intermediate_reward:
            # Intermediate reward calculation
            reward, vals, rewards = self.compute_reward(new_state, max_state, objective_weights=objective_weights)

        # Check for invalid states
        if (np.any(np.isinf(new_state)) or np.any(np.isnan(new_state)) or 
            np.isnan(reward) or np.isinf(reward)):
            reward = neg_reward
            terminated = True

        # Update Pareto front if episode finished and front was updated
        if finished and rewards.get("front_updated", False):
            self.pareto_front = rewards["updated_pareto_front"]

        # Update tracking variables
        self.previous_material = material
        self.length += 1
        self.current_index += 1

        return new_state, rewards, terminated, finished, reward, full_action, vals
