from typing import Optional

import numpy as np
import torch

from coatopt.config.structured_config import CoatingOptimisationConfig
from coatopt.environments.core.state import CoatingState
from coatopt.environments.multiobjective_environment import MultiObjectiveEnvironment


class GeneticCoatingStack(MultiObjectiveEnvironment):
    """
    Genetic algorithm coating environment with dual initialization support.
    """

    def __init__(self, config: Optional[CoatingOptimisationConfig] = None, **kwargs):
        """
        Initialize genetic environment.

        Args:
            config: CoatingOptimisationConfig object (new approach)
            **kwargs: Individual parameters (legacy approach), including:
                     thickness_sigma: Standard deviation for thickness mutations
        """
        # Initialize multi-objective environment
        super().__init__(config, **kwargs)

        # Genetic-specific initialization
        self.thickness_sigma = kwargs.get("thickness_sigma", 1e-4)

    def sample_state_space(
        self,
    ):
        """return air with a thickness of 1

        Returns:
            _type_: _description_
        """

        layers = np.zeros((self.max_layers, self.n_materials + 1))
        reach_end = False
        for i in range(self.max_layers):
            material = np.random.randint(1, self.n_materials)
            if material == self.air_material_index:
                reach_end = True
            if reach_end:
                layers[i, self.air_material_index + 1] = 1
            else:
                layers[i, material + 1] = 1

        layers[:, 0] = np.random.uniform(
            self.min_thickness, self.max_thickness, size=len(layers[:, 0])
        )

        if np.any(layers[:, 0] < 0):
            print(f"state and thickness: sample")
            print(layers)

        return layers

    def sample_action_space(self, current_state):
        """sample from the available state space

        Returns:
            _type_: _description_
        """
        maxind = 0
        for i, current_layer in enumerate(current_state):
            maxind = i
            if current_layer[self.air_material_index] == 1:
                break
        if maxind == 0:
            layer_ind = 0
        else:
            layer_ind = np.random.randint(maxind + 1)

        if self.ignore_air_option:
            new_material = torch.nn.functional.one_hot(
                torch.from_numpy(np.array(np.random.randint(self.n_materials - 1) + 1)),
                num_classes=self.n_materials,
            )
        else:
            new_material = torch.nn.functional.one_hot(
                torch.from_numpy(np.array(np.random.randint(self.n_materials))),
                num_classes=self.n_materials,
            )

        thickness_change = torch.randn(1) * self.thickness_sigma
        new_thickness = current_state[layer_ind, 0] + thickness_change

        while new_thickness < self.min_thickness or new_thickness > self.max_thickness:
            thickness_change = torch.randn(1) * self.thickness_sigma
            new_thickness = current_state[layer_ind, 0] + thickness_change

        if new_thickness < 0:
            print(new_thickness)

        return np.argmax(new_material), new_thickness[0], layer_ind
