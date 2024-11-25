import json
import numpy as np
from abc import ABC, abstractmethod

class Datagen(ABC):
    def __init__(self, data_size, material_params, load_params, mesh_params, problem_setup):
        self.data_size = data_size
        self.material_params = material_params
        self.load_params = load_params
        self.mesh_params = mesh_params
        self.problem_setup = problem_setup

    @abstractmethod
    def _gen_branch_data(self):
        pass

    @abstractmethod
    def _gen_trunk_data(self):
        pass

    @abstractmethod
    def _influencefunc(self):
        pass

    @abstractmethod
    def produce_samples(self, filename):
        pass