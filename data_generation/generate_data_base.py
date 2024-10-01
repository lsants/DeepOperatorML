import json
import numpy as np
from abc import ABC, abstractmethod

class Datagen(ABC):
    def __init__(self, data_size, material_params, load_params, points_range):
        self.data_size = data_size
        self.material_params = material_params
        self.load_params = load_params
        self.points_range = points_range

    @abstractmethod
    def _kernel(self):
        pass

    @abstractmethod
    def _gen_branch_data(self, data_size, material_params, load_params):
        pass

    @abstractmethod
    def _gen_trunk_data(self, data_size, points_range):
        pass

    @abstractmethod
    def _integration(self, branch_features, trunk_features, l_bound=0, u_bound=np.inf):
        pass

    def produce_samples(self, filename):
        xb = self._gen_branch_data(self.data_size, self.material_params, self.load_params)
        xt = self._gen_trunk_data(self.data_size, self.points_range)
        integrals, errors, durations, infos = self._integration(xb, xt)

        # Log sample descriptions
        print(f"Runtime for integration: {durations.mean():.2f} ±  {durations.std():.2f} s")
        print(r"L_inf of " + f"errors: {errors.mean():.2f} ±  {errors.std():.2f}")
        print(f"\nData shapes:\n\t u:\t{xb.shape}\n\t g_u:\t{integrals.shape}\n\t xt:\t{xt.shape}")
        print(f"\nr_min:\t\t\t{xt[:,0].min()} \nr_max:\t\t\t{xt[:,0].max()} \nz_min:\t\t\t{xt[:,1].min()} \nz_max:\t\t\t{xt[:,1].max()}")
        print(infos)
        
        with open('/users/lsantia9/research/high_performance_integration/data/info/integration_info.txt', 'w') as fp:
            fp.write(str(infos))

        np.savez(filename, xb=xb, xt=xt, g_u=integrals)