import time
import numpy as np
import scipy.special as sc
from scipy.integrate import quad_vec
from tqdm.auto import tqdm
from .data_generation_base import Datagen
from .influence import influence

class DimensionlessInfluenceFunction(Datagen):
    def __init__(self, data_size, material_params, load_params, mesh_params, problem_setup):
        super().__init__(data_size, material_params, load_params, mesh_params, problem_setup)

    def _gen_branch_data(self):
        N, _, _ = self.data_size
        omega_max, omega_min, _, _, _, r_source = self.load_params
        Es, vs, _, dens = self.material_params
        e1 = Es / (1 + vs)/ (1 - 2*vs)
        c44 = e1*(1-2*vs)/2
        omega = omega_min + np.random.rand(N)*(omega_max - omega_min)
        delta = omega*r_source*np.sqrt(dens/c44)
        return omega, delta
    
    def _gen_trunk_data(self):
        _, n_r, n_z = self.data_size
        _, _, _, _, _, r_source = self.load_params
        r_min, r_max, z_min, z_max = self.mesh_params
        modified_r_min = r_min + (r_source*1e-2) # To avoid computing at line r=0
        r_field = np.linspace(0, r_max, n_r) / r_source
        z_field = np.linspace(z_min, z_max, n_z) / r_source
        points = r_field, z_field
        return points
    
    def _influencefunc(self, freqs, r_field, z_field):
        # ---------- Get parameters ------------
        N, n_r, n_z = self.data_size
        Es, vs, damp, dens = self.material_params
        _, _, _, z_source, l_source, r_source = self.load_params
        component, loadtype, bvptype = self.problem_setup

        # ------------ Material ------------
        e1 = Es/(1 + vs)/(1 - 2*vs)
        c11 = e1*(1 - vs)
        c12 = e1*vs
        c13 = e1*vs
        c33 = e1*(1 - vs)
        c44 = e1*(1 - 2*vs)/2
        
        # ---------- Displacement matrix ------------
        num_freqs = N
        wd = np.zeros((num_freqs, n_r, n_z), dtype=complex)

        # ------- Setting non-dimensional material constants ----------
        c11 = c11 / c44
        c12 = c12 / c44
        c13 = c13 / c44
        c33 = c33 / c44
        c44 = c44 / c44
        dens = dens / dens
        z_source = z_source / r_source
        r_source = r_source / r_source

        ## -------------- Computing displacement ----------------
        start = time.perf_counter_ns()
        for i in tqdm(range(N), colour='Green'):
            for j in range(n_r):
                for k in range(n_z):
                    wd[i, j, k] = influence(
                                    c11, c12, c13, c33, c44,
                                    dens, damp,
                                    r_field[j], z_field[k],
                                    z_source, r_source, l_source,
                                    freqs[i],
                                    bvptype, loadtype, component
                                )

        end = time.perf_counter_ns()
        duration = (end - start) / 1e9
        return wd, duration
    
    def produce_samples(self, filename):
        _, xb = self._gen_branch_data()
        r, z = self._gen_trunk_data()
        displacements, times = self._influencefunc(xb, r, z)

        print(f"Runtime for integration: {times:.2f} s")
        print(f"\nData shapes:\n\t u:\t{xb.shape}\n\t g_u:\t{displacements.shape}\n\t r:\t{r.shape}\n\t z:\t{z.shape}")
        print(f"\nr_min:\t\t\t{r.min()} \nr_max:\t\t\t{r.max()} \nz_min:\t\t\t{z.min()} \nz_max:\t\t\t{z.max()}")

        np.savez(filename, xb=xb, r=r, z=z, g_u=displacements)
        print(f"Saved at {filename}")