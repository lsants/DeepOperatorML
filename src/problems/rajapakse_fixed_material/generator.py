from __future__ import annotations
import time
import logging
import yaml
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from ..base_generator import BaseProblemGenerator
from .influence import influence

logger = logging.getLogger(__name__)

class RajapakseFixedMaterialGenerator(BaseProblemGenerator):
    def __init__(self, config: str | dict[str, any]):
        super().__init__(config)
        self.libs_path = Path(__file__).parent / 'libs'
        if isinstance(config, (str, Path)):
            self.config_path = config
            self.config = self.load_config()
        else:
            self.config_path = None
            self.config = config

    def load_config(self):
        if self.config_path:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return self.config

    def _get_input_functions(self):
        e1 = self.config["E"] / (1 + self.config["NU"]) / (1 - 2 * self.config["NU"])
        c44 = e1 * (1 - 2 * self.config["NU"]) / 2
        omega = self.config["OMEGA_MIN"] + np.random.rand(self.config["N"]) * (self.config["OMEGA_MAX"] - self.config["OMEGA_MIN"])
        delta = omega * self.config["R_SOURCE"] * np.sqrt(self.config["DENS"] / c44)
        return delta
    
    def _get_coordinates(self):
        modified_r_min = self.config["R_MIN"] + (self.config["R_SOURCE"] * 1e-2) # To avoid computing at line r=0
        r_field = np.linspace(modified_r_min, self.config["R_MAX"], self.config["N_R"]) / self.config["R_SOURCE"]
        z_field = np.linspace(self.config["Z_MIN"], self.config["Z_MAX"], self.config["N_Z"]) / self.config["R_SOURCE"]
        points = r_field, z_field
        return points
    
    def _influencefunc(self, norm_freqs, r_field, z_field):
        # ------------ Material ------------
        e1 = self.config["E"] / (1 + self.config["NU"]) / (1 - 2 * self.config["NU"])
        c11 = e1 * (1 - self.config["NU"])
        c12 = e1 * self.config["NU"]
        c13 = e1 * self.config["NU"]
        c33 = e1 * (1 - self.config["NU"])
        c44 = e1 * (1 - 2 * self.config["NU"]) / 2
        
        # ---------- Displacement matrix ------------
        n_freqs = self.config["N"]
        wd = np.zeros((n_freqs, self.config["N_R"], self.config["N_Z"]), dtype=complex)

        # ------- Setting non-dimensional material constants ----------
        c11 = c11 / c44
        c12 = c12 / c44
        c13 = c13 / c44
        c33 = c33 / c44
        c44 = c44 / c44
        dens = self.config["DENS"] / self.config["DENS"]
        z_source = self.config["Z_SOURCE"] / self.config["R_SOURCE"]
        r_source = self.config["R_SOURCE"] / self.config["R_SOURCE"]

        ## -------------- Computing displacement ----------------
        start = time.perf_counter_ns()
        for i in tqdm(range(n_freqs), colour='Green'):
            for j in range(self.config["N_R"]):
                for k in range(self.config["N_Z"]):
                    wd[i, j, k] = influence(
                                    c11, c12, c13, c33, c44,
                                    dens, self.config["DAMP"],
                                    r_field[j], z_field[k],
                                    z_source, r_source, self.config["L_SOURCE"],
                                    norm_freqs[i],
                                    self.config["BVPTYPE"], self.config["LOADTYPE"], self.config["COMPONENT"]
                                )

        end = time.perf_counter_ns()
        duration = (end - start) / 1e9
        return wd, duration
    
    def generate(self):
        delta = self._get_input_functions()
        coordinates = self._get_coordinates()
        r, z = coordinates
        displacements, times = self._influencefunc(delta, r, z)

        logger.info(f"Runtime for integration: {times:.2f} s")
        logger.info(f"\nData shapes:\n\t u:\t{delta.shape}\n\t g_u:\t{displacements.shape}\n\t r:\t{r.shape}\n\t z:\t{z.shape}")
        logger.info(f"\na0_min:\t\t\t{delta.min()} \na0_max:\t\t\t{delta.max()}")
        logger.info(f"\nr_min:\t\t\t{r.min()} \nr_max:\t\t\t{r.max()} \nz_min:\t\t\t{z.min()} \nz_max:\t\t\t{z.max()}")

        path = Path(self.config["DATA_PATH"])
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, delta=delta, r=r, z=z, g_u=displacements)
        logger.info(f"Saved at {path}")
