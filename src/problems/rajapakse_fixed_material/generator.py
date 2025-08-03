from __future__ import annotations
import time
import yaml
import logging
import numpy as np
from typing import Any
from pathlib import Path
from tqdm.auto import tqdm
from .influence import influence
from ..base_generator import BaseProblemGenerator

logger = logging.getLogger(__name__)


class RajapakseFixedMaterialGenerator(BaseProblemGenerator):
    def __init__(self, config: str | dict[str, Any]):
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
        """Generate branch input functions by sampling normalized frequencies for a fixed material.

        Returns:
            array: Shape (N, 1) consisting of N samples of the normalized frequency a0.
        """
        e1 = self.config["E"] / (1 + self.config["nu"]) / \
            (1 - 2 * self.config["nu"])
        c44 = e1 * (1 - 2 * self.config["nu"]) / 2
        omega = self.config["omega_min"] + np.random.rand(self.config["N"]) * (
            self.config["omega_max"] - self.config["omega_min"])
        delta = omega * self.config["r_source"] * \
            np.sqrt(self.config["dens"] / c44)
        return delta

    def _get_coordinates(self):
        """Generate the trunk data by defining a 2d grid in cylindrical coordinates.
        Returns:
            tuple: [r, z]
        """
        # To avoid computing at line r=0
        modified_r_min = self.config["r_min"] + \
            (self.config["r_source"] * 1e-2)
        r_field = np.linspace(
            modified_r_min, self.config["r_max"], self.config["N_r"]) / self.config["r_source"]
        z_field = np.linspace(
            self.config["z_min"], self.config["z_max"], self.config["N_z"]) / self.config["r_source"]
        return r_field, z_field

    def _influencefunc(self, norm_freqs, r_field, z_field):
        """
        Computes infuence function in cylindrical coordinates by performing numerical integration.
        Here, the branch input is the normalized frequency (a0), and one solution is computed for each normalized frequency
        on the full (N_r, N_z) grid.

        Args:
            norm_freqs (array): 1D array of non-dimensional frequencies samples.
            r_field (array): 1D array of r coordinates.
            z_field (array): 1D array of z coordinates.

        Returns:
            u (ndarray): Array of shape (N, n_r, n_z, 1) containing the displacement field.
            duration (float): Computation time in seconds.
        """
        # ------------ Material ------------
        e1 = self.config["E"] / (1 + self.config["nu"]) / \
            (1 - 2 * self.config["nu"])
        c11 = e1 * (1 - self.config["nu"])
        c12 = e1 * self.config["nu"]
        c13 = e1 * self.config["nu"]
        c33 = e1 * (1 - self.config["nu"])
        c44 = e1 * (1 - 2 * self.config["nu"]) / 2

        # ---------- Displacement matrix ------------
        n_freqs = self.config["N"]
        wd = np.zeros(
            (n_freqs, self.config["N_r"], self.config["N_z"]), dtype=complex)

        # ------- Setting non-dimensional material constants ----------
        c11 = c11 / c44
        c12 = c12 / c44
        c13 = c13 / c44
        c33 = c33 / c44
        c44 = c44 / c44
        dens = self.config["dens"] / self.config["dens"]
        z_source = self.config["z_source"] / self.config["r_source"]
        r_source = self.config["r_source"] / self.config["r_source"]

        # -------------- Computing displacement ----------------
        start = time.perf_counter_ns()
        for i in tqdm(range(n_freqs), colour='Green'):
            for j in range(self.config["N_r"]):
                for k in range(self.config["N_z"]):
                    wd[i, j, k] = influence(
                        c11_val=c11, c12_val=c12, c13_val=c13, c33_val=c33, c44_val=c44,
                        dens_val=dens, damp_val=self.config["damp"],
                        r_campo_val=r_field[j], z_campo_val=z_field[k],
                        z_fonte_val=z_source, r_fonte_val=r_source, l_fonte_val=self.config[
                            "l_source"],
                        freq_val=norm_freqs[i],
                        bvptype_val=self.config["bvptype"], loadtype_val=self.config[
                            "loadtype"], component_val=self.config["component"]
                    )

        end = time.perf_counter_ns()
        duration = (end - start) / 1e9
        return wd, duration

    def generate(self):
        delta = self._get_input_functions()
        coordinates = self._get_coordinates()
        r_field, z_field = coordinates
        u, duration = self._influencefunc(
            norm_freqs=delta, r_field=r_field, z_field=z_field)

        logger.info(f"Runtime for integration: {duration:.3f} s\nData shapes:\n\t u:\t{delta.shape}\n\t g_u:\t{u.shape}\n\t r:\t{r_field.shape}\n\t z:\t{z_field.shape}\na0_min:\t\t\t{delta.min():.3f} \na0_max:\t\t\t{delta.max():.3f}\nr_min:\t\t\t{r_field.min():.3f} \nr_max:\t\t\t{r_field.max():.3f} \nz_min:\t\t\t{z_field.min():.3f} \nz_max:\t\t\t{z_field.max():.3f}\na0_mean:\t\t\t{delta.mean():.3f} \na0_std:\t\t\t{delta.std():.3f}\nr_mean:\t\t\t{r_field.mean():.3f} \nr_std:\t\t\t{r_field.std():.3f} \nz_mean:\t\t\t{z_field.mean():.3f} \nz_std:\t\t\t{z_field.std():.3f}")

        metadata = {
            "runtime_ms": f"{duration:.3f}",
            "parameters": {
                # .item() for scalar array
                "delta": {
                    "shape": [i for i in delta.shape],
                    "min": f"{delta.min():.3E}",
                    "max": f"{delta.max():.3E}",
                    "mean": f"{delta.mean():.3E}",
                    "std": f"{delta.std():.3E}"
                },
            },
            "coordinate_statistics": {
                "r": {
                    "shape": [i for i in r_field.shape],
                    "min": f"{r_field.min():.3f}",
                    "max": f"{r_field.max():.3f}",
                    "mean": f"{r_field.mean():.3f}",
                    "std": f"{r_field.std():.3f}"
                },
                "z": {
                    "shape": [i for i in z_field.shape],
                    "min": f"{z_field.min():.3f}",
                    "max": f"{z_field.max():.3f}",
                    "mean": f"{z_field.mean():.3f}",
                    "std": f"{z_field.std():.3f}"
                }
            },
            "displacement_statistics": {
                "g_u": {
                    "shape": [i for i in u.shape],
                    "min": f"{u.min():.4E}",
                    "max": f"{u.max():.4E}",
                    "mean": f"{u.mean():.4E}",
                    "std": f"{u.std():.4E}"
                }
            }
        }

        path = Path(self.config["data_filename"])
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, delta=delta, r=r_field, z=z_field, g_u=u)
        metadata_path = path.with_suffix('.yaml')  # Changes .npz to .yaml

        with open(metadata_path, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        logger.info(f"Saved data at {path}")
        logger.info(f"Saved metadata at {metadata_path}")
