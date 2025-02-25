import os
import sys
import yaml
import numpy as np
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    stream=sys.stdout
)
from modules.data_generation.data_generation_dynamic_fixed_material import DynamicFixedMaterialProblem
from modules.data_generation.data_generation_kelvin import KelvinsProblemDeterministic

logger = logging.getLogger(__name__)


def gen_data(config_path: str):
    parser = argparse.ArgumentParser()

    parser.add_argument("--problem", type=str, help="Generate data for given problem")
    args = parser.parse_args()

    problem = args.problem.lower()

    with open(config_path) as file:
        p = yaml.safe_load(file)

    np.random.seed(p["SEED"])

    filename = os.path.join(f"{p['DATA_FILENAME']}")

    # --------- Grouping parameters -------------

    if problem == "dynamic_fixed_material":
        data_size = (p["N"], 
                     p["N_R"], 
                     p["N_Z"])
        load_params = ((p["OMEGA_MAX"]), # Gotta fix this, should evaluate
                       (p["OMEGA_MIN"]), 
                       (p["LOAD"]), 
                       (p["Z_SOURCE"]), 
                       (p["L_SOURCE"]), 
                       (p["R_SOURCE"]))
        mesh_params = ((p["R_MIN"], 
                        p["R_MAX"], 
                        p["Z_MIN"], 
                        p["Z_MAX"]))
        problem_setup = (p['COMPONENT'], 
                         p['LOADTYPE'], 
                         p['BVPTYPE'])
        material_params = ((p["E"], 
                            p["NU"], 
                            p["DAMP"], 
                            p["DENS"]))

        influence_functions = DynamicFixedMaterialProblem(
            data_size,
            material_params,
            load_params,
            mesh_params,
            problem_setup,
        )
        influence_functions.produce_samples(filename)

    elif problem == "kelvin":
        data_size = (p["N_F_KELVIN"], 
                     p["N_MU_KELVIN"], 
                     p["N_NU_KELVIN"], 
                     p["N_X_KELVIN"], 
                     p["N_Y_KELVIN"], 
                     p["N_Z_KELVIN"])
        material_params = (p["MU_MIN_KELVIN"], 
                           p["MU_MAX_KELVIN"], 
                           p["NU_MIN_KELVIN"], 
                           p["NU_MAX_KELVIN"])
        load_params = ((p["F_MIN_KELVIN"], 
                        p["F_MAX_KELVIN"]))
        mesh_params = ((p["X_MIN_KELVIN"], 
                        p["X_MAX_KELVIN"], 
                        p["Y_MIN_KELVIN"], 
                        p["Y_MAX_KELVIN"], 
                        p["Z_MIN_KELVIN"], 
                        p["Z_MAX_KELVIN"]))
        problem_setup = p["LOAD_DIRECTION_KELVIN"]

        influence_functions = KelvinsProblemDeterministic(
            data_size,
            material_params,
            load_params,
            mesh_params,
            problem_setup
        )
        influence_functions.produce_samples(filename)

    else:
        print("fatal error: not a valid problem.", file=sys.stderr)

if __name__ == "__main__":

    gen_data("./configs/config_data_generation.yaml")