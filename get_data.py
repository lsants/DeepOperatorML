import os
import sys
import yaml
import numpy as np
import argparse
from modules.evaluate_params import multi_eval as evaluate
from data_generation.data_generation_dimless_green import DimensionlessInfluenceFunction

with open('data_generation_params.yaml') as file:
    p = yaml.safe_load(file)

np.random.seed(p["SEED"])

filename = os.path.join(f"{p['DATA_FILENAME']}")

parser = argparse.ArgumentParser()

parser.add_argument("problem", type=str, help="Generate data for given problem")
args = parser.parse_args()

problem = args.problem.lower()

# --------- Grouping parameters -------------
data_size = (p["N"], p["N_R"], p["N_Z"])
load_params = evaluate((p["OMEGA_MAX"], p["OMEGA_MIN"], p["LOAD"], p["Z_SOURCE"], p["L_SOURCE"], p["R_SOURCE"]))
mesh_params = evaluate((p["R_MIN"], p["R_MAX"], p["Z_MIN"], p["Z_MAX"]))
problem_setup = (p['COMPONENT'], p['LOADTYPE'], p['BVPTYPE'])

if problem == "dimensionless":
    E = (p["E"])
    nu = (p["NU"])
    damp = (p["DAMP"])
    dens = (p["DENS"])
    material_params = evaluate((E, nu, damp, dens))

    influence_functions = DimensionlessInfluenceFunction(
        data_size,
        material_params,
        load_params,
        mesh_params,
        problem_setup,
    )
    influence_functions.produce_samples(filename)

else:
    print("fatal error: not a valid problem.", file=sys.stderr)