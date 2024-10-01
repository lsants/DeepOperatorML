import os
import sys
import yaml
import numpy as np
import argparse
from modules.evaluate_params import multi_eval as evaluate
from data_generation.data_generation_inf_fun import InfluenceFunction

with open('data_generation_params.yaml') as file:
    p = yaml.safe_load(file)

np.random.seed(p["seed"])

filename = os.path.join(p["raw_data_path"], f"{p['data_filename']}.npz")

parser = argparse.ArgumentParser()

parser.add_argument("problem", type=str, help="generate data for given problem")
args = parser.parse_args()

problem = args.problem.lower()

# --------- Grouping parameters -------------
data_size = (p["n"], p["q"])
material_params = evaluate((p["E_max"], p["E_min"], p["nu_max"], p["nu_min"], p["rho_max"], p["rho_min"]))
load_params = evaluate((p["rho_steel"], p["g"], p["h"], p["s1"], p["s2"], p["omega_max"], p["omega_min"]))
points = evaluate((p["r_min"], p["r_max"], p["z_min"], p["z_max"]))
component = p['component']

if problem == "iss":
    consts = evaluate((p["rho_steel"], p["g"], p["h"], p["s1"], p["s2"]))

    influence_functions = InfluenceFunction(
        data_size,
        material_params,
        load_params,
        points,
        component,
        consts,
    )
    influence_functions.produce_samples(filename)

else:
    print("fatal error: not a valid problem.", file=sys.stderr)