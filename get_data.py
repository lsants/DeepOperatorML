import os
import sys
import yaml
import numpy as np
import argparse
from data_generation.data_generation_poly import Poly
from data_generation.data_generation_ssi import SSI

with open('data_generation_params.yaml') as file:
    p = yaml.safe_load(file)

np.random.seed(p["seed"])

filename = os.path.join(p["folder"], f"{p['filename']}.npz")

parser = argparse.ArgumentParser()

parser.add_argument("problem", type=str, help="generate data for give problem")
args = parser.parse_args()

problem = args.problem.lower()

if problem == "polynomial":
    polynomials = Poly(params)
    polynomials.produce_samples(params, filename)
elif problem == "ssi":
    influence_functions = SSI(params, filename)
else:
    print("fatal error: not a valid problem.", file=sys.stderr)