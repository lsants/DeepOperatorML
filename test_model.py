import time
import torch
import numpy as np
import logging
import sys
import matplotlib.pyplot as plt

from modules.inference.run_inference import run_inference
from modules.plotting import kelvin_plots, dynamic_plots
from modules.utilities import dir_functions

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# ---------------- Load configuration (params_test.yaml) -----------------------
config = dir_functions.load_params('params_test.yaml')

# --------- Run inference (returns predictions, ground truth, ...) -----------
preds, ground_truth, trunk_features, config_model = run_inference(config)

# ------------ Choose plotting function (based on problem type) ---------------
problem_type = config.get("PROBLEM_TYPE", "kelvin").lower()
freq = config.get("FREQ", 1e-3)

if problem_type == "kelvin":
    # Assume we have coordinates stored or computed from trunk features.
    # For example, you might have x, y, z arrays in your config.
    coords = {
        'x': config["x_array"],
        'y': config["y_array"],
        'z': config["z_array"]
    }
    # Assume the first output key in the list is the one to plot.
    output_key = config["OUTPUT_KEYS"][0]
    fig = kelvin_plots.plot_field_comparison_kelvin(coords, ground_truth[output_key], preds[output_key], freq)

elif problem_type == "dynamic_fixed_material":
    coords = (config["r_array"], config["z_array"])
    output_key = config["OUTPUT_KEYS"][0]
    fig = dynamic_plots.plot_field_comparison_dynamic(coords, ground_truth[output_key], freq)
else:
    logger.error("Unknown problem type for plotting.")
    sys.exit(1)

fig.savefig("comparison_plot.png")
plt.show()
