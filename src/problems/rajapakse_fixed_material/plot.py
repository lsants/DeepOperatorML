from __future__ import annotations
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from ...modules.utilities import plot_utils as plu
from ...modules.plotting.plot_axis import plot_axis
from ...modules.plotting.plot_field import plot_2D_field
from ...modules.plotting.plot_basis import plot_basis_function
from ...modules.plotting.plot_coeffs import plot_coefficients
from ...modules.pipe.saving import Saver
import src.problems.rajapakse_fixed_material.problem_dependent_postprocessing as postp

logger = logging.getLogger(__file__)

def plot_metrics(data_for_plot: dict[str, np.ndarray], plot_config: dict[str, any], loaded_model_config: dict[str, any]) -> None:

    saver = Saver()
    N, num_input_functions = data_for_plot["branch_features"].shape
    
    percentiles = np.linspace(0, 100, 100 // plot_config["PLOT_PERCENTILES"] + 1)
    selected_indices = {}
    for input_function in range(num_input_functions):
        indices = []
        for perc in percentiles:
            target = np.percentile(data_for_plot["branch_features"][ : , input_function], perc)
            idx = np.argmin(np.abs(data_for_plot["branch_features"][ : , input_function] - target))
            indices.append(idx)
        selected_indices[input_function] = indices
        logger.debug(f"\nSelected indices for {loaded_model_config['INPUT_FUNCTION_LABELS'][input_function]}: {indices}\n")
        logger.info(f"\nSelected values for {loaded_model_config['INPUT_FUNCTION_LABELS'][input_function]}: {data_for_plot['branch_features'][indices]}\n")

    # --------------------- Plotting fields & axes -----------------------
    
    if plot_config.get('PLOT_FIELD', False):
        for input_function, indices in selected_indices.items():
            for count, idx in tqdm(enumerate(indices), colour=plot_config['PLOT_FIELD_BAR_COLOR']):
                param_val = tuple(data_for_plot["branch_features"][idx])
                fig_field = plot_2D_field(
                    coords=data_for_plot["coords_2D"],
                    truth_field=data_for_plot["truth_field_2D"][idx],
                    pred_field=data_for_plot["pred_field_2D"][idx],
                    param_value=param_val,
                    param_labels=loaded_model_config.get("INPUT_FUNCTION_LABELS"),
                    label_mapping=loaded_model_config["OUTPUT_LABELS"]
                )
                
                val_str = ",".join([f"{i:.2f}" for i in param_val])
                saver(figure=fig_field, figure_prefix=f"{count * 10}_th_perc_field_for_param_{loaded_model_config['INPUT_FUNCTION_LABELS'][input_function]}=({val_str})")
                plt.close()
                
                # fig_axis = plot_axis(
                #     coords=data_for_plot["trunk_features"],
                #     truth_field=data_for_plot["truth_field"][idx],
                #     pred_field=data_for_plot["pred_field"][idx],
                #     param_value=param_val,
                #     coord_labels=loaded_model_config.get("COORD_LABELS")
                #     label_mapping=loaded_model_config["OUTPUT_LABELS"]
                # )
                # saver(figure=fig_axis, figure_prefix=f"axis_input_function{input_function}_for_param_{param_val}")

    # --------------------- Plotting basis -----------------------
    
    n_basis = len(data_for_plot["basis_functions_2D"])
    basis_functions = data_for_plot["basis_functions_2D"]

    basis_to_plot = plot_config["BASIS_TO_PLOT"]
    
    if basis_to_plot == 'all':
        basis_to_plot = n_basis
    else:
        basis_to_plot = min(basis_to_plot, n_basis)

    if plot_config.get('PLOT_BASIS', False):
        for i in tqdm(range(1, basis_to_plot + 1), colour=plot_config['PLOT_BASIS_BAR_COLOR']):
            fig_mode = plot_basis_function(data_for_plot["coords_2D"], 
                                        basis_functions[i - 1],
                                        index=i,
                                        basis_config=loaded_model_config['BASIS_CONFIG'],
                                        strategy=loaded_model_config['TRAINING_STRATEGY'],
                                        param_val=None,
                                        output_keys=data_for_plot['output_keys'],
                                        label_mapping=loaded_model_config["OUTPUT_LABELS"])
            fig_path = os.path.join(loaded_model_config["PLOTS_PATH"], f'basis_mode_{i}.png')
            saver.save_plots(file_path=fig_path, figure=fig_mode)
            plt.close()

    # -------------------- Plotting coefficients ------------------

    coeffs = data_for_plot["coefficients"]
    coeffs_mean = coeffs.mean(axis=0)
    coeffs_mean_abs = np.abs(coeffs_mean)
    basis_functions_flipped = postp.flip_sign_of_negative_modes(basis_functions, coeffs_mean)
    modes_to_highlight = postp.get_modes_indices_to_highlight(coeffs_mean_abs, plot_config["PRINCIPAL_MODES_TO_PLOT"])
    fig_coeffs = plot_coefficients(basis_functions_flipped, coeffs_mean_abs, modes_to_highlight, label_mapping=loaded_model_config["OUTPUT_LABELS"])
    fig_path = os.path.join(loaded_model_config["PLOTS_PATH"], f'branch_coeffs_mean.png')
    saver.save_plots(file_path=fig_path, figure=fig_coeffs)

    logger.info("\n----------------------- Plotting succesfully completed ------------------\n")
    
    basis_save = data_for_plot["basis_functions_2D"]
    coeffs_save = data_for_plot["coefficients"]

    data_for_saving = {
        'basis_functions': basis_save,
        'coefficients': coeffs_save
    }

    saver.save_output_data(data_for_saving)

    logger.info("\n----------------------- Outputs succesfully saved ------------------\n")
