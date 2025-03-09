import logging
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from .modules.pipe.saving import Saver
from .modules.utilities import dir_functions
from .modules.pipe.inference import inference
from .modules.plotting.plot_axis import plot_axis
from .modules.plotting.plot_field import plot_2D_field
from .modules.plotting.plot_basis import plot_basis_function
from .modules.plotting.plot_utils import postprocess_for_2D_plot

logger = logging.getLogger(__name__)

def test_model(test_config_path: str, trained_model_config: dict | None=None) -> None:
    
    # -------------------- Load params and initialize model ---------------------
    
    test_config = dir_functions.load_params(test_config_path)
    
    if trained_model_config:
        test_config["MODEL_FOLDER_TO_LOAD"] = trained_model_config["MODEL_FOLDER_TO_SAVE"]
        test_config["MODEL_NAME"] = trained_model_config["MODEL_NAME"]
        test_config["DATA_FILE"] = trained_model_config["DATA_FILE"]
        test_config["OUTPUT_FOLDER"] = test_config["MODEL_FOLDER_TO_LOAD"]
        test_config["IMAGES_FOLDER"] = test_config["OUTPUT_FOLDER"] + '/images'

    model, preds, ground_truth, trunk_features, branch_features, loaded_model_config = inference(test_config)
    
    # -------------------- Initialize saver ---------------------

    saver = Saver(
        model_name=test_config["MODEL_NAME"],
        model_folder=test_config["MODEL_FOLDER_TO_LOAD"],
        data_output_folder=test_config['OUTPUT_FOLDER'],
        figures_folder=test_config["IMAGES_FOLDER"]
    )

    saver(errors=loaded_model_config.get('ERRORS_PHYSICAL', {}))
    saver(time=loaded_model_config.get('INFERENCE_TIME', 0), time_prefix="inference")

    saver.set_logging(False) # Don't print saved paths for plots

    # -------------------- Process data for plots ---------------------

    
    data_for_2D_plotting = postprocess_for_2D_plot(model=model, 
                                                   plot_config=test_config, 
                                                   model_config=loaded_model_config, 
                                                   branch_features=branch_features, 
                                                   trunk_features=trunk_features,
                                                   ground_truth=ground_truth,
                                                   preds=preds)

    N, dims = data_for_2D_plotting["branch_features"].shape
    
    percentiles = np.linspace(0, 100, 100 // test_config["PLOT_PERCENTILES"] + 1)
    selected_indices = {}
    for dim in range(dims):
        indices = []
        for perc in percentiles:
            target = np.percentile(data_for_2D_plotting["branch_features"][ : , dim], perc)
            idx = np.argmin(np.abs(data_for_2D_plotting["branch_features"][ : , dim] - target))
            indices.append(idx)
        selected_indices[dim] = indices
        logger.info(f"\nSelected indices for {loaded_model_config['INPUT_FUNCTION_KEYS'][dim]}: {indices}\n")
        logger.info(f"\nSelected values for {loaded_model_config['INPUT_FUNCTION_KEYS'][dim]}: {data_for_2D_plotting['branch_features'][indices]}\n")

    # --------------------- Plotting fields & axes -----------------------
    
    if test_config.get('PLOT_FIELD', False):
        for dim, indices in tqdm(selected_indices.items(), colour='blue'):
            for count, idx in tqdm(enumerate(indices), colour=test_config['PLOT_FIELD_BAR_COLOR']):
                param_val = tuple(data_for_2D_plotting["branch_features"][idx])
                fig_field = plot_2D_field(
                    coords=data_for_2D_plotting["coords_2D"],
                    truth_field=data_for_2D_plotting["truth_field_2D"][idx],
                    pred_field=data_for_2D_plotting["pred_field_2D"][idx],
                    param_value=param_val,
                    param_labels=loaded_model_config.get("INPUT_FUNCTION_KEYS")
                )
                
                val_str = ",".join([f"{i:.2f}" for i in param_val])
                saver(figure=fig_field, figure_prefix=f"{count * 10}_th_perc_field_for_param_{loaded_model_config['INPUT_FUNCTION_KEYS'][dim]}=({val_str})")
                plt.close()
                
                # fig_axis = plot_axis(
                #     coords=data_for_2D_plotting["trunk_features"],
                #     truth_field=data_for_2D_plotting["truth_field"][idx],
                #     pred_field=data_for_2D_plotting["pred_field"][idx],
                #     param_value=param_val,
                #     coord_labels=loaded_model_config.get("COORD_LABELS")
                # )
                # saver(figure=fig_axis, figure_prefix=f"axis_dim{dim}_for_param_{param_val}")

    # --------------------- Plotting basis -----------------------
    
    n_basis = len(data_for_2D_plotting["basis_functions_2D"])
    
    basis_to_plot = test_config["BASIS_TO_PLOT"]
    
    if basis_to_plot == 'all':
        basis_to_plot = n_basis
    else:
        basis_to_plot = min(basis_to_plot, n_basis)

    if test_config.get('PLOT_BASIS', False):
        for i in tqdm(range(1, basis_to_plot + 1), colour=test_config['PLOT_BASIS_BAR_COLOR']):
            fig_mode = plot_basis_function(data_for_2D_plotting["coords_2D"], 
                                        data_for_2D_plotting["basis_functions_2D"][i - 1],
                                        index=i,
                                        basis_config=loaded_model_config['BASIS_CONFIG'],
                                        strategy=loaded_model_config['TRAINING_STRATEGY'],
                                        param_val=None,
                                        output_keys=data_for_2D_plotting['output_keys'])
            saver(figure=fig_mode, figure_prefix=f"mode_{i}")
            plt.close()

    logger.info("\n----------------------- Plotting succesfully completed ------------------\n")
    
    basis_save = data_for_2D_plotting["basis_functions_2D"]
    coeffs_save = data_for_2D_plotting["coefficients"]

    data_for_saving = {
        'basis_functions': basis_save,
        'coefficients': coeffs_save
    }

    saver.save_output_data(data_for_saving)

    logger.info("\n----------------------- Outputs succesfully saved ------------------\n")

if __name__ == "__main__":
    test_model("./configs/inference/config_test.yaml")
