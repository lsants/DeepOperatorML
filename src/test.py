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

def test_model(config_path: str, trained_model_config: dict | None=None) -> None:
    # -------------------- Load params and initialize model ---------------------
    config = dir_functions.load_params(config_path)
    
    if trained_model_config:
        trained_model_folder = trained_model_config["MODEL_FOLDER"]
        trained_model_name = trained_model_config["MODELNAME"]
        trained_model_datafile = trained_model_config["DATAFILE"]
        config["MODEL_FOLDER"] = trained_model_folder
        config["MODELNAME"] = trained_model_name
        config["DATAFILE"] = trained_model_datafile

    model, preds, ground_truth, trunk_features, branch_features, config_model = inference(config)
    
    # -------------------- Initialize saver ---------------------

    if trained_model_config:
        config_model['MODELNAME'] = trained_model_name
        config_model['MODEL_FOLDER'] = trained_model_folder
        config_model["IMAGES_FOLDER"] = trained_model_folder + '/images'

    saver = Saver(
        model_name=config_model['MODELNAME'],
        model_folder=config_model['MODEL_FOLDER'],
        data_output_folder=config_model['MODEL_FOLDER'],
        figures_folder=config_model["IMAGES_FOLDER"]
    )

    saver(errors=config_model.get('ERRORS_PHYSICAL', {}))
    saver(time=config_model.get('INFERENCE_TIME', 0), time_prefix="inference")

    saver.set_logging(False) # Don't print saved paths for plots

    # -------------------- Process data for plots ---------------------

    data_for_2D_plotting = postprocess_for_2D_plot(model=model, 
                                                       plot_config=config, 
                                                       model_config=config_model, 
                                                       branch_features=branch_features, 
                                                       trunk_features=trunk_features,
                                                       ground_truth=ground_truth,
                                                       preds=preds)

    N, d = data_for_2D_plotting["branch_features"].shape

    percentiles = np.linspace(0, 100, 100 // config["PLOT_PERCENTILES"] + 1)
    selected_indices = {}
    for dim in range(d):
        indices = []
        for perc in percentiles:
            target = np.percentile(data_for_2D_plotting["branch_features"][:, dim], perc)
            idx = np.argmin(np.abs(data_for_2D_plotting["branch_features"][:, dim] - target))
            indices.append(idx)
        selected_indices[dim] = indices
        logger.info(f"\nSelected indices for {config_model['INPUT_FUNCTION_KEYS'][dim]}: {indices}\n")
        logger.info(f"\nSelected values for {config_model['INPUT_FUNCTION_KEYS'][dim]}: {data_for_2D_plotting['branch_features'][indices]}\n")

    # --------------------- Plotting fields & axes -----------------------
    if config.get('PLOT_FIELD', False):
        for dim, indices in tqdm(selected_indices.items(), colour='blue'):
            for count, idx in tqdm(enumerate(indices), colour=config['PLOT_FIELD_BAR_COLOR']):
                param_val = tuple(data_for_2D_plotting["branch_features"][idx])
                fig_field = plot_2D_field(
                    coords=data_for_2D_plotting["coords_2D"],
                    truth_field=data_for_2D_plotting["truth_field_2D"][idx],
                    pred_field=data_for_2D_plotting["pred_field_2D"][idx],
                    param_value=param_val,
                    param_labels=config_model.get("INPUT_FUNCTION_KEYS")
                )
                
                val_str = ",".join([f"{i:.2f}" for i in param_val])
                saver(figure=fig_field, figure_prefix=f"{count * 10}_th_perc_field_for_param_{config_model['INPUT_FUNCTION_KEYS'][dim]}=({val_str})")
                plt.close()
                
                # fig_axis = plot_axis(
                #     coords=data_for_2D_plotting["trunk_features"],
                #     truth_field=data_for_2D_plotting["truth_field"][idx],
                #     pred_field=data_for_2D_plotting["pred_field"][idx],
                #     param_value=param_val,
                #     coord_labels=config_model.get("COORD_LABELS")
                # )
                # saver(figure=fig_axis, figure_prefix=f"axis_dim{dim}_for_param_{param_val}")

    # --------------------- Plotting basis -----------------------
    n_basis = min(config_model.get('BASIS_FUNCTIONS'), len(data_for_2D_plotting["basis_functions_2D"]))
    if config.get('PLOT_BASIS', False):
        for i in tqdm(range(1, n_basis + 1), colour=config['PLOT_BASIS_BAR_COLOR']):
            fig_mode = plot_basis_function(data_for_2D_plotting["coords_2D"], 
                                        data_for_2D_plotting["basis_functions_2D"][i - 1],
                                        index=i,
                                        basis_config=config_model['BASIS_CONFIG'],
                                        strategy=config_model['TRAINING_STRATEGY'],
                                        param_val=None,
                                        output_keys=data_for_2D_plotting['output_keys'])
            saver(figure=fig_mode, figure_prefix=f"mode_{i}")
            plt.close()

    logger.info("\n----------------------- Plotting succesfully completed ------------------\n")

if __name__ == "__main__":
    test_model("./configs/config_test.yaml")
