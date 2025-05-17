from __future__ import annotations
import logging
from . import inference as inf
from ..data_processing.postprocessing_helper import run_post_processing, run_plotting

logger = logging.getLogger(__name__)

def test_model(test_config, model_to_test_config) -> None:
    
    # -------------------- Load params and initialize model ---------------------
    test_device, test_precision = test_config['DEVICE'], test_config['PRECISION']
    model, data_outputs = inf.inference(model_to_test_config, 
                                    test_device, 
                                    test_precision)

    # ------------------------ Process output data ------------------------------
    processed_data_outputs = run_post_processing(data_outputs, model, model_to_test_config)
    
    # run_plotting(processed_data_outputs, test_config, model_to_test_config["IMAGES_FOLDER"])
