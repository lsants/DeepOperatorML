from __future__ import annotations
import os
import numpy as np
import torch
import yaml
from pathlib import Path
import importlib.util

def run_post_processing(data_outputs: dict[str, torch.Tensor], model_config: dict[str, any]) -> None:
    # Load post-processing script path from config
    script_path = os.path.join('./src/problems', model_config['PROBLEM'])
    
    # Dynamically import and run
    spec = importlib.util.spec_from_file_location(f'postprocesssing', script_path)
    postproc_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(postproc_module)
    postproc_module.process_results(raw_dir, processed_dir)

def run_plotting(model_config: str, processed_dir: str, plots_dir: str) -> None:
    # Load plotting script path from config
    problem_config = yaml.safe_load(Path("config/problems.yaml").read_text())
    script_path = problem_config[model_config]["plotting_script"]
    
    # Dynamically import and run
    spec = importlib.util.spec_from_file_location(f"{model_config['PROBLEM']}_plotter", script_path)
    plot_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot_module)
    plot_module.plot_metrics(processed_dir, plots_dir)