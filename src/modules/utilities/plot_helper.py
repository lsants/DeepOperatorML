import importlib.util
import yaml
from pathlib import Path

def load_and_run_plotting(problem_key: str, results_dir: str, output_dir: str) -> None:
    config_path = Path("config/problems/" + problem_key + "/config_test.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get plotting script path
    script_path = config[problem_key]["plotting_script"]
    
    # Dynamically import the plotting module
    spec = importlib.util.spec_from_file_location(f"{problem_key}_plotter", script_path)
    plot_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot_module)
    
    # Call the standardized function
    plot_module.plot_metrics(results_dir, output_dir)