
import pytest
import yaml
import numpy as np
from pathlib import Path
from src.modules.problems.kelvin.generator import KelvinProblemGenerator

@pytest.fixture
def sample_kelvin_config(tmp_path):
    return {
        "DATA_PATH": str(tmp_path / "displacements.npz"),
        "SEED": 42,
        "N": 500,
        "F_MIN": 6,
        "F_MAX": 8,
        "MU_MIN": 6,
        "MU_MAX": 10.48,
        "NU_MIN": 0.2,
        "NU_MAX": 0.45,
        "N_X": 30,
        "N_Y": 30,
        "N_Z": 30,
        "X_MIN": 0.01,
        "X_MAX": 1,
        "Y_MIN": 0.01,
        "Y_MAX": 1,
        "Z_MIN": 0.1,
        "Z_MAX": 1,
        "LOAD_DIRECTION": 'z'
    }

def test_generator_initialization(sample_kelvin_config, tmp_path):
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(sample_kelvin_config, f)
    
    generator = KelvinProblemGenerator(config_file)
    assert generator.config == sample_kelvin_config

def test_data_generation(sample_kelvin_config, tmp_path):
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(sample_kelvin_config, f)
    
    generator = KelvinProblemGenerator(config_file)
    generator.generate()
    
    output_file = Path(sample_kelvin_config["DATA_PATH"])
    assert output_file.exists()
    
    data = np.load(output_file)
    assert 'F' in data
    assert 'mu' in data
    assert 'nu' in data
    assert 'x' in data
    assert 'y' in data
    assert 'z' in data
    assert 'g_u' in data
    assert data['g_u'].shape == (500, 30, 30, 30, 3)

def test_mocked_generation(sample_kelvin_config, mock_influence):
    generator = KelvinProblemGenerator(sample_kelvin_config)
    generator.generate()  # Uses mocked influence function