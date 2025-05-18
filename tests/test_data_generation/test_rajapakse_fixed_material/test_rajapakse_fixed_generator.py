
import pytest
import yaml
import numpy as np
from pathlib import Path
from src.modules.problems.rajapakse_fixed_material.generator import RajapakseFixedMaterialGenerator


def test_generator_initialization(sample_rajapakse_fixed_config, tmp_path):
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(sample_rajapakse_fixed_config, f)

    generator = RajapakseFixedMaterialGenerator(config_file)
    assert generator.config == sample_rajapakse_fixed_config


def test_data_generation(sample_rajapakse_fixed_config, tmp_path):
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(sample_rajapakse_fixed_config, f)

    generator = RajapakseFixedMaterialGenerator(config_file)
    generator.generate()

    output_file = Path(sample_rajapakse_fixed_config["DATA_PATH"])
    assert output_file.exists()

    data = np.load(output_file)
    assert 'delta' in data
    assert 'r' in data
    assert 'z' in data
    assert 'g_u' in data
    assert data['g_u'].shape == (1, 1, 1)


def test_mocked_generation(sample_rajapakse_fixed_config, mock_influence):
    generator = RajapakseFixedMaterialGenerator(sample_rajapakse_fixed_config)
    generator.generate()  # Uses mocked influence function
