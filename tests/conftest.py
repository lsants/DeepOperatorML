# tests/conftest.py
# tests/conftest.py
import sys
import os
import pytest
import tempfile
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

@pytest.fixture(scope="session")
def tmp_data_dir():
    """Session-wide temporary directory for test data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_rajapakse_fixed_config(tmp_data_dir):
    """Fixture for rajapakse material test configuration"""
    return {
        "DATA_PATH": str(tmp_data_dir / "displacements.npz"),
        "SEED": 42,
        "N_R": 1,
        "N_Z": 1 ,
        "R_MIN": 0 ,
        "Z_MIN": 0,
        "R_MAX": 20,
        "Z_MAX": 20,
        "N" : 1,
        "OMEGA_MIN": 0,
        "OMEGA_MAX": 5200,
        "LOAD": 6006663.0,
        "Z_SOURCE": 0,
        "L_SOURCE": 0,
        "R_SOURCE": 2,
        "E": 25.0E+09,
        "NU": 2.42277267E-01,
        "DAMP": 0.01,
        "DENS": 1.73511990E+03,
        "COMPONENT": 1,
        "LOADTYPE": 3,
        "BVPTYPE": 2,
    }

@pytest.fixture
def sample_kelvin_config(tmp_data_dir):
    return {
        "DATA_PATH": str(tmp_data_dir / "displacements.npz"),
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

@pytest.fixture
def mock_influence(monkeypatch):
    """Mock the Pascal library call for fast testing"""
    def mock_influence(*args, **kwargs):
        return complex(0.123, -0.456)  # Deterministic dummy value
    
    monkeypatch.setattr(
        "src.modules.problems.rajapakse_fixed_material.influence.influence",
        mock_influence
    )

@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry before each test"""
    from src.modules.problems import ProblemRegistry
    original = ProblemRegistry._generators.copy()
    yield
    ProblemRegistry._generators = original.copy()