import shutil
from pathlib import Path
import pytest
import importlib
import tempfile
from unittest.mock import MagicMock

def test_problem_registration(tmp_path, monkeypatch):
    """Test problem registration by creating temp implementation"""
    # Create temporary problem INSIDE the package structure
    from src.modules.problems import __file__ as pkg_file
    pkg_dir = Path(pkg_file).parent
    
    # Create test problem directory
    test_problem_dir = pkg_dir / "test_problem"
    test_problem_dir.mkdir()
    
    # Add __init__.py with registration
    (test_problem_dir / "__init__.py").write_text("""
from .generator import TestGenerator
PROBLEM_NAME = "test_problem"
Generator = TestGenerator
""")
    
    (test_problem_dir / "generator.py").write_text("""
class TestGenerator:
    def __init__(self, config):
        self.config = config
""")

    # Force auto-discovery
    from src.modules.problems import ProblemRegistry
    ProblemRegistry.auto_discover()
    
    assert "test_problem" in ProblemRegistry._generators
    
    # Cleanup
    shutil.rmtree(test_problem_dir)


def test_ignore_invalid_modules(tmp_path):
    """Test that modules without required attributes are ignored"""
    # Create invalid problem module
    invalid_dir = tmp_path / "invalid_problem"
    invalid_dir.mkdir()
    (invalid_dir / "__init__.py").write_text("")  # No required attributes
    
    from src.modules.problems import ProblemRegistry
    original_count = len(ProblemRegistry._generators)
    
    # Force re-discovery
    ProblemRegistry.auto_discover()
    assert len(ProblemRegistry._generators) == original_count
def test_get_generator_flow():
    """Test full generator retrieval workflow"""
    from src.modules.problems import ProblemRegistry
    
    # Using known existing problem
    config = {"DATA_PATH": "test.npz", "N": 1}
    generator = ProblemRegistry.get_generator("rajapakse_fixed_material", config)
    
    assert generator is not None
    assert hasattr(generator, "generate")
    assert generator.config == config

def test_unknown_problem():
    from src.modules.problems import ProblemRegistry
    
    with pytest.raises(ValueError) as excinfo:
        ProblemRegistry.get_generator("non_existent_problem", {})
    
    assert "No generator for problem 'non_existent_problem'" in str(excinfo.value)