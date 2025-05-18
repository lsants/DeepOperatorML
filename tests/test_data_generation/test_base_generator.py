import pytest
from src.modules.problems.base_generator import BaseProblemGenerator

def test_base_class_abstract():
    with pytest.raises(TypeError):
        BaseProblemGenerator()  # Can't instantiate abstract class

def test_required_methods():
    assert 'generate' in BaseProblemGenerator.__abstractmethods__
    assert 'load_config' in BaseProblemGenerator.__abstractmethods__