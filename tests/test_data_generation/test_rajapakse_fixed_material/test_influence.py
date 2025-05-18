import pytest
from src.modules.problems.rajapakse_fixed_material.influence import influence

def test_influence_function():
    # Test with known values from validation case
    result = influence(
        c11_val=1.0, c12_val=0.3, c13_val=0.3, c33_val=1.0, c44_val=0.35,
        dens_val=1.0, damp_val=0.01,
        r_campo_val=0.5, z_campo_val=-1.0,
        z_fonte_val=0.0, r_fonte_val=0.1, l_fonte_val=0.2,
        freq_val=10.0,
        bvptype_val=3, loadtype_val=1, component_val=2
    )
    
    # Check expected properties
    assert isinstance(result, complex)
    assert abs(result.real) > 1e-6  # Non-zero real part
    assert abs(result.imag) > 1e-6  # Non-zero imaginary part

def test_library_loading():
    """Test that the Pascal library exists in the correct location relative to the generator"""
    from src.modules.problems.rajapakse_fixed_material.generator import RajapakseFixedMaterialGenerator
    import pathlib
    
    generator_path = pathlib.Path(RajapakseFixedMaterialGenerator.__module__.replace(".", "/")).resolve()
    expected_libs_path = generator_path.parent / "libs"
    
    generator = RajapakseFixedMaterialGenerator({
        "DATA_PATH": "dummy.npz",
        "N_R": 1,
        "N_Z": 1
    })
    
    assert generator.libs_path == expected_libs_path, (
        f"Library path mismatch!\n"
        f"Expected: {expected_libs_path}\n"
        f"Actual: {generator.libs_path}"
    )
    
    assert any(generator.libs_path.glob("axsgrsce.*")), \
        f"No Pascal libraries found in {generator.libs_path}"