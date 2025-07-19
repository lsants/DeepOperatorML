from .generator import RajapakseFixedMaterialGenerator
from .plot_axis import plot_axis
from .plot_field import plot_2D_field
from .plot_basis import plot_basis
from .plot_coeffs import plot_coefficients, plot_coefficients_mean
from . import postprocessing as ppr
# Required interface for auto-registration
PROBLEM_NAME = "rajapakse_fixed_material"
Generator = RajapakseFixedMaterialGenerator
