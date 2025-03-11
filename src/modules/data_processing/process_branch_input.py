from abc import ABC, abstractmethod
import numpy as np

class XbComputationStrategy(ABC):
    """Base class for computing xb.
    """
    @abstractmethod
    def compute_xb(self, input_funcs: list[np.array]) -> np.ndarray:
        pass

class MeshgridStrategy(XbComputationStrategy):
    def compute_xb(self, input_funcs: list[np.ndarray]) -> np.ndarray:
        sensor_mesh = np.meshgrid(*input_funcs, indexing='ij')
        xb = np.column_stack([m.flatten() for m in sensor_mesh])
        return xb if xb.ndim > 1 else xb.reshape(len(xb), -1)
    
class ColumnStackStrategy(XbComputationStrategy):
    def compute_xb(self, input_funcs: list[np.ndarray]) -> np.ndarray:
        return np.column_stack(input_funcs)
    
branch_processing_map = {
    'meshgrid': MeshgridStrategy,
    'stack': ColumnStackStrategy
}