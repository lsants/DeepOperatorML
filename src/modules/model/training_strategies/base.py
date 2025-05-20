from abc import ABC, abstractmethod
from dataclasses import dataclass

from h11 import Data
from .config import StrategyConfig
from ...pipe.pipeline_config import DataConfig, TrainConfig, PhaseConfig


class TrainingStrategy(ABC):
    def __init__(self, strategy_config: StrategyConfig) -> None:
        self.train_cfg = train_cfg
        self.data_cfg = data_cfg
    
    @abstractmethod
    def initialize_components(self):
        """Create branch/trunk using registry"""
    
    @abstractmethod
    def get_phases(self) -> list[PhaseConfig]:
        """Return ordered training phases"""
    
    @abstractmethod
    def preprocess_data(self, data: dict):
        """Strategy-specific data processing"""
    
    @abstractmethod
    def configure_output_handling(self):
        """Modify output handling based on strategy"""