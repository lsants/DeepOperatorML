import logging
from typing import Any
from abc import ABC, abstractmethod

class BaseProblemGenerator(ABC):
    def __init__(self, config: Any):
        self.config_path = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def load_config(self):
        """Load problem specifig config"""
        pass

    @abstractmethod
    def generate(self):
        """Generate and save data"""
        pass

    @classmethod
    def register(cls, name):
        ProblemRegistry.register(name, cls)