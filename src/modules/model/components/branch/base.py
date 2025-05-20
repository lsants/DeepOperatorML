from abc import ABC, abstractmethod

class Branch(ABC):
    @abstractmethod
    def __init__(self, config: dict):
        pass