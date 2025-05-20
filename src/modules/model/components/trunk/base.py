from abc import ABC, abstractmethod

class Trunk(ABC):
    @abstractmethod
    def __init__(self, config: dict):
        pass