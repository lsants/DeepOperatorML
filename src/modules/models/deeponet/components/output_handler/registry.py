from src.modules.models.deeponet.components.output_handler.protocol import OutputHandler
from src.modules.models.deeponet.components.output_handler.config import OutputConfig

class OutputRegistry:
    _handlers = {}

    @classmethod
    def register(cls, name: str):
        def decorator(handler_class):
            cls._handlers[name] = handler_class
            return handler_class
        return decorator

    @classmethod
    def create(cls, config: OutputConfig) -> OutputHandler:
        handler_class = cls._handlers.get(config.handler_type)
        if not handler_class:
            raise ValueError(f"Output handler {config.handler_type} not registered")
        return handler_class(config)
