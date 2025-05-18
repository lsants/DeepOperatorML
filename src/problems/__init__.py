import importlib
import pkgutil
from pathlib import Path

class ProblemRegistry:
    _generators = {}

    @classmethod
    def register(cls, name, generator_class):
        cls._generators[name.lower()] = generator_class

    @classmethod
    def get_generator(cls, name, config):
        try:
            return cls._generators[name.lower()](config)
        except KeyError:
            raise ValueError(f"No generator for problem '{name}'") from None

    @classmethod
    def auto_discover(cls):
        """Simple auto-discovery that only needs to be called once"""
        import importlib
        from pathlib import Path
        
        # Look in the current package directory
        package_path = Path(__file__).parent
        for dir_name in [d.name for d in package_path.iterdir() if d.is_dir()]:
            try:
                module = importlib.import_module(f".{dir_name}", package="src.problems")
                if hasattr(module, "PROBLEM_NAME") and hasattr(module, "Generator"):
                    cls.register(module.PROBLEM_NAME, module.Generator)
            except ImportError:
                continue

# Auto-discover on first import
ProblemRegistry.auto_discover()