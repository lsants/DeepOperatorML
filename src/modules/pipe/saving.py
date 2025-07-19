from __future__ import annotations
import logging
import torch
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional


class Saver:
    def __init__(
        self,
        model_name: Optional[str] = None,
        full_logging: bool = False
    ) -> None:
        """
        Initializes the Saver.

        Args:
            model_name: Optional name for logging context.
            full_logging: Whether to log save actions.
        """
        self.name = model_name
        self.full_logging = full_logging
        self.logger = logging.getLogger(__name__)

    def set_logging(self, full_logging: bool) -> None:
        """Enable or disable logging of save actions."""
        self.full_logging = full_logging

    def _make_serializable(self, obj: Any) -> Any:
        """
        Recursively convert objects to JSON/YAML serializable types.
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, (float, int, str, bool)) or obj is None:
            return obj
        return str(obj)

    def save_checkpoint(
        self,
        file_path: str,
        model_dict: Dict[str, Any],
    ) -> None:
        """
        Save model and optimizer state dictionaries to a checkpoint file.
        """
        torch.save(model_dict, file_path)
        if self.full_logging:
            self.logger.info(f"Checkpoint saved to: {file_path}")

    def save_model_state(self, file_path: str, model_state: Any) -> None:
        """Save raw model state to a file."""
        torch.save(model_state, file_path)
        if self.full_logging:
            self.logger.info(f"Model state saved to: {file_path}")

    def save_model_info(
        self,
        file_path: str | Path,
        model_info: Dict[str, Any]
    ) -> None:
        """Save model metadata or hyperparameters to a YAML file."""
        serializable = self._make_serializable(model_info)
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(serializable, f, indent=4)
        if self.full_logging:
            self.logger.info(f"Model info saved to: {file_path}")

    def save_history(
        self,
        file_path: str | Path,
        history: Dict[str, Any]
    ) -> None:
        """Save training history or metrics to a YAML or TXT file."""
        serializable = self._make_serializable(history)
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(serializable, f, indent=4)
        if self.full_logging:
            self.logger.info(f"History saved to: {file_path}")

    def save_plots(
        self,
        file_path: str | Path,
        figure: plt.Figure
    ) -> None:
        """Save a matplotlib figure to an image file."""
        figure.savefig(file_path)
        if self.full_logging:
            self.logger.info(f"Figure saved to: {file_path}")

    def save_errors(self, file_path: str | Path, errors: Dict[str, Any]) -> None:
        """Save error logs or dictionaries to a YAML or TXT file."""
        serializable = self._make_serializable(errors)
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(serializable, f, indent=4)
        if self.full_logging:
            self.logger.info(f"Errors saved to: {file_path}")

    def save_time(self, file_path: str | Path, times: Dict[str, Any]) -> None:
        """Save timing information or profiling results to a YAML or TXT file."""
        serializable = self._make_serializable(times)
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(serializable, f, indent=4)
        if self.full_logging:
            self.logger.info(f"Time information saved to: {file_path}")

    def save_metrics(self, file_path: str | Path, metrics: dict[str, Any]) -> None:
        """Save metrics to a YAML or TXT file."""
        serializable = self._make_serializable(metrics)
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(serializable, f, indent=4)
        if self.full_logging:
            self.logger.info(f"Metrics saved to: {file_path}")

    def save_transform_pipeline(
        self,
        file_path: str | Path,
        transform_pipeline: Any
    ) -> None:
        """Save a transform pipeline to a file."""
        transform_pipeline.save(file_path)
        if self.full_logging:
            self.logger.info(f"Transform pipeline saved to: {file_path}")

    def save_output_data(
        self,
        file_path: str,
        data: Dict[str, np.ndarray]
    ) -> None:
        """Save prediction outputs or numpy arrays to an NPZ file."""
        np.savez(file_path, **data)
        if self.full_logging:
            self.logger.info(f"Output data saved to: {file_path}")
