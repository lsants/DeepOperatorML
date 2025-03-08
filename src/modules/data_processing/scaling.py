import torch

class Scaling:
    def __init__(self, min_val: float | torch.Tensor | None=None, max_val: float | torch.Tensor | None=None, mean: float | torch.Tensor | None=None, std: float | torch.Tensor | None=None) -> None:
        """
        A generic class for scaling values, supporting both normalization and standardization.

        Args:
            min_val (float or Tensor, optional): Minimum value for normalization.
            max_val (float or Tensor, optional): Maximum value for normalization.
            mean (float or Tensor, optional): Mean value for standardization.
            std (float or Tensor, optional): Standard deviation for standardization.
        """
        self.min_val = min_val
        self.max_val = max_val
        self.mean = mean
        self.std = std

        if not ((min_val is not None and max_val is not None) or (mean is not None and std is not None)):
            raise ValueError("Either min_val and max_val or mean and std must be provided.")
        
    def __repr__(self):
        return f"Scaling(min={self.min_val}, max={self.max_val})"

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Normalizes values to the range [0, 1].

        Args:
            values (Tensor): Input values.

        Returns:
            Tensor: Normalized values.
        """
        if self.min_val is None or self.max_val is None:
            raise ValueError("min_val and max_val must be provided for normalization.")
        
        v_min = torch.as_tensor(self.min_val, dtype=values.dtype, device=values.device)
        v_max = torch.as_tensor(self.max_val, dtype=values.dtype, device=values.device)
        return (values - v_min) / (v_max - v_min)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Denormalizes values from the range [0, 1] back to the original range.

        Args:
            values (Tensor): Input normalized values.

        Returns:
            Tensor: Denormalized values.
        """
        if self.min_val is None or self.max_val is None:
            raise ValueError("min_val and max_val must be provided for denormalization.")
        
        v_min = torch.as_tensor(self.min_val, dtype=values.dtype, device=values.device)
        v_max = torch.as_tensor(self.max_val, dtype=values.dtype, device=values.device)
        return values * (v_max - v_min) + v_min

    def standardize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Standardizes values using the provided mean and standard deviation.

        Args:
            values (Tensor): Input values.

        Returns:
            Tensor: Standardized values.
        """
        if self.mean is None or self.std is None:
            raise ValueError("mean and std must be provided for standardization.")
        
        mu = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
        sigma = torch.as_tensor(self.std, dtype=values.dtype, device=values.device)
        return (values - mu) / sigma

    def destandardize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Destandardizes values back to the original scale using mean and standard deviation.

        Args:
            values (Tensor): Input standardized values.

        Returns:
            Tensor: Destandardized values.
        """
        if self.mean is None or self.std is None:
            raise ValueError("mean and std must be provided for destandardization.")
        
        mu = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
        sigma = torch.as_tensor(self.std, dtype=values.dtype, device=values.device)
        return values * sigma + mu