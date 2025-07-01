# loss_fns.py
import torch
from torch.nn.functional import huber_loss as _huber_loss


def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Handles multi-channel outputs via mean over all dimensions"""
    return torch.mean((y_pred - y_true) ** 2)


def rmse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(mse_loss(y_pred, y_true))


def mag_phase_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Expects 2-channel output: [real, imaginary]"""
    if y_pred.shape[1] != 2 or y_true.shape[1] != 2:
        raise ValueError(
            "Mag-phase loss requires 2 output channels (real & imaginary)")

    # Compute magnitudes and phases
    mag_pred = torch.norm(y_pred, dim=1)
    phase_pred = torch.atan2(y_pred[:, 1], y_pred[:, 0])

    mag_true = torch.norm(y_true, dim=1)
    phase_true = torch.atan2(y_true[:, 1], y_true[:, 0])

    # Phase difference with periodicity handling
    phase_diff = torch.atan2(torch.sin(phase_true - phase_pred),
                             torch.cos(phase_true - phase_pred))

    return torch.mean((mag_true - mag_pred)**2) + torch.mean(phase_diff**2)


def huber_loss(y_pred: torch.Tensor, y_true: torch.Tensor, delta: float = 1.5, reduction: str = 'mean') -> torch.Tensor:
    return _huber_loss(y_pred, y_true, delta=delta, reduction=reduction)


LOSS_FUNCTIONS = {
    "mse": mse_loss,
    "rmse": rmse_loss,
    "mag_phase": mag_phase_loss,
    'huber': huber_loss
}
