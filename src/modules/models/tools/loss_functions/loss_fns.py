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

def mse_dissipative(y_pred: torch.Tensor, y_true: torch.Tensor):
    def energy(v: torch.Tensor):
        return 0.5 * (v**2).mean(dim=(-2, -1))
    mse = mse_loss(y_pred, y_true)
    E_pred = energy(y_pred)
    E_true = energy(y_true)
    c = 0.99
    dissipative_loss = torch.relu(E_pred - c*E_true).pow(2).mean()
    decay = 5e-3
    return mse + decay * dissipative_loss

def wasserstein_loss(pred, true, p=2):
    """
    Approximate Wasserstein distance between distributions
    Uses sliced Wasserstein for computational efficiency
    """
    batch_size, seq_len, dim = pred.shape
    
    pred_flat = pred.reshape(-1, dim)  # (batch_size * seq_len, dim)
    true_flat = true.reshape(-1, dim)
    
    n_projections = 50
    losses = []
    
    for _ in range(n_projections):
        direction = torch.randn(dim).to(device='mps')
        direction = direction / torch.norm(direction)
        
        pred_proj = torch.matmul(pred_flat, direction)
        true_proj = torch.matmul(true_flat, direction)
        
        pred_sorted, _ = torch.sort(pred_proj)
        true_sorted, _ = torch.sort(true_proj)
        
        if p == 1:
            loss = torch.mean(torch.abs(pred_sorted - true_sorted))
        else:
            loss = torch.mean((pred_sorted - true_sorted)**p)**(1/p)
        
        losses.append(loss)
    
    return torch.stack(losses).mean()



import torch
import torch.nn.functional as F

class OT_measure:
    def __init__(self, with_geomloss, blur):
        from geomloss import SamplesLoss
        if with_geomloss == 1:
            self.loss_geom = SamplesLoss(loss="sinkhorn", p=2, blur=blur, backend = 'online')
            print('You have selected to use Sinkhorn loss.')
        elif with_geomloss == 2:
            self.loss_geom = SamplesLoss(loss="sinkhorn", p=2, blur=blur, backend = 'online')
            print('You have selected to use MMD loss.')

    def loss(self, anchor_stats, out_stats):
        normalized_std = anchor_stats.std(dim = 1)[:, None, :].repeat(1, anchor_stats.shape[1], 1)
        normalized_mean = anchor_stats.mean(dim = 1)[:, None, :].repeat(1, anchor_stats.shape[1], 1)
        out_stats = (out_stats-normalized_mean) / normalized_std
        anchor_stats = (anchor_stats - normalized_mean) / normalized_std
        loss_geom_list = self.loss_geom(anchor_stats.contiguous(), out_stats.contiguous())
        loss_geom_mean = loss_geom_list.mean()

        return loss_geom_mean

import os
import numpy as np
from torch.utils.data import Dataset
import torch
import glob, pdb, random
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import KernelDensity
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable

def cal_stats_l96(anchor_t, out_t, args = None, index = 0):
    def compute_batch_gradient(input, wrt = 'T', order = 1):
        input = input.clone()
        assert len(input.shape) == 3 # B x T x d
        B, T, d = input.shape
        if wrt == 'T':
            ans = input.permute(0, 2, 1).reshape(B*d, T)
            grad = torch.gradient(ans, dim = 1)[0]
            if order > 1:
                grad = torch.gradient(grad, dim = 1)[0]
            grad =  grad.reshape(B, d, T).permute(0, 2, 1)

            mask = torch.ones_like(ans)
            mask[:, :1*order] = 0
            mask[:, -1*order:] = 0
            mask = mask.reshape(B, d, T).permute(0, 2, 1)
        elif wrt == 'd':
            ans = input.reshape(B*T, d)
            grad = torch.gradient(ans, dim = 1)[0]
            if order > 1:
                grad = torch.gradient(grad, dim = 1)[0]
            grad = grad.reshape(B, T, d)

            mask = torch.ones_like(ans)
            mask[:, :1*order] = 0
            mask[:, -1*order:] = 0
            mask = mask.reshape(B, T, d)
        return grad, mask
    var = anchor_t
    var_k_1 = torch.roll(var, 1, dims = 2)
    var_k_2 = torch.roll(var, 2, dims = 2)
    var_k_p_1 = torch.roll(var, -1, dims = 2)
    grad_t, mask = compute_batch_gradient(var, wrt = 'T', order = 1)
    #########################out stats#########################################
    var_out = out_t
    var_k_1_out = torch.roll(var_out, 1, dims = 2)
    var_k_2_out = torch.roll(var_out, 2, dims = 2)
    var_k_p_1_out = torch.roll(var_out, -1, dims = 2)
    grad_t_out, mask = compute_batch_gradient(var_out, wrt = 'T', order = 1)
    ###################### assembling stats ####################################
    advection_stats = var_k_1 * (var_k_2 - var_k_p_1)
    advection_stats_out = var_k_1_out * (var_k_2_out - var_k_p_1_out)
    advection_stats = advection_stats[:, 2:-2, 2:-2]
    grad_t = grad_t[:, 2:-2, 2:-2]
    var = var[:, 2:-2, 2:-2]
    advection_stats_out = advection_stats_out[:, 2:-2, 2:-2]
    grad_t_out = grad_t_out[:, 2:-2, 2:-2]
    var_out = var_out[:, 2:-2, 2:-2]
    b_tz = advection_stats.shape[0]
    anchor_stats = torch.stack([advection_stats, grad_t, var], dim = -1).reshape(b_tz, -1, 3)
    out_stats = torch.stack([advection_stats_out, grad_t_out, var_out], dim = -1).reshape(b_tz, -1, 3)
    return anchor_stats, out_stats





LOSS_FUNCTIONS = {
    "mse": mse_loss,
    "rmse": rmse_loss,
    "mag_phase": mag_phase_loss,
    'huber': huber_loss,
    "mse_dissipative": mse_dissipative,
    'wasserstein': wasserstein_loss
}
