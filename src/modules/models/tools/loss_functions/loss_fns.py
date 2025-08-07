import torch
from torch.nn.functional import huber_loss as _huber_loss
from geomloss import SamplesLoss


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

def cal_stats_l63(anchor_t: torch.Tensor, out_t: torch.Tensor):
    """
    Compute statistics for Lorenz63 system.
    
    Lorenz63 equations:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y  
    dz/dt = xy - βz
    
    Input shape: B x T x 3 (batch, time, [x,y,z])
    """
    def grad_t_centered(u):
        return 0.5 * (u[:, 2:] - u[:, :-2])          # (B, T-2, d)

    def stats_l63(u):
        x, y, z = u.unbind(-1)                       # (B, T,)

        g = grad_t_centered(u)                       # (B, T-2, d)
        x, y, z = x[:, 1:-1], y[:, 1:-1], z[:, 1:-1] # align with g

        stats = torch.stack([
            y - x,                                   # linear_x
            g[..., 0],                               # FDA with respect to x
            x,                                              
            x * z,                                   # nonlinear_y
            -y,                                      # linear_y
            g[..., 1],                               # FDA with respect to y
            y,
            x * y,                                   # nonlinear_z
            -z,                                      # linear_z
            g[..., 2],                               # FDA with respect to z
            z
        ], dim=-1)                                   # (B, T-2, 11)

        return stats

    with torch.no_grad():
        anchor_stats_flat = stats_l63(anchor_t)
    
    out_stats_flat = stats_l63(out_t)
    
    return (anchor_stats_flat, out_stats_flat)

class OT_measure:
    def __init__(self, backend, blur=0.05):
        self.loss_geom = SamplesLoss(loss="sinkhorn", p=2, blur=blur, backend=backend)

    def loss(self, anchor_stats, out_stats):
        anchor_stats = anchor_stats.detach()
        normalized_std = anchor_stats.std(dim=1, keepdim=True) + 1e-8
        normalized_mean = anchor_stats.mean(dim=1, keepdim=True)
        out_stats = (out_stats-normalized_mean) / normalized_std
        anchor_stats = (anchor_stats - normalized_mean) / normalized_std
        loss_geom_list = self.loss_geom(anchor_stats.contiguous(), out_stats.contiguous())
        loss_geom_mean = loss_geom_list.mean()

        return loss_geom_mean

class CombinedLoss:
    def __init__(self, penalty: float, backend: str):
        self.mse_loss = mse_loss
        self.OT_loss = OT_measure(backend=backend).loss
        self.penalty = penalty

    def loss(self, y_pred, y_true):
        summary_statistics = cal_stats_l63(y_true, y_pred)
        anchor_stats, out_stats = summary_statistics
        mse_term = self.mse_loss(y_pred, y_true)
        ot_term = self.OT_loss(anchor_stats, out_stats)

        return mse_term + self.penalty * ot_term
    

def cal_stats_l63_old(anchor_t: torch.Tensor, out_t: torch.Tensor):
    """
    Compute statistics for Lorenz63 system.
    
    Lorenz63 equations:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y  
    dz/dt = xy - βz
    
    Input shape: B x T x 3 (batch, time, [x,y,z])
    """
    
    def compute_batch_gradient(input, wrt='T', order=1):
        input = input.clone()
        assert len(input.shape) == 3  # B x T x d
        B, T, d = input.shape
        
        if wrt == 'T':
            ans = input.permute(0, 2, 1).reshape(B*d, T)
            grad = torch.gradient(ans, dim=1)[0]
            if order > 1:
                grad = torch.gradient(grad, dim=1)[0]
            grad = grad.reshape(B, d, T).permute(0, 2, 1)
            mask = torch.ones_like(ans)
            mask[:, :1*order] = 0
            mask[:, -1*order:] = 0
            mask = mask.reshape(B, d, T).permute(0, 2, 1)
        elif wrt == 'd':
            ans = input.reshape(B*T, d)
            grad = torch.gradient(ans, dim=1)[0]
            if order > 1:
                grad = torch.gradient(grad, dim=1)[0]
            grad = grad.reshape(B, T, d)
            mask = torch.ones_like(ans)
            mask[:, :1*order] = 0
            mask[:, -1*order:] = 0
            mask = mask.reshape(B, T, d)
        return grad, mask
    
    var = anchor_t  # B x T x 3
    x, y, z = var[:, :, 0], var[:, :, 1], var[:, :, 2]
    
    grad_t, _ = compute_batch_gradient(var, wrt='T', order=1)

    var_out = out_t  # B x T x 3
    x_out, y_out, z_out = var_out[:, :, 0], var_out[:, :, 1], var_out[:, :, 2]
    
    grad_t_out, _ = compute_batch_gradient(var_out, wrt='T', order=1)
    
    ###################### Lorenz63 specific terms ####################################
    
    # Lorenz63 nonlinear terms:
    # dx/dt = σ(y - x)           -> linear term: σ(y - x)
    # dy/dt = x(ρ - z) - y       -> nonlinear term: x(ρ - z), linear term: -y
    # dz/dt = xy - βz            -> nonlinear term: xy, linear term: -βz
    
    linear_x = y - x 
    nonlinear_y = x * z 
    linear_y = -y 
    nonlinear_z = x * y 
    linear_z = -z
    
    linear_x_out = y_out - x_out
    nonlinear_y_out = x_out * z_out
    linear_y_out = -y_out
    nonlinear_z_out = x_out * y_out
    linear_z_out = -z_out
    
    # Apply boundary masking (remove boundary points affected by gradient computation)
    boundary = 2
    if var.shape[1] > 2 * boundary:  # Only apply if we have enough time points
        grad_t = grad_t[:, boundary:-boundary, :]
        var = var[:, boundary:-boundary, :]
        
        linear_x = linear_x[:, boundary:-boundary]
        nonlinear_y = nonlinear_y[:, boundary:-boundary]
        linear_y = linear_y[:, boundary:-boundary]
        nonlinear_z = nonlinear_z[:, boundary:-boundary]
        linear_z = linear_z[:, boundary:-boundary]
        
        grad_t_out = grad_t_out[:, boundary:-boundary, :]
        var_out = var_out[:, boundary:-boundary, :]
        
        linear_x_out = linear_x_out[:, boundary:-boundary]
        nonlinear_y_out = nonlinear_y_out[:, boundary:-boundary]
        linear_y_out = linear_y_out[:, boundary:-boundary]
        nonlinear_z_out = nonlinear_z_out[:, boundary:-boundary]
        linear_z_out = linear_z_out[:, boundary:-boundary]
    
    x_stats = torch.stack([linear_x, grad_t[:, :, 0], var[:, :, 0]], dim=-1)
    y_stats = torch.stack([nonlinear_y, linear_y, grad_t[:, :, 1], var[:, :, 1]], dim=-1)
    z_stats = torch.stack([nonlinear_z, linear_z, grad_t[:, :, 2], var[:, :, 2]], dim=-1)
    
    # Same for output data
    x_stats_out = torch.stack([linear_x_out, grad_t_out[:, :, 0], var_out[:, :, 0]], dim=-1)
    y_stats_out = torch.stack([nonlinear_y_out, linear_y_out, grad_t_out[:, :, 1], var_out[:, :, 1]], dim=-1)
    z_stats_out = torch.stack([nonlinear_z_out, linear_z_out, grad_t_out[:, :, 2], var_out[:, :, 2]], dim=-1)
    
    b_tz = var.shape[0]
    t_points = var.shape[1]
    
    anchor_stats_flat = torch.cat([
        x_stats.reshape(b_tz, t_points, 3),
        y_stats.reshape(b_tz, t_points, 4),
        z_stats.reshape(b_tz, t_points, 4)
    ], dim=-1)  # Shape: B x T x 11
    
    out_stats_flat = torch.cat([
        x_stats_out.reshape(b_tz, t_points, 3),
        y_stats_out.reshape(b_tz, t_points, 4),
        z_stats_out.reshape(b_tz, t_points, 4)
    ], dim=-1)  # Shape: B x T x 11
    
    return (anchor_stats_flat, out_stats_flat)

LOSS_FUNCTIONS = {
    "mse": mse_loss,
    "rmse": rmse_loss,
    "mag_phase": mag_phase_loss,
    'huber': huber_loss,
    "mse_dissipative": mse_dissipative,
    'ot': CombinedLoss(0.1, 'tensorized').loss
}


if __name__ == '__main__':
    i = torch.rand((100, 50, 3))
    o = torch.rand((100, 50, 3))
    i_stats, o_stats = cal_stats_l63(i, o)
    print('o', o_stats.shape)
    loss_fn = CombinedLoss(0.01, 'tensorized').loss
    loss = loss_fn(i_stats, o_stats)
    print(loss)
