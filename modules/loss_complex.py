import torch

def loss_complex(g_u_real, g_u_imag, pred_real, pred_imag):
    loss_real = torch.mean((pred_real - g_u_real) ** 2)
    loss_imag = torch.mean((pred_imag - g_u_imag) ** 2)
    loss = loss_real + loss_imag

    return loss