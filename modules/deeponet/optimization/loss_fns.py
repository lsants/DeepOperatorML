import torch

def mse_loss(targets, preds):
    loss = 0
    for target, pred in zip(targets, preds):
         loss += torch.mean((pred - target) ** 2)
    return loss

def mag_phase_loss(targets, preds):
     if len(targets) < 2 or len(preds) < 2:
          raise ValueError("Magnitude-Phase loss requires complex (2-sized) output.")
     mag_target = torch.sqrt(targets[0] ** 2 + targets[1] ** 2)
     phase_target = torch.atan2(targets[1], targets[0])
     mag_pred = torch.sqrt(preds[0] ** 2 + preds[1] ** 2)
     phase_pred = torch.atan2(preds[1], preds[0])

     loss = torch.mean((mag_target - mag_pred) ** 2 + \
          torch.min(abs(phase_target - phase_pred), 2 * torch.pi - abs(phase_target - phase_pred)))
     return loss

LOSS_FUNCTIONS = {
     "mse" : mse_loss,
     "mag_phase" : mag_phase_loss
}