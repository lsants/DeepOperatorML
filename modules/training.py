import torch
from .loss_complex import loss_complex

class TrainModel:
    def __init__(self, model, optimizer, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def __call__(self, sample, val=False):
        if not val:
            loss, pred_real, pred_imag = self.train_step(sample)
        else:
            loss, pred_real, pred_imag = self.val_step(sample)

        outputs = {'loss': loss,
                    'pred_real': pred_real,
                    'pred_imag': pred_imag}
        
        return outputs

    def train_step(self, sample):
        self.model.train()
        xb, xt = sample['xb'], sample['xt']
        g_u_real, g_u_imag = sample['g_u_real'], sample['g_u_imag']

        self.optimizer.zero_grad()
        pred_real, pred_imag = self.model(xb, xt)
        loss = loss_complex(g_u_real, g_u_imag, pred_real, pred_imag)
        loss.backward()
        self.optimizer.step()

        return loss.item(), pred_real, pred_imag

    def val_step(self, sample):
        self.model.eval()
        xb, xt = sample['xb'], sample['xt']
        g_u_real, g_u_imag = sample['g_u_real'], sample['g_u_imag']

        with torch.no_grad():  
            pred_real, pred_imag = self.model(xb, xt)
            loss = loss_complex(g_u_real, g_u_imag, pred_real, pred_imag)

        return loss.item(), pred_real, pred_imag