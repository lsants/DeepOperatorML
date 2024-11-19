import torch
from .loss_complex import loss_complex

class ModelTrainer:
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
        xb, xt = sample.get('xb'), sample.get('xt')
        g_u_real, g_u_imag = sample['g_u_real'], sample['g_u_imag']

        def closure():
            self.optimizer.zero_grad()
            pred_real, pred_imag = self.model(xb=xb, xt=xt)
            loss = loss_complex(g_u_real, g_u_imag, pred_real, pred_imag)
            loss.backward()
            return loss
        
        pred_real, pred_imag = self.model(xb, xt)
        loss = self.optimizer.step(closure)

        return loss.item(), pred_real, pred_imag

    def val_step(self, sample):
        self.model.eval()
        xb, xt = sample.get('xb'), sample.get('xt')
        g_u_real, g_u_imag = sample['g_u_real'], sample['g_u_imag']

        with torch.no_grad():  
            pred_real, pred_imag = self.model(xb=xb, xt=xt)
            loss = loss_complex(g_u_real, g_u_imag, pred_real, pred_imag)

        return loss.item(), pred_real, pred_imag
    

class TwoStepTrainer(ModelTrainer):
    def __init__(self, model, optimizer, scheduler=None, training_phase='both'):
        super().__init__(model, optimizer, scheduler)
        self.training_phase = training_phase

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
        xb, xt = sample.get('xb'), sample.get('xt')
        g_u_real, g_u_imag = sample['g_u_real'], sample['g_u_imag']

        self.optimizer.zero_grad()
        if self.training_phase == 'trunk':
            pred_real, pred_imag = self.model(xt=xt)
            print(pred_real.shape)
            print(g_u_real.shape)
            loss = loss_complex(g_u_real, g_u_imag, pred_real, pred_imag)
        elif self.training_phase == 'branch':
            pred_real, pred_imag = self.model(xb=xb)
            coefs_real = self.model.R @ self.model.A_real
            coefs_imag = self.model.R @ self.model.A_imag
            loss = loss_complex(coefs_real, coefs_imag, pred_real, pred_imag)
        else:
            pred_real, pred_imag = self.model(xb=xb, xt=xt)
            loss = loss_complex(g_u_real, g_u_imag, pred_real, pred_imag)

        loss.backward()
        self.optimizer.step()
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    print(f"{name}: grad norm = {param.grad.norm()}")
                else:
                    print(f"{name}: grad is None")

        if self.training_phase == 'trunk':
            pred_real, pred_imag = self.model(xt=xt)

        return loss.item(), pred_real, pred_imag

    def val_step(self, sample):
        self.model.eval()
        xb, xt = sample.get('xb'), sample.get('xt')
        g_u_real, g_u_imag = sample['g_u_real'], sample['g_u_imag']

        with torch.no_grad():
            if self.training_phase == 'trunk':
                basis_real, basis_imag = self.model(xt=xt)
                loss = loss_complex(g_u_real, g_u_imag, basis_real, basis_imag)
                return loss.item(), basis_real, basis_imag
            elif self.training_phase == 'branch':
                pred_real, pred_imag = self.model(xb=xb)
                coefs_real = self.model.R @ self.model.A_real
                coefs_imag = self.model.R @ self.model.A_imag
                loss = loss_complex(coefs_real, coefs_imag, pred_real, pred_imag)
            else:
                pred_real, pred_imag = self.model(xb=xb, xt=xt)
                loss = loss_complex(g_u_real, g_u_imag, pred_real, pred_imag)

        if self.training_phase == 'trunk':
            pred_real, pred_imag = self.model(xt=xt)
        else:
            pred_real, pred_imag = self.model(xb=xb, xt=xt)

        return loss.item(), pred_real, pred_imag