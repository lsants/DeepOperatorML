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
    def __init__(self, model, optimizer, scheduler=None, training_phase='both', index_mapping=None):
        super().__init__(model, optimizer, scheduler)
        self.training_phase = training_phase
        self.index_map_for_A = index_mapping

    def __call__(self, sample):
        if self.training_phase == 'branch':
            loss, coefs_real, coefs_imag, pred_real, pred_imag = self.train_step(sample)
            outputs = {'loss': loss,
                       'pred_real': pred_real,
                       'pred_imag': pred_imag,
                       'coefs_real': coefs_real,
                       'coefs_imag': coefs_imag}
        else:
            loss, pred_real, pred_imag = self.train_step(sample)
            outputs = {'loss': loss,
                    'pred_real': pred_real,
                    'pred_imag': pred_imag}
        
        return outputs

    def train_step(self, sample):
        self.model.train()
        xb, xt = sample.get('xb'), sample.get('xt')
        g_u_real, g_u_imag = sample['g_u_real'], sample['g_u_imag']
        indices = sample['index']


        self.optimizer.zero_grad()

        if self.training_phase == 'trunk':
            pred_real, pred_imag = self.model(xt=xt)
            pred_real, pred_imag = pred_real.T, pred_imag.T

            loss = loss_complex(g_u_real, g_u_imag, pred_real, pred_imag)

            loss.backward()
            self.optimizer.step()

            return loss.item(), pred_real, pred_imag

        elif self.training_phase == 'branch':

            coefs_real = self.model.R @ self.model.A_list[0]
            coefs_imag = self.model.R @ self.model.A_list[1]

            num_basis = coefs_real.shape[0] # dims: (N, K) where K is batch size

            branch_out = self.model(xb=xb) # dims: (K, 2N)
            branch_out = branch_out.T # dims: (2N, K)
            
            pred_real = branch_out[ : num_basis , : ] # (N, K)
            pred_imag = branch_out[num_basis : , : ]  # (N, K)

            loss = loss_complex(coefs_real, coefs_imag, pred_real, pred_imag)

            loss.backward()
            self.optimizer.step()
            return loss.item(), coefs_real, coefs_imag, pred_real, pred_imag
        
        else:
            pred_real, pred_imag = self.model(xb=xb, xt=xt)
            loss = loss_complex(g_u_real, g_u_imag, pred_real, pred_imag)

            return loss.item(), pred_real, pred_imag