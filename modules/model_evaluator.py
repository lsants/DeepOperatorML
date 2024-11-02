import torch

class Evaluator:
    def __init__(self, error_norm):
        self.error_norm = error_norm
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_real_error_history = []
        self.train_imag_error_history = []
        self.val_real_error_history = []
        self.val_imag_error_history = []

    def compute_batch_error(self, g_u, pred):
        with torch.no_grad():
            batch_error = torch.linalg.vector_norm((pred - g_u), ord=self.error_norm)\
                        / torch.linalg.vector_norm(g_u, ord=self.error_norm)
        return batch_error.detach().numpy()
    
    def store_epoch_train_loss(self, epoch_loss):
        self.train_loss_history.append(epoch_loss)

    def store_epoch_val_loss(self, epoch_loss):
        self.val_loss_history.append(epoch_loss)
    
    def store_epoch_train_real_error(self, epoch_error):
        self.train_real_error_history.append(epoch_error)
    
    def store_epoch_train_imag_error(self, epoch_error):
        self.train_imag_error_history.append(epoch_error)
    
    def store_epoch_val_real_error(self, epoch_error):
        self.val_real_error_history.append(epoch_error)
    
    def store_epoch_val_imag_error(self, epoch_error):
        self.val_imag_error_history.append(epoch_error)

    def get_loss_history(self):
        losses = {'train' : self.train_loss_history,
                  'val' : self.val_loss_history}
        return losses
    
    def get_error_history(self):
        train_errors = {'real' : self.train_real_error_history,
                        'imag' : self.train_imag_error_history}
        val_errors = {'real' : self.val_real_error_history,
                      'imag' : self.val_imag_error_history}
        
        errors = {'train' : train_errors,
                  'val': val_errors}
        return errors
