import torch

class TrainEvaluator:
    def __init__(self, error_norm):
        self.error_norm = error_norm
        self.train_loss_history = None
        self.val_loss_history = None
        self.train_real_error_history = None
        self.train_imag_error_history = None
        self.val_real_error_history = None
        self.val_imag_error_history = None
        self.learning_rate_history = None

    def compute_error(self, g_u, pred):
        with torch.no_grad():
            error = torch.linalg.vector_norm((pred - g_u), ord=self.error_norm)\
                        / torch.linalg.vector_norm(g_u, ord=self.error_norm)
        return error.detach().cpu().numpy()
    
    def store_epoch_train_loss(self, epoch_loss):
        if not self.train_loss_history:
            self.train_loss_history = []
        self.train_loss_history.append(epoch_loss)

    def store_epoch_val_loss(self, epoch_loss):
        if not self.val_loss_history:
            self.val_loss_history = []
        self.val_loss_history.append(epoch_loss)
    
    def store_epoch_train_real_error(self, epoch_error):
        if not self.train_real_error_history:
            self.train_real_error_history = []
        self.train_real_error_history.append(epoch_error)
    
    def store_epoch_train_imag_error(self, epoch_error):
        if not self.train_imag_error_history:
            self.train_imag_error_history = []
        self.train_imag_error_history.append(epoch_error)
    
    def store_epoch_val_real_error(self, epoch_error):
        if not self.val_real_error_history:
            self.val_real_error_history = []
        self.val_real_error_history.append(epoch_error)
    
    def store_epoch_val_imag_error(self, epoch_error):
        if not self.val_imag_error_history:
            self.val_imag_error_history = []
        self.val_imag_error_history.append(epoch_error)

    def store_epoch_learning_rate(self, learning_rate):
        if not self.learning_rate_history:
            self.learning_rate_history = []
        self.learning_rate_history.append(learning_rate)

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

    def get_lr_history(self):
        lrs = self.learning_rate_history
        return lrs