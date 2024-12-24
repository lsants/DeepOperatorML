class HistoryStorer:
    def __init__(self, phases):
        """
        Initializes the evaluator with separate storage for training and validation metrics.
        """
        if not isinstance(phases, list):
            raise ValueError("Phases must be provided as a list.")
        
        # Dynamically initialize history based on provided phases
        self.history = {
            phase: {'train_loss': [], 
                    'train_errors': [], 
                    'val_loss': [], 
                    'val_errors': [],
                    'learning_rate': []}
            for phase in phases
        }

    def store_epoch_train_loss(self, phase, loss):
        if phase not in self.history:
            raise ValueError(f"Unknown phase: {phase}")
        self.history[phase]['train_loss'].append(loss)

    def store_epoch_train_errors(self, phase, errors):
        if phase not in self.history:
            raise ValueError(f"Unknown phase: {phase}")
        self.history[phase]['train_errors'].append(errors)

    def store_epoch_val_loss(self, phase, loss):
        if phase not in self.history:
            raise ValueError(f"Unknown phase: {phase}")
        self.history[phase]['val_loss'].append(loss)

    def store_epoch_val_errors(self, phase, errors):
        if phase not in self.history:
            raise ValueError(f"Unknown phase: {phase}")
        self.history[phase]['val_errors'].append(errors)

    def store_learning_rate(self, phase, learning_rate):
        if phase not in self.history:
            raise ValueError(f"Unknown phase: {phase}")
        self.history[phase]['learning_rate'].append(learning_rate)

    def get_history(self):
        return self.history

    def has_validation_data(self, phase):
        if phase in self.history:
            return bool(self.history[phase]['val_loss'])
        return False
