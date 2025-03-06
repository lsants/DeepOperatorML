class HistoryStorer:
    def __init__(self, phases: list) -> None:
        """
        Initializes the evaluator with separate storage for training and validation metrics.
        """
        self.history = {
            phase: {'train_loss': [], 
                    'train_errors': [], 
                    'val_loss': [], 
                    'val_errors': [],
                    'learning_rate': []}
            for phase in phases
        }

    def store_epoch_train_loss(self, phase: str, loss: float) -> None:
        if phase not in self.history:
            raise ValueError(f"Unknown phase: {phase}")
        self.history[phase]['train_loss'].append(loss)

    def store_epoch_train_errors(self, phase: str, errors) -> None:
        if phase not in self.history:
            raise ValueError(f"Unknown phase: {phase}")
        self.history[phase]['train_errors'].append(errors)

    def store_epoch_val_loss(self, phase: str, loss: float) -> None:
        if phase not in self.history:
            raise ValueError(f"Unknown phase: {phase}")
        self.history[phase]['val_loss'].append(loss)

    def store_epoch_val_errors(self, phase: str, errors) -> None:
        if phase not in self.history:
            raise ValueError(f"Unknown phase: {phase}")
        self.history[phase]['val_errors'].append(errors)

    def store_learning_rate(self, phase: str, learning_rate: float) -> None:
        if phase not in self.history:
            raise ValueError(f"Unknown phase: {phase}")
        self.history[phase]['learning_rate'].append(learning_rate)

    def get_history(self) -> dict[str, dict[str, list[float]]]:
        return self.history

    def has_validation_data(self, phase: str) -> bool:
        if phase in self.history:
            return bool(self.history[phase]['val_loss'])
        return False
