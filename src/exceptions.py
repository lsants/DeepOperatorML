class InvalidStrategyCombinationError(Exception):
    """ Raised when an invalid combination of training strategy and output handling is used."""
    def __init__(self, strategy, handling, reason, suggestion):
        message = (
            f"'{handling}' is incompatible with '{strategy.__class__.__name__}'.\n"
            f"Reason: {reason}\n"
            f"Suggestion: {suggestion}"
        )
        super().__init__(message)


class MissingSettingError(Exception):
    """Raised when a critical setting is missing for model/data/plots."""
    pass