class InvalidStrategyCombinationError(Exception):
    """ Raised when an invalid combination of training strategy and output handling is used."""
    pass

class MissingSettingError(Exception):
    """Raised when a critical setting is missing for model/data/plots."""
    pass