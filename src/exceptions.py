from __future__ import annotations

class ConfigValidationError(ValueError):
    """Raised when a configuration fails validation checks"""
    def __init__(self, message: str, section: str | None = None):
        super().__init__(message)
        self.section = section  # Track which config section failed
        self.message = f"[{section}] {message}" if section else message

    def __str__(self):
        return self.message