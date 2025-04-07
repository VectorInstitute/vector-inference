"""Exceptions for the vector inference package."""


class ModelConfigurationError(Exception):
    """Raised when the model config or weights are missing or invalid."""

    pass


class MissingRequiredFieldsError(ValueError):
    """Raised when required fields are missing from the provided parameters."""

    pass


class ModelNotFoundError(KeyError):
    """Raised when the specified model name is not found in the configuration."""

    pass


class SlurmJobError(RuntimeError):
    """Raised when there's an error with a Slurm job."""

    pass


class APIError(Exception):
    """Base exception for API errors."""

    pass


class ServerError(Exception):
    """Exception raised when there's an error with the inference server."""

    pass
