"""Shared bridge error semantics (no SDK dependency)."""


class HabitatAdapterError(RuntimeError):
    """Raised for user-facing protocol errors in bridge requests."""
