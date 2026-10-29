"""Compatibility imports for the original CPU pilot loss API."""

from .losses import (KNOWN_TRANSITIONS, observed_log_probabilities, per_example_loss,
                     validate_transition_tensor as validate_transition)

__all__ = ["KNOWN_TRANSITIONS", "observed_log_probabilities", "per_example_loss",
           "validate_transition"]
