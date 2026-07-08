"""
GUM - General User Models

A Python package for managing user feedback and interactions.
"""

__version__ = "0.1.2"

from .gum import gum
from .gumbo import Gumbo, Suggestion, expected_utility, lexical_overlap

__all__ = ["gum", "Gumbo", "Suggestion", "expected_utility", "lexical_overlap"]