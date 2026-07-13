"""
GUM - General User Models

A Python package for managing user feedback and interactions.
"""

__version__ = "0.1.2"

from .gum import gum
from .gumbo import Gumbo, Suggestion, TokenBucket, expected_utility, lexical_overlap
from .executor import (
    AgentBackend,
    AgentResult,
    ClaudeCLIBackend,
    ExecutionOutcome,
    Executor,
    RiskAssessment,
)

__all__ = [
    "gum", "Gumbo", "Suggestion", "TokenBucket", "expected_utility", "lexical_overlap",
    "Executor", "RiskAssessment", "AgentBackend", "AgentResult",
    "ClaudeCLIBackend", "ExecutionOutcome",
]