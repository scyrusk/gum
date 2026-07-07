"""
Observer module for GUM - General User Models.

This module provides observer classes for different types of user interactions.
"""

from .observer import Observer
from .screen import Screen

__all__ = ["Observer", "Screen"]

# The Calendar observer depends on optional extras (aiohttp, ics) and reaches out
# to remote calendars, so it is imported lazily. Install with `pip install .[calendar]`
# to enable it; its absence never breaks the local screen-based GUM.
try:
    from .calendar import Calendar  # noqa: F401
    __all__.append("Calendar")
except ImportError:
    pass