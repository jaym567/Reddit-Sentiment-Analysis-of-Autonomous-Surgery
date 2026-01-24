"""
Reddit Relevance Filter Module

A context-aware filtering module for Reddit posts and comments that identifies
content relevant to autonomous, robotic, and AI-assisted surgery research.
"""

from .models import RelevanceState, ParentContext, FilteredItem, FilterConfig
from .relevance_filter import RelevanceFilter

__all__ = [
    'RelevanceState',
    'ParentContext',
    'FilteredItem',
    'FilterConfig',
    'RelevanceFilter',
]

__version__ = '0.1.0'
