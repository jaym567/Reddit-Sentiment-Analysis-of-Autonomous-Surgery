"""
Logging configuration for the relevance filter module.

This module provides centralized logging configuration for all components.
"""

import logging
import sys


def setup_logging(log_level: str = "INFO"):
    """
    Set up logging configuration for the relevance filter.
    
    Args:
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
    """
    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set level for relevance_filter module
    logger = logging.getLogger('relevance_filter')
    logger.setLevel(numeric_level)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Name of the module
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f'relevance_filter.{name}')
