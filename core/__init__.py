"""
Core Business Logic Module for Funnel Analytics Platform
========================================================

This module contains the core business logic classes:
- FunnelCalculator: Main funnel calculation engine
- DataSourceManager: Data loading and management
- FunnelConfigManager: Configuration management

Usage:
    from core import FunnelCalculator, DataSourceManager, FunnelConfigManager
"""

from .calculator import FunnelCalculator
from .config_manager import FunnelConfigManager
from .data_source import DataSourceManager

__all__ = ["DataSourceManager", "FunnelCalculator", "FunnelConfigManager"]
