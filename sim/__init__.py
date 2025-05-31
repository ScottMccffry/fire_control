"""
Wildfire simulation interface using WRF-SFIRE via wrfxpy.

This module provides interfaces to run wildfire simulations and extract
flame front data for use in drone swarm coordination algorithms.
"""

from .wrf_interface import WRFFireSimulator
from .fire_state import FireState, FlameFont
from .mock_fire import MockFireSimulator

__all__ = [
    'WRFFireSimulator',
    'MockFireSimulator', 
    'FireState',
    'FlameFont'
] 