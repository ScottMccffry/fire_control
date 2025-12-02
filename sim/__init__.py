"""
Wildfire simulation interface using WRF-SFIRE via wrfxpy.

This module provides interfaces to run wildfire simulations and extract
flame front data for use in drone swarm coordination algorithms.
"""

from .fire_state import FireState, FlameFont
from .mock_fire import MockFireSimulator

# Optional imports that require additional dependencies
try:
    from .wrf_interface import WRFFireSimulator
except ImportError:
    WRFFireSimulator = None

try:
    from .fire_physics import ImprovedFireSimulator, RothermelFireModel, FuelModel
except ImportError:
    ImprovedFireSimulator = None
    RothermelFireModel = None
    FuelModel = None

__all__ = [
    'WRFFireSimulator',
    'MockFireSimulator',
    'ImprovedFireSimulator',
    'RothermelFireModel',
    'FuelModel',
    'FireState',
    'FlameFont'
] 