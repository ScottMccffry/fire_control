"""
Wildfire simulation interface using WRF-SFIRE via wrfxpy.

This module provides interfaces to run wildfire simulations and extract
flame front data for use in drone swarm coordination algorithms.
"""

from .fire_state import FireState, FlameFont
from .mock_fire import MockFireSimulator

# The WRF-SFIRE interface depends on heavy geospatial libraries (xarray,
# netCDF4). They are not needed for the mock simulator or RL prototyping, so
# import it lazily and don't fail the whole package if they're unavailable.
try:
    from .wrf_interface import WRFFireSimulator
except ImportError:  # pragma: no cover - optional dependency
    WRFFireSimulator = None

__all__ = [
    'WRFFireSimulator',
    'MockFireSimulator',
    'FireState',
    'FlameFont'
] 