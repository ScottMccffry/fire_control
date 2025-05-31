"""
Fire state data structures for representing wildfire simulation output.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from datetime import datetime


@dataclass
class FlameFont:
    """Represents a flame front at a specific time step."""
    
    # Spatial data
    burned_area: np.ndarray  # 2D binary array (1=burned, 0=unburned)
    heat_intensity: np.ndarray  # 2D array of heat intensity values (kW/m²)
    fire_perimeter: List[Tuple[float, float]]  # List of (lat, lon) coordinates
    
    # Temporal data
    timestamp: datetime
    time_step: int
    
    # Metadata
    grid_resolution: float  # meters per grid cell
    bounds: Tuple[float, float, float, float]  # (min_lat, min_lon, max_lat, max_lon)
    
    # Fire characteristics
    spread_rate: Optional[np.ndarray] = None  # m/s in each direction
    fuel_moisture: Optional[np.ndarray] = None  # %
    wind_speed: Optional[np.ndarray] = None  # m/s
    wind_direction: Optional[np.ndarray] = None  # degrees from north
    
    def get_active_fire_cells(self) -> List[Tuple[int, int]]:
        """Get coordinates of currently burning cells."""
        # Find cells with high heat intensity (active fire)
        active_threshold = np.percentile(self.heat_intensity[self.heat_intensity > 0], 75)
        active_cells = np.where(self.heat_intensity >= active_threshold)
        return list(zip(active_cells[0], active_cells[1]))
    
    def get_fire_size(self) -> float:
        """Get total burned area in square meters."""
        burned_cells = np.sum(self.burned_area)
        cell_area = self.grid_resolution ** 2
        return burned_cells * cell_area
    
    def grid_to_latlon(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid coordinates to lat/lon."""
        min_lat, min_lon, max_lat, max_lon = self.bounds
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        
        lat = min_lat + (row / self.burned_area.shape[0]) * lat_range
        lon = min_lon + (col / self.burned_area.shape[1]) * lon_range
        
        return lat, lon
    
    def latlon_to_grid(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert lat/lon coordinates to grid indices."""
        min_lat, min_lon, max_lat, max_lon = self.bounds
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        
        row = int(((lat - min_lat) / lat_range) * self.burned_area.shape[0])
        col = int(((lon - min_lon) / lon_range) * self.burned_area.shape[1])
        
        # Clamp to valid range
        row = max(0, min(row, self.burned_area.shape[0] - 1))
        col = max(0, min(col, self.burned_area.shape[1] - 1))
        
        return row, col


@dataclass  
class FireState:
    """Complete fire state including history and predictions."""
    
    # Current state
    current_flame_front: FlameFont
    
    # Historical data
    flame_front_history: List[FlameFont]
    
    # Environmental conditions
    weather_data: Dict[str, Any]
    terrain_data: Dict[str, Any]
    fuel_data: Dict[str, Any]
    
    # Simulation metadata
    simulation_id: str
    domain_name: str  # e.g., "california_camp_fire", "australia_bushfire"
    start_time: datetime
    current_time: datetime
    
    def get_fire_progression_rate(self) -> float:
        """Calculate average fire spread rate over recent history."""
        if len(self.flame_front_history) < 2:
            return 0.0
            
        recent_frames = self.flame_front_history[-10:]  # Last 10 time steps
        size_changes = []
        
        for i in range(1, len(recent_frames)):
            prev_size = recent_frames[i-1].get_fire_size()
            curr_size = recent_frames[i].get_fire_size()
            size_changes.append(curr_size - prev_size)
            
        return np.mean(size_changes) if size_changes else 0.0
    
    def get_fire_centroid(self) -> Tuple[float, float]:
        """Get the centroid of the current fire perimeter."""
        if not self.current_flame_front.fire_perimeter:
            return 0.0, 0.0
            
        lats = [p[0] for p in self.current_flame_front.fire_perimeter]
        lons = [p[1] for p in self.current_flame_front.fire_perimeter]
        
        return np.mean(lats), np.mean(lons)
    
    def predict_next_spread(self, time_horizon_minutes: int = 60) -> np.ndarray:
        """
        Simple fire spread prediction based on current conditions.
        Returns probability map of fire spread in the next time_horizon_minutes.
        """
        current_burned = self.current_flame_front.burned_area
        heat_intensity = self.current_flame_front.heat_intensity
        
        # Simple cellular automata-based prediction
        spread_prob = np.zeros_like(current_burned, dtype=float)
        
        # Fire spreads to adjacent unburned cells based on heat intensity
        for i in range(1, current_burned.shape[0] - 1):
            for j in range(1, current_burned.shape[1] - 1):
                if current_burned[i, j] == 0:  # Unburned cell
                    # Check adjacent cells for fire
                    adjacent_heat = [
                        heat_intensity[i-1, j], heat_intensity[i+1, j],
                        heat_intensity[i, j-1], heat_intensity[i, j+1]
                    ]
                    max_adjacent_heat = max(adjacent_heat)
                    
                    if max_adjacent_heat > 0:
                        # Base spread probability on heat intensity and wind
                        base_prob = min(0.8, max_adjacent_heat / 1000.0)  # Normalize heat
                        
                        # Modify by wind if available
                        if self.current_flame_front.wind_speed is not None:
                            wind_factor = 1.0 + (self.current_flame_front.wind_speed[i, j] / 20.0)
                            base_prob *= wind_factor
                            
                        # Modify by fuel moisture if available  
                        if self.current_flame_front.fuel_moisture is not None:
                            moisture_factor = 1.0 - (self.current_flame_front.fuel_moisture[i, j] / 100.0)
                            base_prob *= moisture_factor
                            
                        spread_prob[i, j] = min(1.0, base_prob)
        
        return spread_prob 