"""
Mock wildfire simulator for rapid RL prototyping.

This module provides a simplified cellular automata-based fire simulation
that can be used for testing drone swarm algorithms without the overhead
of running full WRF-SFIRE simulations.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging

from .fire_state import FireState, FlameFont


class MockFireSimulator:
    """
    Simplified cellular automata fire simulator for rapid prototyping.
    
    This simulator models fire spread using simple rules based on:
    - Wind speed and direction
    - Fuel moisture content
    - Terrain slope
    - Fire intensity feedback
    """

    # 8-connected neighbour offsets, in the scan order used by the (legacy)
    # per-cell spread loop. The vectorized step relies on this ordering so that
    # ties in neighbour heat resolve to the same direction.
    _NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def __init__(self,
                 grid_size: Tuple[int, int] = (100, 100),
                 cell_size_meters: float = 100.0,
                 bounds: Tuple[float, float, float, float] = (34.0, -119.0, 35.0, -118.0)):
        """
        Initialize mock fire simulator.
        
        Args:
            grid_size: (rows, cols) of simulation grid
            cell_size_meters: Size of each grid cell in meters
            bounds: (min_lat, min_lon, max_lat, max_lon) of simulation domain
        """
        self.grid_size = grid_size
        self.cell_size = cell_size_meters
        self.bounds = bounds
        
        # Simulation state
        self.current_step = 0
        self.time_step_minutes = 5  # 5 minute time steps
        self.start_time = datetime.now()
        
        # Grid arrays
        self.burned_area = np.zeros(grid_size, dtype=int)
        self.heat_intensity = np.zeros(grid_size, dtype=float)
        self.fuel_load = np.random.uniform(0.5, 1.0, grid_size)  # Fuel availability
        self.fuel_moisture = np.random.uniform(0.1, 0.4, grid_size)  # Moisture content
        self.elevation = self._generate_terrain()
        
        # Weather conditions
        self.wind_speed = 5.0  # m/s
        self.wind_direction = 45.0  # degrees from north
        self.ambient_temp = 25.0  # Celsius
        self.humidity = 0.3  # relative humidity
        
        # Fire spread parameters
        self.base_spread_rate = 0.1  # base probability of spread per time step
        self.max_heat_intensity = 1000.0  # kW/m²
        self.ignition_threshold = 300.0  # kW/m² needed to ignite adjacent cells
        
        # History
        self.fire_history: List[FlameFont] = []
        
        # Water drop effects tracking
        self.water_effects = np.zeros(grid_size, dtype=float)  # Water suppression map
        self.water_decay_rate = 0.1  # How fast water effects decay per time step

        # Long-term retardant line (0..1 per cell). Raises the LOCAL ignition
        # threshold by up to `retardant_kw` and slows ignition, so a treated cell
        # resists fire but is NOT fireproof: a hot enough neighbour can still
        # breach it. Defaults to 0 everywhere (no effect -> backward compatible).
        self.retardant = np.zeros(grid_size, dtype=float)
        self.retardant_kw = 380.0  # extra kW/m^2 of incoming heat needed at R=1

        self.logger = logging.getLogger(__name__)

    def _generate_terrain(self) -> np.ndarray:
        """Generate realistic terrain elevation."""
        rows, cols = self.grid_size
        
        # Create some hills and valleys using Perlin-like noise
        x = np.linspace(0, 4*np.pi, cols)
        y = np.linspace(0, 4*np.pi, rows)
        X, Y = np.meshgrid(x, y)
        
        # Combine multiple frequency components
        terrain = (
            50 * np.sin(X) * np.cos(Y) +
            25 * np.sin(2*X) * np.cos(2*Y) +
            10 * np.sin(4*X) * np.cos(4*Y) +
            np.random.normal(0, 5, (rows, cols))
        )
        
        # Ensure positive elevation
        terrain = terrain - terrain.min() + 10
        
        return terrain
    
    def set_ignition_points(self, ignition_points: List[Tuple[int, int]]):
        """
        Set initial fire ignition points.
        
        Args:
            ignition_points: List of (row, col) grid coordinates for ignitions
        """
        for row, col in ignition_points:
            if 0 <= row < self.grid_size[0] and 0 <= col < self.grid_size[1]:
                self.burned_area[row, col] = 1
                self.heat_intensity[row, col] = self.max_heat_intensity * 0.8
                
        self.logger.info(f"Set {len(ignition_points)} ignition points")
    
    def set_weather(self, wind_speed: float, wind_direction: float, 
                   temperature: float, humidity: float):
        """Update weather conditions."""
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.ambient_temp = temperature
        self.humidity = humidity
    
    def apply_water_drop(self, row: int, col: int, water_amount: float = 10.0):
        """
        Apply water drop effect at specified location.
        
        Args:
            row, col: Grid coordinates
            water_amount: Liters of water dropped
        """
        if 0 <= row < self.grid_size[0] and 0 <= col < self.grid_size[1]:
            # Water effect radius (roughly 3x3 area for 10L)
            radius = max(1, int(np.sqrt(water_amount / 3.14)))
            
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    r, c = row + dr, col + dc
                    if 0 <= r < self.grid_size[0] and 0 <= c < self.grid_size[1]:
                        # Distance-based water effect
                        dist = np.sqrt(dr*dr + dc*dc)
                        if dist <= radius:
                            water_effect = water_amount * (1 - dist / radius)
                            self.water_effects[r, c] += water_effect
                            
                            # Immediate suppression effect
                            if self.heat_intensity[r, c] > 0:
                                suppression = min(self.heat_intensity[r, c], water_effect * 50)
                                self.heat_intensity[r, c] = max(0, self.heat_intensity[r, c] - suppression)
    
    def step(self) -> FireState:
        """
        Advance fire simulation by one time step.
        
        Returns:
            Current FireState after the step
        """
        self.current_step += 1

        # --- Vectorized fire spread (equivalent to the former per-cell loop) ---
        H, W = self.grid_size
        heat = self.heat_intensity
        burned = self.burned_area

        # 1) Heat decay; water accelerates it. Empty cells stay at 0.
        decay_rate = 0.05 + self.water_effects * 0.01
        new_heat = np.maximum(0.0, heat * (1.0 - decay_rate))

        # 2) Hottest 8-connected neighbour, plus the wind factor for the
        #    direction of that neighbour. Border neighbours are zero-filled.
        padded = np.zeros((H + 2, W + 2), dtype=float)
        padded[1:-1, 1:-1] = heat
        max_adj = np.zeros((H, W), dtype=float)
        wind_factor = np.ones((H, W), dtype=float)
        for dr, dc in self._NEIGHBORS:
            nbr = padded[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
            fire_direction = np.degrees(np.arctan2(dr, dc))
            wind_alignment = np.cos(np.radians(fire_direction - self.wind_direction))
            wf = 1.0 + (self.wind_speed / 20.0) * max(0.0, float(wind_alignment))
            # Keep the (first) hottest neighbour's wind factor, matching the
            # original scan order and its strict-greater update rule.
            newer = nbr > max_adj
            wind_factor = np.where(newer, wf, wind_factor)
            max_adj = np.where(newer, nbr, max_adj)

        # 3) Slope factor (faster uphill); border cells are neutral (1.0).
        padded_e = np.pad(self.elevation, 1, mode="edge")
        max_slope = np.zeros((H, W), dtype=float)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nbr_e = padded_e[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
            max_slope = np.maximum(max_slope, (nbr_e - self.elevation) / self.cell_size)
        slope_factor = np.minimum(2.0, 1.0 + np.maximum(0.0, max_slope) * 2.0)
        slope_factor[0, :] = slope_factor[-1, :] = 1.0
        slope_factor[:, 0] = slope_factor[:, -1] = 1.0

        # 4) Ignition probability for unburned cells with a hot-enough neighbour.
        #    Retardant raises the local threshold (needs a hotter neighbour to
        #    ignite) and damps the probability -- strong but not impassable.
        eff_threshold = self.ignition_threshold + self.retardant * self.retardant_kw
        candidate = (burned == 0) & (max_adj > eff_threshold)
        heat_factor = np.minimum(1.0, max_adj / self.max_heat_intensity)
        fuel_factor = self.fuel_load * (1.0 - self.fuel_moisture)
        water_suppression = np.minimum(0.9, self.water_effects / 50.0)
        jitter = 0.8 + 0.4 * np.random.random((H, W))
        prob = (self.base_spread_rate * heat_factor * fuel_factor
                * wind_factor * slope_factor * (1.0 - water_suppression)
                * (1.0 - 0.7 * self.retardant) * jitter)
        prob = np.where(candidate, np.clip(prob, 0.0, 1.0), 0.0)

        # 5) Stochastic ignition + initial heat for newly burned cells.
        ignite = (np.random.random((H, W)) < prob) & candidate
        init_jitter = 0.8 + 0.4 * np.random.random((H, W))
        initial_heat = (self.max_heat_intensity * 0.6 * self.fuel_load
                        * (1.0 - self.fuel_moisture) * init_jitter)

        new_burned = burned.copy()
        new_burned[ignite] = 1
        new_heat = np.where(ignite, initial_heat, new_heat)

        # Update state
        self.burned_area = new_burned
        self.heat_intensity = new_heat
        
        # Decay water effects
        self.water_effects *= (1 - self.water_decay_rate)
        
        # Create current flame front
        current_time = self.start_time + timedelta(minutes=self.current_step * self.time_step_minutes)
        flame_front = self._create_flame_front(current_time)
        
        # Update history
        self.fire_history.append(flame_front)
        if len(self.fire_history) > 100:  # Keep last 100 steps
            self.fire_history = self.fire_history[-100:]
        
        # Create fire state
        fire_state = FireState(
            current_flame_front=flame_front,
            flame_front_history=self.fire_history.copy(),
            weather_data={
                'wind_speed': self.wind_speed,
                'wind_direction': self.wind_direction,
                'temperature': self.ambient_temp,
                'humidity': self.humidity
            },
            terrain_data={'elevation': self.elevation.tolist()},
            fuel_data={'fuel_load': self.fuel_load.tolist(), 'fuel_moisture': self.fuel_moisture.tolist()},
            simulation_id=f"mock_sim_{id(self)}",
            domain_name="mock_domain",
            start_time=self.start_time,
            current_time=current_time
        )
        
        return fire_state
    
    def _calculate_ignition_probability(self, row: int, col: int) -> float:
        """Calculate probability of ignition for a cell."""
        base_prob = 0.0
        
        # Check adjacent cells for fire
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        max_adjacent_heat = 0.0
        wind_factor = 1.0
        
        for dr, dc in directions:
            adj_r, adj_c = row + dr, col + dc
            
            if 0 <= adj_r < self.grid_size[0] and 0 <= adj_c < self.grid_size[1]:
                adj_heat = self.heat_intensity[adj_r, adj_c]
                
                if adj_heat > max_adjacent_heat:
                    max_adjacent_heat = adj_heat
                    
                    # Calculate wind effect based on fire direction
                    fire_direction = np.degrees(np.arctan2(dr, dc))
                    wind_alignment = np.cos(np.radians(fire_direction - self.wind_direction))
                    wind_factor = 1.0 + (self.wind_speed / 20.0) * max(0, wind_alignment)
        
        if max_adjacent_heat > self.ignition_threshold:
            # Base probability from adjacent fire
            heat_factor = min(1.0, max_adjacent_heat / self.max_heat_intensity)
            base_prob = self.base_spread_rate * heat_factor
            
            # Modify by fuel conditions
            fuel_factor = self.fuel_load[row, col] * (1 - self.fuel_moisture[row, col])
            base_prob *= fuel_factor
            
            # Modify by wind
            base_prob *= wind_factor
            
            # Modify by slope (fire spreads faster uphill)
            slope_factor = self._calculate_slope_factor(row, col)
            base_prob *= slope_factor
            
            # Reduce by water effects
            water_suppression = min(0.9, self.water_effects[row, col] / 50.0)
            base_prob *= (1 - water_suppression)
            
            # Add some randomness
            base_prob *= (0.8 + 0.4 * random.random())
        
        return min(1.0, max(0.0, base_prob))
    
    def _calculate_initial_heat(self, row: int, col: int) -> float:
        """Calculate initial heat intensity for newly ignited cell."""
        base_heat = self.max_heat_intensity * 0.6
        
        # Modify by fuel load
        fuel_factor = self.fuel_load[row, col]
        heat = base_heat * fuel_factor
        
        # Modify by moisture (more moisture = less heat)
        moisture_factor = 1.0 - self.fuel_moisture[row, col]
        heat *= moisture_factor
        
        # Add randomness
        heat *= (0.8 + 0.4 * random.random())
        
        return heat
    
    def _calculate_slope_factor(self, row: int, col: int) -> float:
        """Calculate slope effect on fire spread."""
        # Simple slope calculation using adjacent elevation differences
        rows, cols = self.grid_size
        
        if row == 0 or row == rows-1 or col == 0 or col == cols-1:
            return 1.0
        
        # Calculate maximum upslope direction
        current_elev = self.elevation[row, col]
        max_slope = 0.0
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            adj_elev = self.elevation[row + dr, col + dc]
            slope = (adj_elev - current_elev) / self.cell_size
            max_slope = max(max_slope, slope)
        
        # Fire spreads faster uphill
        slope_factor = 1.0 + max(0, max_slope) * 2.0
        
        return min(2.0, slope_factor)
    
    def _create_flame_front(self, timestamp: datetime) -> FlameFont:
        """Create FlameFont object from current state."""
        # Extract fire perimeter
        fire_perimeter = self._extract_fire_perimeter()
        
        # Convert wind to grid arrays
        wind_speed_grid = np.full(self.grid_size, self.wind_speed)
        wind_direction_grid = np.full(self.grid_size, self.wind_direction)
        
        return FlameFont(
            burned_area=self.burned_area.copy(),
            heat_intensity=self.heat_intensity.copy(),
            fire_perimeter=fire_perimeter,
            timestamp=timestamp,
            time_step=self.current_step,
            grid_resolution=self.cell_size,
            bounds=self.bounds,
            spread_rate=None,  # Could calculate this
            fuel_moisture=self.fuel_moisture.copy(),
            wind_speed=wind_speed_grid,
            wind_direction=wind_direction_grid
        )
    
    def _extract_fire_perimeter(self) -> List[Tuple[float, float]]:
        """Extract fire perimeter coordinates."""
        from scipy import ndimage
        
        # Find fire perimeter using morphological operations
        if np.any(self.burned_area):
            # Dilate burned area and find boundary
            dilated = ndimage.binary_dilation(self.burned_area)
            perimeter = dilated & ~self.burned_area
            
            # Get coordinates of perimeter cells
            perimeter_coords = np.where(perimeter)
            
            # Convert to lat/lon
            perimeter_latlon = []
            for i, j in zip(perimeter_coords[0], perimeter_coords[1]):
                lat, lon = self._grid_to_latlon(i, j)
                perimeter_latlon.append((lat, lon))
            
            return perimeter_latlon
        
        return []
    
    def _grid_to_latlon(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid coordinates to lat/lon."""
        min_lat, min_lon, max_lat, max_lon = self.bounds
        
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        
        lat = min_lat + (row / self.grid_size[0]) * lat_range
        lon = min_lon + (col / self.grid_size[1]) * lon_range
        
        return lat, lon
    
    def _latlon_to_grid(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert lat/lon coordinates to grid indices."""
        min_lat, min_lon, max_lat, max_lon = self.bounds
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        
        row = int(((lat - min_lat) / lat_range) * self.grid_size[0])
        col = int(((lon - min_lon) / lon_range) * self.grid_size[1])
        
        # Clamp to valid range
        row = max(0, min(row, self.grid_size[0] - 1))
        col = max(0, min(col, self.grid_size[1] - 1))
        
        return row, col
    
    def reset(self):
        """Reset simulation to initial state."""
        self.current_step = 0
        self.start_time = datetime.now()
        self.burned_area = np.zeros(self.grid_size, dtype=int)
        self.heat_intensity = np.zeros(self.grid_size, dtype=float)
        self.water_effects = np.zeros(self.grid_size, dtype=float)
        self.retardant = np.zeros(self.grid_size, dtype=float)
        self.fire_history = []
    
    def get_total_burned_area(self) -> float:
        """Get total burned area in square meters."""
        burned_cells = np.sum(self.burned_area)
        return burned_cells * (self.cell_size ** 2)
    
    def get_active_fire_cells(self) -> List[Tuple[int, int]]:
        """Get coordinates of actively burning cells."""
        active_threshold = 100.0  # kW/m²
        active_coords = np.where(self.heat_intensity >= active_threshold)
        return list(zip(active_coords[0], active_coords[1])) 