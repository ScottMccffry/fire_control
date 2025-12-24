"""
Improved fire physics model with Rothermel-based fire spread.

This module provides more realistic fire spread calculations based on
the Rothermel fire spread model, used by fire management agencies.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import IntEnum


class FuelModel(IntEnum):
    """Standard fuel model types (Anderson 13 fuel models)."""
    SHORT_GRASS = 1
    TIMBER_GRASS = 2
    TALL_GRASS = 3
    CHAPARRAL = 4
    BRUSH = 5
    DORMANT_BRUSH = 6
    SOUTHERN_ROUGH = 7
    CLOSED_TIMBER_LITTER = 8
    HARDWOOD_LITTER = 9
    TIMBER_UNDERSTORY = 10
    LIGHT_SLASH = 11
    MEDIUM_SLASH = 12
    HEAVY_SLASH = 13


@dataclass
class FuelProperties:
    """Physical properties for a fuel model."""
    name: str
    fuel_load_kg_m2: float  # Dead fuel load (kg/m^2)
    fuel_depth_m: float  # Fuel bed depth (m)
    dead_extinction_moisture: float  # Dead fuel moisture of extinction
    heat_content_kj_kg: float  # Heat content (kJ/kg)
    surface_area_volume_ratio: float  # Surface area to volume ratio (1/m)
    mineral_content: float  # Total mineral content (fraction)
    effective_mineral_content: float  # Effective mineral content (fraction)


# Standard fuel model properties
FUEL_MODELS: Dict[FuelModel, FuelProperties] = {
    FuelModel.SHORT_GRASS: FuelProperties(
        name="Short Grass",
        fuel_load_kg_m2=0.166,
        fuel_depth_m=0.305,
        dead_extinction_moisture=0.12,
        heat_content_kj_kg=18608,
        surface_area_volume_ratio=11483,
        mineral_content=0.0555,
        effective_mineral_content=0.01,
    ),
    FuelModel.TIMBER_GRASS: FuelProperties(
        name="Timber (grass and understory)",
        fuel_load_kg_m2=0.897,
        fuel_depth_m=0.305,
        dead_extinction_moisture=0.15,
        heat_content_kj_kg=18608,
        surface_area_volume_ratio=9842,
        mineral_content=0.0555,
        effective_mineral_content=0.01,
    ),
    FuelModel.CHAPARRAL: FuelProperties(
        name="Chaparral",
        fuel_load_kg_m2=4.593,
        fuel_depth_m=1.829,
        dead_extinction_moisture=0.20,
        heat_content_kj_kg=18608,
        surface_area_volume_ratio=4921,
        mineral_content=0.0555,
        effective_mineral_content=0.01,
    ),
    FuelModel.TIMBER_UNDERSTORY: FuelProperties(
        name="Timber with understory",
        fuel_load_kg_m2=1.345,
        fuel_depth_m=0.305,
        dead_extinction_moisture=0.25,
        heat_content_kj_kg=18608,
        surface_area_volume_ratio=6562,
        mineral_content=0.0555,
        effective_mineral_content=0.01,
    ),
}


class RothermelFireModel:
    """
    Rothermel fire spread model implementation.

    Based on Rothermel, R.C. (1972) "A Mathematical Model for Predicting
    Fire Spread in Wildland Fuels"
    """

    def __init__(self):
        """Initialize Rothermel model."""
        # Physical constants
        self.rho_p = 513.0  # Oven-dry particle density (kg/m^3)

    def calculate_spread_rate(
        self,
        fuel_model: FuelModel,
        fuel_moisture: float,
        wind_speed_mps: float,
        slope_degrees: float,
    ) -> float:
        """
        Calculate fire spread rate using Rothermel equations.

        Args:
            fuel_model: Fuel model type
            fuel_moisture: Dead fuel moisture content (fraction, e.g., 0.1 = 10%)
            wind_speed_mps: Midflame wind speed (m/s)
            slope_degrees: Terrain slope (degrees)

        Returns:
            Fire spread rate in m/min
        """
        fuel = FUEL_MODELS.get(fuel_model, FUEL_MODELS[FuelModel.SHORT_GRASS])

        # Check if fire can spread (moisture above extinction = no spread)
        if fuel_moisture >= fuel.dead_extinction_moisture:
            return 0.0

        # Packing ratio (beta)
        beta = fuel.fuel_load_kg_m2 / (fuel.fuel_depth_m * self.rho_p)

        # Optimum packing ratio
        sigma = fuel.surface_area_volume_ratio
        beta_op = 3.348 * (sigma ** -0.8189)

        # Relative packing ratio
        beta_ratio = beta / beta_op

        # Reaction intensity calculation
        w_n = fuel.fuel_load_kg_m2 * (1 - fuel.mineral_content)
        eta_s = 0.174 * (fuel.effective_mineral_content ** -0.19)

        # Moisture damping coefficient
        r_m = fuel_moisture / fuel.dead_extinction_moisture
        eta_m = 1 - 2.59 * r_m + 5.11 * (r_m ** 2) - 3.52 * (r_m ** 3)
        eta_m = max(0, eta_m)

        # Maximum reaction velocity
        sigma_15 = sigma ** 1.5
        gamma_max = sigma_15 / (495 + 0.0594 * sigma_15)

        # Optimum reaction velocity
        a = 133 * (sigma ** -0.7913)
        gamma = gamma_max * (beta_ratio ** a) * np.exp(a * (1 - beta_ratio))

        # Reaction intensity (kW/m^2)
        i_r = gamma * w_n * fuel.heat_content_kj_kg * eta_m * eta_s

        # Propagating flux ratio
        xi = np.exp((0.792 + 0.681 * (sigma ** 0.5)) * (beta + 0.1)) / (192 + 0.2595 * sigma)

        # Effective heating number
        epsilon = np.exp(-138 / sigma)

        # Heat of preignition (kJ/kg)
        q_ig = 250 + 1116 * fuel_moisture

        # No-wind, no-slope spread rate (m/min)
        rho_b = fuel.fuel_load_kg_m2 / fuel.fuel_depth_m
        r_0 = (i_r * xi) / (rho_b * epsilon * q_ig)

        # Wind factor
        c = 7.47 * np.exp(-0.133 * (sigma ** 0.55))
        b = 0.02526 * (sigma ** 0.54)
        e = 0.715 * np.exp(-0.000359 * sigma)

        # Convert wind speed to ft/min for standard formula, then back
        wind_ft_min = wind_speed_mps * 196.85  # m/s to ft/min
        phi_w = c * (wind_ft_min ** b) * (beta_ratio ** (-e))

        # Slope factor
        slope_tan = np.tan(np.radians(slope_degrees))
        phi_s = 5.275 * (beta ** -0.3) * (slope_tan ** 2)

        # Combined spread rate (m/min)
        r = r_0 * (1 + phi_w + phi_s)

        return max(0, r)

    def calculate_fireline_intensity(
        self,
        spread_rate_m_min: float,
        fuel_model: FuelModel,
        fuel_moisture: float,
    ) -> float:
        """
        Calculate fireline intensity (Byram's intensity).

        Args:
            spread_rate_m_min: Fire spread rate (m/min)
            fuel_model: Fuel model type
            fuel_moisture: Fuel moisture content (fraction)

        Returns:
            Fireline intensity in kW/m
        """
        fuel = FUEL_MODELS.get(fuel_model, FUEL_MODELS[FuelModel.SHORT_GRASS])

        # Available fuel (reduced by moisture)
        w_a = fuel.fuel_load_kg_m2 * (1 - fuel_moisture)

        # Heat content
        h = fuel.heat_content_kj_kg

        # Byram's fireline intensity (kW/m)
        i_b = (h * w_a * spread_rate_m_min) / 60

        return i_b

    def calculate_flame_length(self, fireline_intensity_kw_m: float) -> float:
        """
        Calculate flame length from fireline intensity.

        Uses Byram's flame length equation.

        Args:
            fireline_intensity_kw_m: Fireline intensity (kW/m)

        Returns:
            Flame length in meters
        """
        # Convert to BTU/ft/s for standard formula
        i_imperial = fireline_intensity_kw_m * 0.2889

        # Byram's equation (result in feet)
        l_ft = 0.45 * (i_imperial ** 0.46)

        # Convert back to meters
        return l_ft * 0.3048


class ImprovedFireSimulator:
    """
    Improved fire simulator with Rothermel physics.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (100, 100),
        cell_size_meters: float = 100.0,
        bounds: Tuple[float, float, float, float] = (34.0, -119.0, 35.0, -118.0),
        use_rothermel: bool = True,
    ):
        """Initialize improved fire simulator."""
        self.grid_size = grid_size
        self.cell_size = cell_size_meters
        self.bounds = bounds
        self.use_rothermel = use_rothermel

        self.rothermel = RothermelFireModel()

        # Initialize grids
        self._init_grids()

        # Simulation state
        self.current_step = 0
        self.time_step_seconds = 300  # 5 minutes

        # Weather
        self.wind_speed = 5.0  # m/s
        self.wind_direction = 0.0  # degrees from north
        self.temperature = 25.0  # Celsius
        self.humidity = 0.3

    def _init_grids(self):
        """Initialize simulation grids."""
        rows, cols = self.grid_size

        # Fire state
        self.burning = np.zeros((rows, cols), dtype=bool)
        self.burned = np.zeros((rows, cols), dtype=bool)
        self.heat_intensity = np.zeros((rows, cols), dtype=np.float32)
        self.time_of_arrival = np.full((rows, cols), -1.0, dtype=np.float32)

        # Fuel
        self.fuel_model = np.full((rows, cols), FuelModel.SHORT_GRASS, dtype=np.int32)
        self.fuel_moisture = np.random.uniform(0.05, 0.15, (rows, cols)).astype(np.float32)
        self.fuel_consumed = np.zeros((rows, cols), dtype=np.float32)

        # Terrain
        self.elevation = self._generate_terrain()
        self.slope = self._calculate_slope()
        self.aspect = self._calculate_aspect()

        # Water effects
        self.water_applied = np.zeros((rows, cols), dtype=np.float32)

    def _generate_terrain(self) -> np.ndarray:
        """Generate realistic terrain elevation."""
        rows, cols = self.grid_size

        # Multi-octave noise for terrain
        terrain = np.zeros((rows, cols))

        for octave in range(4):
            freq = 2 ** octave
            amp = 1.0 / freq

            x = np.linspace(0, freq * np.pi, cols)
            y = np.linspace(0, freq * np.pi, rows)
            X, Y = np.meshgrid(x, y)

            terrain += amp * (np.sin(X + np.random.random()) *
                             np.cos(Y + np.random.random()))

        # Scale to reasonable elevation range (500-1500m)
        terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
        terrain = terrain * 1000 + 500

        return terrain.astype(np.float32)

    def _calculate_slope(self) -> np.ndarray:
        """Calculate slope from elevation."""
        dy, dx = np.gradient(self.elevation, self.cell_size)
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        return slope.astype(np.float32)

    def _calculate_aspect(self) -> np.ndarray:
        """Calculate aspect (direction of slope) from elevation."""
        dy, dx = np.gradient(self.elevation, self.cell_size)
        aspect = np.degrees(np.arctan2(-dx, dy))
        aspect = np.where(aspect < 0, aspect + 360, aspect)
        return aspect.astype(np.float32)

    def set_ignition_points(self, points: list):
        """Set initial fire ignition points."""
        for row, col in points:
            if 0 <= row < self.grid_size[0] and 0 <= col < self.grid_size[1]:
                self.burning[row, col] = True
                self.heat_intensity[row, col] = 500.0  # Initial intensity
                self.time_of_arrival[row, col] = 0.0

    def set_weather(
        self,
        wind_speed: float,
        wind_direction: float,
        temperature: float,
        humidity: float,
    ):
        """Set weather conditions."""
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.temperature = temperature
        self.humidity = humidity

        # Update fuel moisture based on humidity
        base_moisture = 0.05 + 0.2 * humidity
        self.fuel_moisture = np.clip(
            self.fuel_moisture * (0.8 + 0.4 * humidity),
            base_moisture * 0.5,
            0.3
        )

    def step(self) -> Dict[str, Any]:
        """Advance simulation by one time step."""
        self.current_step += 1

        if self.use_rothermel:
            self._step_rothermel()
        else:
            self._step_simple()

        # Decay heat for burned cells
        burned_mask = self.burned & ~self.burning
        self.heat_intensity[burned_mask] *= 0.9

        # Apply water effects
        self._apply_water_decay()

        # Return state summary
        return self._get_state_summary()

    def _step_rothermel(self):
        """Fire spread step using Rothermel model."""
        rows, cols = self.grid_size
        time_hours = self.time_step_seconds / 3600

        # Get currently burning cells
        burning_cells = np.where(self.burning)

        # Track new ignitions
        new_burning = np.zeros_like(self.burning)

        for i, j in zip(burning_cells[0], burning_cells[1]):
            # Get local fuel model and moisture
            fuel = FuelModel(self.fuel_model[i, j])
            moisture = self.fuel_moisture[i, j]

            # Check neighbors for potential spread
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ni, nj = i + di, j + dj

                if not (0 <= ni < rows and 0 <= nj < cols):
                    continue

                if self.burned[ni, nj] or self.burning[ni, nj]:
                    continue

                # Calculate spread direction
                spread_dir = np.degrees(np.arctan2(dj, -di))
                if spread_dir < 0:
                    spread_dir += 360

                # Calculate effective wind speed in spread direction
                wind_alignment = np.cos(np.radians(spread_dir - self.wind_direction))
                effective_wind = self.wind_speed * max(0, wind_alignment)

                # Calculate slope in spread direction
                if di != 0 or dj != 0:
                    elev_diff = self.elevation[ni, nj] - self.elevation[i, j]
                    dist = self.cell_size * np.sqrt(di**2 + dj**2)
                    slope_deg = np.degrees(np.arctan(elev_diff / dist))
                else:
                    slope_deg = 0

                # Get spread rate
                spread_rate = self.rothermel.calculate_spread_rate(
                    fuel, moisture, effective_wind, slope_deg
                )

                # Calculate spread distance in this time step
                spread_distance = spread_rate * (self.time_step_seconds / 60)  # m

                # Probability of ignition based on spread distance
                cell_dist = self.cell_size * np.sqrt(di**2 + dj**2)
                ignition_prob = min(1.0, spread_distance / cell_dist)

                # Water suppression reduces ignition probability
                water_factor = 1 - min(0.9, self.water_applied[ni, nj] / 50)
                ignition_prob *= water_factor

                # Stochastic ignition
                if np.random.random() < ignition_prob:
                    new_burning[ni, nj] = True

                    # Calculate initial intensity
                    intensity = self.rothermel.calculate_fireline_intensity(
                        spread_rate, fuel, moisture
                    )
                    self.heat_intensity[ni, nj] = min(1000, intensity)

            # Check if current cell should stop burning
            self.fuel_consumed[i, j] += 0.1  # Fuel consumption rate
            if self.fuel_consumed[i, j] >= 1.0:
                self.burning[i, j] = False
                self.burned[i, j] = True

        # Apply new ignitions
        self.burning |= new_burning
        self.time_of_arrival[new_burning] = self.current_step * self.time_step_seconds

    def _step_simple(self):
        """Simple cellular automata fire spread (fallback)."""
        rows, cols = self.grid_size

        # Get burning cells
        burning_cells = np.where(self.burning)

        new_burning = np.zeros_like(self.burning)

        for i, j in zip(burning_cells[0], burning_cells[1]):
            # Check neighbors
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue

                    ni, nj = i + di, j + dj

                    if not (0 <= ni < rows and 0 <= nj < cols):
                        continue

                    if self.burned[ni, nj] or self.burning[ni, nj]:
                        continue

                    # Simple spread probability
                    prob = 0.1 * (1 - self.fuel_moisture[ni, nj])

                    # Wind effect
                    spread_dir = np.degrees(np.arctan2(dj, -di))
                    wind_align = np.cos(np.radians(spread_dir - self.wind_direction))
                    prob *= 1 + 0.5 * max(0, wind_align) * (self.wind_speed / 10)

                    # Water suppression
                    prob *= 1 - min(0.9, self.water_applied[ni, nj] / 50)

                    if np.random.random() < prob:
                        new_burning[ni, nj] = True
                        self.heat_intensity[ni, nj] = 500

            # Fuel consumption
            self.fuel_consumed[i, j] += 0.15
            if self.fuel_consumed[i, j] >= 1.0:
                self.burning[i, j] = False
                self.burned[i, j] = True

        self.burning |= new_burning

    def apply_water_drop(
        self,
        row: int,
        col: int,
        water_liters: float,
        radius_cells: int = 2,
    ):
        """Apply water drop at location."""
        rows, cols = self.grid_size

        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                nr, nc = row + dr, col + dc

                if 0 <= nr < rows and 0 <= nc < cols:
                    dist = np.sqrt(dr**2 + dc**2)
                    if dist <= radius_cells:
                        # Water amount decreases with distance
                        water = water_liters * (1 - dist / (radius_cells + 1))
                        self.water_applied[nr, nc] += water

                        # Direct suppression of heat
                        suppression = water * 30
                        self.heat_intensity[nr, nc] = max(
                            0, self.heat_intensity[nr, nc] - suppression
                        )

                        # Extinguish if enough water
                        if self.water_applied[nr, nc] > 40 and self.burning[nr, nc]:
                            self.burning[nr, nc] = False
                            self.burned[nr, nc] = True

    def _apply_water_decay(self):
        """Decay water effects over time."""
        self.water_applied *= 0.95

    def _get_state_summary(self) -> Dict[str, Any]:
        """Get current state summary."""
        return {
            'step': self.current_step,
            'burning_cells': int(np.sum(self.burning)),
            'burned_cells': int(np.sum(self.burned)),
            'total_affected': int(np.sum(self.burning | self.burned)),
            'mean_intensity': float(np.mean(self.heat_intensity[self.burning])) if np.any(self.burning) else 0,
            'max_intensity': float(np.max(self.heat_intensity)),
        }

    def get_active_fire_cells(self) -> list:
        """Get list of actively burning cells."""
        coords = np.where(self.burning)
        return list(zip(coords[0], coords[1]))

    def reset(self):
        """Reset simulation to initial state."""
        self.current_step = 0
        self._init_grids()
