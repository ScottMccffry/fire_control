"""
Core drone class for wildfire suppression operations.

This module defines the Drone class with realistic constraints including
battery life, water capacity, movement speed, and refill behavior.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
from enum import Enum
import numpy as np
import logging
from datetime import datetime, timedelta


class DroneState(Enum):
    """Possible states for a firefighting drone."""
    IDLE = "idle"
    FLYING_TO_FIRE = "flying_to_fire"
    DROPPING_WATER = "dropping_water"
    FLYING_TO_REFILL = "flying_to_refill"
    REFILLING = "refilling"
    RETURNING_TO_BASE = "returning_to_base"
    LANDED = "landed"
    OUT_OF_BATTERY = "out_of_battery"


@dataclass
class DroneCapabilities:
    """Physical capabilities and constraints of a drone."""
    max_flight_time_hours: float = 2.0  # Maximum flight time
    water_capacity_liters: float = 10.0  # Water tank capacity
    max_speed_ms: float = 15.0  # Maximum speed in m/s
    water_drop_rate_lps: float = 5.0  # Water drop rate in liters per second
    refill_time_seconds: float = 30.0  # Time to refill water tank
    battery_drain_rate: float = 0.02  # Battery % per minute of flight
    
    # Communication and sensing
    communication_range_m: float = 1000.0  # Radio range
    sensor_range_m: float = 500.0  # Fire detection range


class Drone:
    """
    Autonomous firefighting drone with realistic constraints.
    
    Each drone can:
    - Fly with limited battery (2 hours max)
    - Carry 10L of water with sprinkler system
    - Navigate to fire locations
    - Refill at designated water sources (lakes)
    - Coordinate with other drones
    """
    
    def __init__(self, 
                 drone_id: str,
                 initial_position: Tuple[float, float],  # (lat, lon)
                 capabilities: Optional[DroneCapabilities] = None):
        """
        Initialize a firefighting drone.
        
        Args:
            drone_id: Unique identifier for the drone
            initial_position: Starting (latitude, longitude) position
            capabilities: Physical capabilities (uses defaults if None)
        """
        self.drone_id = drone_id
        self.capabilities = capabilities or DroneCapabilities()
        
        # Position and movement
        self.position = initial_position  # (lat, lon)
        self.target_position: Optional[Tuple[float, float]] = None
        self.velocity = (0.0, 0.0)  # (lat_velocity, lon_velocity) in degrees/second
        self.altitude_m = 0.0  # Current altitude in meters
        
        # State management
        self.state = DroneState.IDLE
        self.last_state_change = datetime.now()
        
        # Resources
        self.battery_level = 1.0  # 0.0 to 1.0 (100%)
        self.water_level = 1.0  # 0.0 to 1.0 (100%)
        self.total_flight_time = 0.0  # Total flight time in minutes
        
        # Mission data
        self.assigned_fire_location: Optional[Tuple[float, float]] = None
        self.assigned_refill_location: Optional[Tuple[float, float]] = None
        self.mission_start_time: Optional[datetime] = None
        
        # Performance tracking
        self.total_water_dropped = 0.0  # Liters
        self.total_distance_flown = 0.0  # Meters
        self.missions_completed = 0
        self.fire_cells_suppressed = 0
        
        # Communication
        self.messages_received: List[Dict[str, Any]] = []
        self.nearby_drones: List[str] = []  # IDs of drones in communication range
        
        self.logger = logging.getLogger(f"Drone-{drone_id}")
    
    def update(self, dt_seconds: float, fire_map: np.ndarray, 
               refill_locations: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Update drone state for one time step.
        
        Args:
            dt_seconds: Time step in seconds
            fire_map: Current fire intensity map
            refill_locations: Available water refill locations (lat, lon)
            
        Returns:
            Action taken this step and any observations
        """
        action_taken = {"type": "none", "details": {}}
        
        # Update flight time and battery
        if self.state not in [DroneState.LANDED, DroneState.REFILLING]:
            self.total_flight_time += dt_seconds / 60.0
            self.battery_level -= (self.capabilities.battery_drain_rate / 60.0) * (dt_seconds / 60.0)
            self.battery_level = max(0.0, self.battery_level)
        
        # Check for critical battery
        if self.battery_level <= 0.1 and self.state != DroneState.OUT_OF_BATTERY:
            self._emergency_landing()
            action_taken = {"type": "emergency_landing", "details": {"battery": self.battery_level}}
        
        # State-specific behavior
        elif self.state == DroneState.IDLE:
            action_taken = self._handle_idle_state(fire_map, refill_locations)
        
        elif self.state == DroneState.FLYING_TO_FIRE:
            action_taken = self._handle_flying_to_fire(dt_seconds)
        
        elif self.state == DroneState.DROPPING_WATER:
            action_taken = self._handle_dropping_water(dt_seconds)
        
        elif self.state == DroneState.FLYING_TO_REFILL:
            action_taken = self._handle_flying_to_refill(dt_seconds)
        
        elif self.state == DroneState.REFILLING:
            action_taken = self._handle_refilling(dt_seconds)
        
        elif self.state == DroneState.RETURNING_TO_BASE:
            action_taken = self._handle_returning_to_base(dt_seconds)
        
        # Update position if moving
        if self.target_position and self.state in [
            DroneState.FLYING_TO_FIRE, 
            DroneState.FLYING_TO_REFILL, 
            DroneState.RETURNING_TO_BASE
        ]:
            self._update_position(dt_seconds)
        
        return action_taken
    
    def _handle_idle_state(self, fire_map: np.ndarray, 
                          refill_locations: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Handle behavior when drone is idle."""
        # Check if water refill is needed
        if self.water_level < 0.2:  # Less than 20% water
            self._start_refill_mission(refill_locations)
            return {"type": "start_refill", "details": {"water_level": self.water_level}}
        
        # Look for nearby fires to suppress
        fire_location = self._find_nearest_fire(fire_map)
        if fire_location:
            self._start_fire_mission(fire_location)
            return {"type": "start_fire_mission", "details": {"target": fire_location}}
        
        return {"type": "idle", "details": {}}
    
    def _handle_flying_to_fire(self, dt_seconds: float) -> Dict[str, Any]:
        """Handle flying to fire location."""
        if self._at_target_position():
            self._start_water_drop()
            return {"type": "arrived_at_fire", "details": {"position": self.position}}
        
        return {"type": "flying_to_fire", "details": {"progress": self._get_progress_to_target()}}
    
    def _handle_dropping_water(self, dt_seconds: float) -> Dict[str, Any]:
        """Handle water dropping operation."""
        if self.water_level <= 0.0:
            self._complete_water_drop()
            return {"type": "water_drop_complete", "details": {"water_dropped": self.capabilities.water_capacity_liters}}
        
        # Drop water at specified rate
        water_to_drop = min(self.water_level * self.capabilities.water_capacity_liters,
                           self.capabilities.water_drop_rate_lps * dt_seconds)
        
        self.water_level -= water_to_drop / self.capabilities.water_capacity_liters
        self.water_level = max(0.0, self.water_level)
        self.total_water_dropped += water_to_drop
        
        return {"type": "dropping_water", "details": {
            "water_dropped": water_to_drop,
            "position": self.position,
            "remaining_water": self.water_level
        }}
    
    def _handle_flying_to_refill(self, dt_seconds: float) -> Dict[str, Any]:
        """Handle flying to refill location."""
        if self._at_target_position():
            self._start_refilling()
            return {"type": "arrived_at_refill", "details": {"position": self.position}}
        
        return {"type": "flying_to_refill", "details": {"progress": self._get_progress_to_target()}}
    
    def _handle_refilling(self, dt_seconds: float) -> Dict[str, Any]:
        """Handle water refilling operation."""
        refill_rate = 1.0 / self.capabilities.refill_time_seconds  # % per second
        water_added = refill_rate * dt_seconds
        
        self.water_level = min(1.0, self.water_level + water_added)
        
        if self.water_level >= 1.0:
            self._complete_refilling()
            return {"type": "refill_complete", "details": {"water_level": self.water_level}}
        
        return {"type": "refilling", "details": {"water_level": self.water_level}}
    
    def _handle_returning_to_base(self, dt_seconds: float) -> Dict[str, Any]:
        """Handle returning to base/landing."""
        if self._at_target_position():
            self.state = DroneState.LANDED
            self.missions_completed += 1
            return {"type": "landed", "details": {"missions_completed": self.missions_completed}}
        
        return {"type": "returning_to_base", "details": {"progress": self._get_progress_to_target()}}
    
    def _start_fire_mission(self, fire_location: Tuple[float, float]):
        """Start a mission to suppress fire at given location."""
        self.assigned_fire_location = fire_location
        self.target_position = fire_location
        self.state = DroneState.FLYING_TO_FIRE
        self.mission_start_time = datetime.now()
        self.altitude_m = 50.0  # Flight altitude
        
        self.logger.info(f"Starting fire mission to {fire_location}")
    
    def _start_refill_mission(self, refill_locations: List[Tuple[float, float]]):
        """Start a mission to refill water."""
        if not refill_locations:
            self.logger.warning("No refill locations available")
            return
        
        # Find nearest refill location
        nearest_refill = min(refill_locations, 
                           key=lambda loc: self._distance_to_point(loc))
        
        self.assigned_refill_location = nearest_refill
        self.target_position = nearest_refill
        self.state = DroneState.FLYING_TO_REFILL
        self.altitude_m = 50.0
        
        self.logger.info(f"Starting refill mission to {nearest_refill}")
    
    def _start_water_drop(self):
        """Begin water dropping operation."""
        self.state = DroneState.DROPPING_WATER
        self.altitude_m = 10.0  # Lower altitude for water drop
        self.velocity = (0.0, 0.0)  # Stop moving
    
    def _complete_water_drop(self):
        """Complete water dropping and decide next action."""
        self.assigned_fire_location = None
        self.fire_cells_suppressed += 1
        
        if self.water_level < 0.1 or self.battery_level < 0.3:
            # Need refill or return to base
            self.state = DroneState.IDLE
        else:
            # Look for more fires
            self.state = DroneState.IDLE
    
    def _start_refilling(self):
        """Begin refilling operation."""
        self.state = DroneState.REFILLING
        self.altitude_m = 0.0  # Land for refilling
        self.velocity = (0.0, 0.0)
    
    def _complete_refilling(self):
        """Complete refilling and return to firefighting."""
        self.assigned_refill_location = None
        self.state = DroneState.IDLE
        self.altitude_m = 50.0  # Take off
    
    def _emergency_landing(self):
        """Emergency landing due to low battery."""
        self.state = DroneState.OUT_OF_BATTERY
        self.altitude_m = 0.0
        self.velocity = (0.0, 0.0)
        self.target_position = None
        
        self.logger.warning(f"Emergency landing due to low battery: {self.battery_level:.1%}")
    
    def _find_nearest_fire(self, fire_map: np.ndarray) -> Optional[Tuple[float, float]]:
        """Find nearest active fire location."""
        # This is a simplified version - would need proper coordinate conversion
        # For now, just find any active fire cells
        if np.any(fire_map > 100):  # Heat threshold for active fire
            fire_coords = np.where(fire_map > 100)
            if len(fire_coords[0]) > 0:
                # Return first fire location (simplified)
                return (float(fire_coords[0][0]), float(fire_coords[1][0]))
        
        return None
    
    def _update_position(self, dt_seconds: float):
        """Update position based on movement towards target."""
        if not self.target_position:
            return
        
        # Calculate direction to target
        lat_diff = self.target_position[0] - self.position[0]
        lon_diff = self.target_position[1] - self.position[1]
        distance = np.sqrt(lat_diff**2 + lon_diff**2)
        
        if distance > 0:
            # Normalize direction and apply speed
            speed_deg_per_sec = self.capabilities.max_speed_ms / 111000.0  # Rough conversion
            
            self.velocity = (
                (lat_diff / distance) * speed_deg_per_sec,
                (lon_diff / distance) * speed_deg_per_sec
            )
            
            # Update position
            new_lat = self.position[0] + self.velocity[0] * dt_seconds
            new_lon = self.position[1] + self.velocity[1] * dt_seconds
            
            # Track distance flown
            distance_flown = self.capabilities.max_speed_ms * dt_seconds
            self.total_distance_flown += distance_flown
            
            self.position = (new_lat, new_lon)
    
    def _at_target_position(self, tolerance_degrees: float = 0.001) -> bool:
        """Check if drone has reached target position."""
        if not self.target_position:
            return True
        
        distance = self._distance_to_point(self.target_position)
        return distance < tolerance_degrees
    
    def _distance_to_point(self, point: Tuple[float, float]) -> float:
        """Calculate distance to a point in degrees."""
        lat_diff = point[0] - self.position[0]
        lon_diff = point[1] - self.position[1]
        return np.sqrt(lat_diff**2 + lon_diff**2)
    
    def _get_progress_to_target(self) -> float:
        """Get progress towards target as percentage."""
        if not self.target_position:
            return 1.0
        
        # This would need the original distance for proper calculation
        return min(1.0, 1.0 - self._distance_to_point(self.target_position))
    
    def receive_message(self, message: Dict[str, Any], sender_id: str):
        """Receive a message from another drone or control system."""
        message['sender'] = sender_id
        message['timestamp'] = datetime.now()
        self.messages_received.append(message)
        
        # Keep only recent messages
        if len(self.messages_received) > 50:
            self.messages_received = self.messages_received[-50:]
    
    def send_status_update(self) -> Dict[str, Any]:
        """Generate status update for swarm coordination."""
        return {
            'drone_id': self.drone_id,
            'position': self.position,
            'state': self.state.value,
            'battery_level': self.battery_level,
            'water_level': self.water_level,
            'target_position': self.target_position,
            'altitude_m': self.altitude_m,
            'timestamp': datetime.now()
        }
    
    def get_sensor_data(self, fire_map: np.ndarray) -> Dict[str, Any]:
        """Get sensor observations of nearby fire activity."""
        # Simplified sensor model
        nearby_fire_intensity = 0.0
        fire_direction = None
        
        # In a real implementation, this would check fire_map around current position
        # For now, just return basic sensor data
        
        return {
            'position': self.position,
            'fire_intensity_nearby': nearby_fire_intensity,
            'fire_direction': fire_direction,
            'visibility': 1.0,  # Clear visibility
            'wind_detected': (0.0, 0.0)  # (speed, direction)
        }
    
    def is_operational(self) -> bool:
        """Check if drone is operational and can perform missions."""
        return (self.battery_level > 0.1 and 
                self.state not in [DroneState.OUT_OF_BATTERY, DroneState.LANDED])
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance statistics for this drone."""
        return {
            'drone_id': self.drone_id,
            'total_flight_time_minutes': self.total_flight_time,
            'total_water_dropped_liters': self.total_water_dropped,
            'total_distance_flown_meters': self.total_distance_flown,
            'missions_completed': self.missions_completed,
            'fire_cells_suppressed': self.fire_cells_suppressed,
            'current_battery': self.battery_level,
            'current_water': self.water_level,
            'operational': self.is_operational()
        } 