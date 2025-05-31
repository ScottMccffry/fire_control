"""
Baseline drone behavior policies for wildfire suppression.

This module provides various baseline policies that can be used for
comparison with reinforcement learning approaches.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime
import logging

from .drone import Drone, DroneState


class DronePolicy(ABC):
    """Abstract base class for drone behavior policies."""
    
    @abstractmethod
    def select_action(self, drone: Drone, fire_map: np.ndarray, 
                     other_drones: List[Drone], refill_locations: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Select action for a drone given current state.
        
        Args:
            drone: The drone making the decision
            fire_map: Current fire intensity map
            other_drones: List of other drones in the swarm
            refill_locations: Available water refill locations
            
        Returns:
            Action dictionary with type and parameters
        """
        pass
    
    def get_policy_name(self) -> str:
        """Get human-readable policy name."""
        return self.__class__.__name__


class RandomPolicy(DronePolicy):
    """
    Random policy that makes random decisions.
    
    Useful as a baseline for comparison and for exploration.
    """
    
    def __init__(self, action_probabilities: Optional[Dict[str, float]] = None):
        """
        Initialize random policy.
        
        Args:
            action_probabilities: Custom probabilities for different actions
        """
        self.action_probs = action_probabilities or {
            'move_to_fire': 0.4,
            'move_to_refill': 0.2,
            'stay_idle': 0.2,
            'random_move': 0.2
        }
        
        # Normalize probabilities
        total = sum(self.action_probs.values())
        self.action_probs = {k: v/total for k, v in self.action_probs.items()}
        
        self.logger = logging.getLogger("RandomPolicy")
    
    def select_action(self, drone: Drone, fire_map: np.ndarray, 
                     other_drones: List[Drone], refill_locations: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Select random action based on drone state and probabilities."""
        
        # If drone is busy, let it continue
        if drone.state != DroneState.IDLE:
            return {"type": "continue", "details": {}}
        
        # If drone needs water or battery, prioritize those
        if drone.water_level < 0.2:
            return self._select_refill_action(drone, refill_locations)
        
        if drone.battery_level < 0.3:
            return {"type": "return_to_base", "details": {}}
        
        # Random action selection
        action_type = np.random.choice(
            list(self.action_probs.keys()),
            p=list(self.action_probs.values())
        )
        
        if action_type == 'move_to_fire':
            return self._select_fire_action(drone, fire_map)
        elif action_type == 'move_to_refill':
            return self._select_refill_action(drone, refill_locations)
        elif action_type == 'random_move':
            return self._select_random_move_action(drone)
        else:  # stay_idle
            return {"type": "idle", "details": {}}
    
    def _select_fire_action(self, drone: Drone, fire_map: np.ndarray) -> Dict[str, Any]:
        """Select a random fire location to target."""
        fire_threshold = 100.0
        fire_locations = np.where(fire_map >= fire_threshold)
        
        if len(fire_locations[0]) > 0:
            # Pick random fire location
            idx = random.randint(0, len(fire_locations[0]) - 1)
            fire_row, fire_col = fire_locations[0][idx], fire_locations[1][idx]
            
            return {
                "type": "target_fire",
                "details": {
                    "grid_location": (fire_row, fire_col),
                    "intensity": float(fire_map[fire_row, fire_col])
                }
            }
        
        return {"type": "idle", "details": {}}
    
    def _select_refill_action(self, drone: Drone, refill_locations: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Select a random refill location."""
        if refill_locations:
            refill_location = random.choice(refill_locations)
            return {
                "type": "target_refill",
                "details": {"location": refill_location}
            }
        
        return {"type": "idle", "details": {}}
    
    def _select_random_move_action(self, drone: Drone) -> Dict[str, Any]:
        """Select a random movement direction."""
        # Random movement within some bounds
        current_lat, current_lon = drone.position
        
        # Move within 0.01 degrees (roughly 1km)
        delta_lat = (random.random() - 0.5) * 0.02
        delta_lon = (random.random() - 0.5) * 0.02
        
        new_lat = current_lat + delta_lat
        new_lon = current_lon + delta_lon
        
        return {
            "type": "move_to_position",
            "details": {"target_position": (new_lat, new_lon)}
        }


class GreedyFirePolicy(DronePolicy):
    """
    Greedy policy that always targets the nearest/highest intensity fire.
    
    Simple heuristic policy that provides reasonable baseline performance.
    """
    
    def __init__(self, priority_mode: str = "nearest", coordination_enabled: bool = False):
        """
        Initialize greedy fire policy.
        
        Args:
            priority_mode: "nearest", "highest_intensity", or "combined"
            coordination_enabled: Whether to avoid targeting same fires as other drones
        """
        self.priority_mode = priority_mode
        self.coordination_enabled = coordination_enabled
        self.logger = logging.getLogger("GreedyFirePolicy")
    
    def select_action(self, drone: Drone, fire_map: np.ndarray, 
                     other_drones: List[Drone], refill_locations: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Select greedy action based on priority mode."""
        
        # Handle non-idle states
        if drone.state != DroneState.IDLE:
            return {"type": "continue", "details": {}}
        
        # Check resource constraints
        if drone.water_level < 0.1:
            return self._select_best_refill(drone, refill_locations)
        
        if drone.battery_level < 0.2:
            return {"type": "return_to_base", "details": {}}
        
        # Find best fire target
        best_fire = self._find_best_fire_target(drone, fire_map, other_drones)
        
        if best_fire:
            return {
                "type": "target_fire",
                "details": {
                    "grid_location": best_fire["location"],
                    "intensity": best_fire["intensity"],
                    "priority_score": best_fire["score"]
                }
            }
        
        # No fires found, go idle or patrol
        return {"type": "idle", "details": {}}
    
    def _find_best_fire_target(self, drone: Drone, fire_map: np.ndarray, 
                              other_drones: List[Drone]) -> Optional[Dict[str, Any]]:
        """Find the best fire target based on priority mode."""
        fire_threshold = 100.0
        fire_locations = np.where(fire_map >= fire_threshold)
        
        if len(fire_locations[0]) == 0:
            return None
        
        # Get targets of other drones if coordination is enabled
        other_targets = set()
        if self.coordination_enabled:
            for other_drone in other_drones:
                if (other_drone.assigned_fire_location and 
                    other_drone.drone_id != drone.drone_id):
                    other_targets.add(other_drone.assigned_fire_location)
        
        best_fire = None
        best_score = float('-inf')
        
        for i in range(len(fire_locations[0])):
            fire_row, fire_col = fire_locations[0][i], fire_locations[1][i]
            fire_intensity = float(fire_map[fire_row, fire_col])
            
            # Skip if another drone is already targeting this fire
            fire_latlon = self._grid_to_latlon(fire_row, fire_col, drone.position)
            if self.coordination_enabled and fire_latlon in other_targets:
                continue
            
            # Calculate priority score based on mode
            score = self._calculate_fire_priority(
                drone.position, (fire_row, fire_col), fire_intensity
            )
            
            if score > best_score:
                best_score = score
                best_fire = {
                    "location": (fire_row, fire_col),
                    "intensity": fire_intensity,
                    "score": score
                }
        
        return best_fire
    
    def _calculate_fire_priority(self, drone_pos: Tuple[float, float], 
                               fire_grid_pos: Tuple[int, int], intensity: float) -> float:
        """Calculate priority score for a fire based on policy mode."""
        
        # Convert grid position to lat/lon for distance calculation
        fire_latlon = self._grid_to_latlon(fire_grid_pos[0], fire_grid_pos[1], drone_pos)
        distance = self._calculate_distance(drone_pos, fire_latlon)
        
        if self.priority_mode == "nearest":
            # Prioritize by inverse distance (closer = higher priority)
            return 1.0 / (distance + 0.001)  # Avoid division by zero
        
        elif self.priority_mode == "highest_intensity":
            # Prioritize by fire intensity
            return intensity
        
        elif self.priority_mode == "combined":
            # Combine distance and intensity
            distance_score = 1.0 / (distance + 0.001)
            intensity_score = intensity / 1000.0  # Normalize intensity
            return 0.6 * distance_score + 0.4 * intensity_score
        
        else:
            return random.random()  # Fallback to random
    
    def _select_best_refill(self, drone: Drone, refill_locations: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Select nearest refill location."""
        if not refill_locations:
            return {"type": "idle", "details": {}}
        
        nearest_refill = min(refill_locations, 
                           key=lambda loc: self._calculate_distance(drone.position, loc))
        
        return {
            "type": "target_refill",
            "details": {"location": nearest_refill}
        }
    
    def _grid_to_latlon(self, row: int, col: int, reference_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Convert grid coordinates to approximate lat/lon."""
        # Simplified conversion - in practice would use proper projection
        grid_size = 100  # Assume 100x100 grid
        cell_size_degrees = 0.01  # Roughly 1km per cell
        
        # Use reference position as origin
        ref_lat, ref_lon = reference_pos
        
        lat = ref_lat + (row - grid_size/2) * cell_size_degrees
        lon = ref_lon + (col - grid_size/2) * cell_size_degrees
        
        return lat, lon
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate distance between two positions."""
        lat_diff = pos1[0] - pos2[0]
        lon_diff = pos1[1] - pos2[1]
        return np.sqrt(lat_diff**2 + lon_diff**2)


class CoordinatedPolicy(DronePolicy):
    """
    Coordinated policy that uses swarm intelligence principles.
    
    Implements coordination mechanisms like task allocation,
    load balancing, and communication-based decision making.
    """
    
    def __init__(self, coordination_radius: float = 0.05, 
                 communication_enabled: bool = True,
                 load_balancing: bool = True):
        """
        Initialize coordinated policy.
        
        Args:
            coordination_radius: Radius for local coordination (degrees)
            communication_enabled: Whether drones can communicate
            load_balancing: Whether to balance workload across drones
        """
        self.coordination_radius = coordination_radius
        self.communication_enabled = communication_enabled
        self.load_balancing = load_balancing
        
        # Shared knowledge (simulated swarm intelligence)
        self.fire_assignments: Dict[Tuple[int, int], List[str]] = {}  # fire -> [drone_ids]
        self.workload_tracker: Dict[str, int] = {}  # drone_id -> assigned_fires
        
        self.logger = logging.getLogger("CoordinatedPolicy")
    
    def select_action(self, drone: Drone, fire_map: np.ndarray, 
                     other_drones: List[Drone], refill_locations: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Select action using coordination principles."""
        
        # Handle non-idle states
        if drone.state != DroneState.IDLE:
            return {"type": "continue", "details": {}}
        
        # Resource management
        if drone.water_level < 0.15:
            return self._coordinated_refill_selection(drone, other_drones, refill_locations)
        
        if drone.battery_level < 0.25:
            return {"type": "return_to_base", "details": {}}
        
        # Update coordination information
        self._update_coordination_data(drone, other_drones)
        
        # Find coordinated fire target
        fire_target = self._find_coordinated_fire_target(drone, fire_map, other_drones)
        
        if fire_target:
            return {
                "type": "target_fire",
                "details": {
                    "grid_location": fire_target["location"],
                    "intensity": fire_target["intensity"],
                    "coordination_score": fire_target["score"],
                    "team_size": fire_target["team_size"]
                }
            }
        
        # No immediate fire target, perform patrol or assist others
        assist_action = self._find_assistance_opportunity(drone, other_drones)
        if assist_action:
            return assist_action
        
        return {"type": "patrol", "details": {}}
    
    def _update_coordination_data(self, drone: Drone, other_drones: List[Drone]):
        """Update shared coordination information."""
        # Clear old assignments
        self.fire_assignments.clear()
        self.workload_tracker.clear()
        
        # Collect current assignments from all drones
        all_drones = [drone] + other_drones
        
        for d in all_drones:
            # Track workload
            if d.assigned_fire_location:
                fire_grid = self._latlon_to_grid(d.assigned_fire_location[0], d.assigned_fire_location[1])
                
                if fire_grid not in self.fire_assignments:
                    self.fire_assignments[fire_grid] = []
                self.fire_assignments[fire_grid].append(d.drone_id)
                
                # Update workload
                if d.drone_id not in self.workload_tracker:
                    self.workload_tracker[d.drone_id] = 0
                self.workload_tracker[d.drone_id] += 1
    
    def _find_coordinated_fire_target(self, drone: Drone, fire_map: np.ndarray, 
                                    other_drones: List[Drone]) -> Optional[Dict[str, Any]]:
        """Find fire target using coordination principles."""
        fire_threshold = 100.0
        fire_locations = np.where(fire_map >= fire_threshold)
        
        if len(fire_locations[0]) == 0:
            return None
        
        best_target = None
        best_score = float('-inf')
        
        for i in range(len(fire_locations[0])):
            fire_row, fire_col = fire_locations[0][i], fire_locations[1][i]
            fire_intensity = float(fire_map[fire_row, fire_col])
            fire_pos = (fire_row, fire_col)
            
            # Calculate coordination score
            score = self._calculate_coordination_score(
                drone, fire_pos, fire_intensity, other_drones
            )
            
            if score > best_score:
                best_score = score
                
                # Calculate team size for this fire
                team_size = len(self.fire_assignments.get(fire_pos, [])) + 1
                
                best_target = {
                    "location": fire_pos,
                    "intensity": fire_intensity,
                    "score": score,
                    "team_size": team_size
                }
        
        return best_target
    
    def _calculate_coordination_score(self, drone: Drone, fire_pos: Tuple[int, int], 
                                    intensity: float, other_drones: List[Drone]) -> float:
        """Calculate coordination score for fire assignment."""
        fire_latlon = self._grid_to_latlon(fire_pos[0], fire_pos[1], drone.position)
        distance = self._calculate_distance(drone.position, fire_latlon)
        
        # Base score: intensity / distance
        base_score = intensity / (distance + 0.001)
        
        # Coordination factors
        
        # 1. Avoid over-assignment (too many drones on same fire)
        current_assignment_count = len(self.fire_assignments.get(fire_pos, []))
        optimal_team_size = max(1, int(intensity / 300))  # Larger fires need more drones
        
        if current_assignment_count >= optimal_team_size:
            over_assignment_penalty = 0.1  # Heavily penalize over-assignment
        else:
            over_assignment_penalty = 1.0
        
        # 2. Load balancing (prefer drones with lighter workload)
        current_workload = self.workload_tracker.get(drone.drone_id, 0)
        avg_workload = np.mean(list(self.workload_tracker.values())) if self.workload_tracker else 0
        
        if self.load_balancing:
            if current_workload <= avg_workload:
                load_balance_bonus = 1.2
            else:
                load_balance_bonus = 0.8
        else:
            load_balance_bonus = 1.0
        
        # 3. Communication network effect (prefer fires where teammates are nearby)
        nearby_teammates = 0
        if self.communication_enabled:
            for other_drone in other_drones:
                other_distance = self._calculate_distance(drone.position, other_drone.position)
                if other_distance <= self.coordination_radius:
                    nearby_teammates += 1
        
        communication_bonus = 1.0 + (nearby_teammates * 0.1)
        
        # 4. Fire urgency (spreading fires get higher priority)
        urgency_multiplier = min(2.0, intensity / 500.0)
        
        # Combine all factors
        final_score = (base_score * 
                      over_assignment_penalty * 
                      load_balance_bonus * 
                      communication_bonus * 
                      urgency_multiplier)
        
        return final_score
    
    def _coordinated_refill_selection(self, drone: Drone, other_drones: List[Drone], 
                                    refill_locations: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Select refill location with coordination to avoid congestion."""
        if not refill_locations:
            return {"type": "idle", "details": {}}
        
        # Count drones at each refill location
        refill_congestion = {loc: 0 for loc in refill_locations}
        
        for other_drone in other_drones:
            if (other_drone.state in [DroneState.FLYING_TO_REFILL, DroneState.REFILLING] and
                other_drone.assigned_refill_location):
                
                # Find closest refill location to assigned location
                closest_refill = min(refill_locations,
                                   key=lambda loc: self._calculate_distance(
                                       other_drone.assigned_refill_location, loc))
                refill_congestion[closest_refill] += 1
        
        # Select refill location balancing distance and congestion
        best_refill = None
        best_score = float('-inf')
        
        for refill_loc in refill_locations:
            distance = self._calculate_distance(drone.position, refill_loc)
            congestion = refill_congestion[refill_loc]
            
            # Score: lower distance and lower congestion are better
            distance_score = 1.0 / (distance + 0.001)
            congestion_penalty = 1.0 / (congestion + 1)
            
            score = distance_score * congestion_penalty
            
            if score > best_score:
                best_score = score
                best_refill = refill_loc
        
        return {
            "type": "target_refill",
            "details": {
                "location": best_refill,
                "expected_congestion": refill_congestion[best_refill]
            }
        }
    
    def _find_assistance_opportunity(self, drone: Drone, other_drones: List[Drone]) -> Optional[Dict[str, Any]]:
        """Find opportunity to assist other drones."""
        
        # Look for drones that need assistance
        for other_drone in other_drones:
            # Help drones that are out of battery
            if other_drone.state == DroneState.OUT_OF_BATTERY:
                distance = self._calculate_distance(drone.position, other_drone.position)
                if distance <= self.coordination_radius:
                    return {
                        "type": "assist_drone",
                        "details": {
                            "target_drone": other_drone.drone_id,
                            "assistance_type": "battery_rescue",
                            "location": other_drone.position
                        }
                    }
            
            # Help with large fires (join existing fire suppression)
            if (other_drone.assigned_fire_location and 
                other_drone.state in [DroneState.FLYING_TO_FIRE, DroneState.DROPPING_WATER]):
                
                fire_grid = self._latlon_to_grid(
                    other_drone.assigned_fire_location[0], 
                    other_drone.assigned_fire_location[1]
                )
                
                # Join if fire has high intensity and not too many drones already
                current_team_size = len(self.fire_assignments.get(fire_grid, []))
                if current_team_size < 3:  # Max 3 drones per fire
                    return {
                        "type": "target_fire",
                        "details": {
                            "grid_location": fire_grid,
                            "assistance_mode": True,
                            "team_leader": other_drone.drone_id
                        }
                    }
        
        return None
    
    def _grid_to_latlon(self, row: int, col: int, reference_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Convert grid coordinates to lat/lon."""
        # Simplified conversion
        grid_size = 100
        cell_size_degrees = 0.01
        
        ref_lat, ref_lon = reference_pos
        lat = ref_lat + (row - grid_size/2) * cell_size_degrees
        lon = ref_lon + (col - grid_size/2) * cell_size_degrees
        
        return lat, lon
    
    def _latlon_to_grid(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert lat/lon to grid coordinates."""
        # Simplified conversion
        grid_size = 100
        cell_size_degrees = 0.01
        
        # Assume grid centered at (34.5, -118.5)
        center_lat, center_lon = 34.5, -118.5
        
        row = int((lat - center_lat) / cell_size_degrees + grid_size/2)
        col = int((lon - center_lon) / cell_size_degrees + grid_size/2)
        
        # Clamp to valid range
        row = max(0, min(row, grid_size - 1))
        col = max(0, min(col, grid_size - 1))
        
        return row, col
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate distance between two positions."""
        lat_diff = pos1[0] - pos2[0]
        lon_diff = pos1[1] - pos2[1]
        return np.sqrt(lat_diff**2 + lon_diff**2) 