"""
Drone swarm management for coordinated wildfire suppression.

This module provides the DroneSwarm class for managing large numbers
of autonomous drones with efficient coordination protocols.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict

from .drone import Drone, DroneState, DroneCapabilities


class DroneSwarm:
    """
    Manages a large swarm of firefighting drones (5,000-10,000 units).
    
    Provides:
    - Efficient swarm coordination algorithms
    - Load balancing for fire suppression
    - Communication network simulation
    - Performance monitoring and optimization
    """
    
    def __init__(self, 
                 swarm_size: int = 5000,
                 deployment_area: Tuple[float, float, float, float] = (34.0, -119.0, 35.0, -118.0),
                 refill_locations: List[Tuple[float, float]] = None,
                 drone_capabilities: Optional[DroneCapabilities] = None):
        """
        Initialize drone swarm.
        
        Args:
            swarm_size: Number of drones in the swarm (5,000-10,000)
            deployment_area: (min_lat, min_lon, max_lat, max_lon) coverage area
            refill_locations: List of water refill points (lakes)
            drone_capabilities: Default capabilities for all drones
        """
        self.swarm_size = swarm_size
        self.deployment_area = deployment_area
        self.refill_locations = refill_locations or self._generate_default_refill_locations()
        self.default_capabilities = drone_capabilities or DroneCapabilities()
        
        # Create drone swarm
        self.drones: Dict[str, Drone] = {}
        self._initialize_swarm()
        
        # Coordination data structures
        self.active_fires: Dict[Tuple[int, int], float] = {}  # (row, col) -> intensity
        self.fire_assignments: Dict[str, Tuple[int, int]] = {}  # drone_id -> fire_location
        self.refill_queues: Dict[Tuple[float, float], List[str]] = defaultdict(list)
        
        # Performance tracking
        self.start_time = datetime.now()
        self.total_water_dropped = 0.0
        self.total_fires_suppressed = 0
        self.coordination_messages_sent = 0
        
        # Threading for parallel updates
        self.max_workers = min(32, max(4, swarm_size // 200))  # Scale workers with swarm size
        self.update_lock = threading.Lock()
        
        # Communication network simulation
        self.communication_range_m = self.default_capabilities.communication_range_m
        self.network_topology: Dict[str, List[str]] = {}  # drone_id -> [connected_drone_ids]
        
        # Logging
        self.logger = logging.getLogger("DroneSwarm")
        self.logger.info(f"Initialized swarm with {swarm_size} drones")
        
        # Task assignment strategies
        self.assignment_strategy = "greedy"  # "greedy", "auction", "clusters"
        self._update_network_topology()
    
    def _initialize_swarm(self):
        """Create and deploy initial drone swarm."""
        min_lat, min_lon, max_lat, max_lon = self.deployment_area
        
        # Generate initial positions in a grid pattern for coverage
        grid_size = int(np.ceil(np.sqrt(self.swarm_size)))
        lat_step = (max_lat - min_lat) / grid_size
        lon_step = (max_lon - min_lon) / grid_size
        
        drone_count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if drone_count >= self.swarm_size:
                    break
                
                # Calculate position with some randomization
                base_lat = min_lat + i * lat_step + lat_step * 0.1 * np.random.random()
                base_lon = min_lon + j * lon_step + lon_step * 0.1 * np.random.random()
                
                drone_id = f"drone_{drone_count:05d}"
                drone = Drone(
                    drone_id=drone_id,
                    initial_position=(base_lat, base_lon),
                    capabilities=self.default_capabilities
                )
                
                self.drones[drone_id] = drone
                drone_count += 1
            
            if drone_count >= self.swarm_size:
                break
    
    def _generate_default_refill_locations(self) -> List[Tuple[float, float]]:
        """Generate default water refill locations (simulated lakes)."""
        min_lat, min_lon, max_lat, max_lon = self.deployment_area
        
        # Place refill stations roughly every 10km
        num_stations = max(4, int((max_lat - min_lat) * (max_lon - min_lon) * 111000 / 10000))
        
        refill_points = []
        for i in range(num_stations):
            lat = min_lat + (max_lat - min_lat) * np.random.random()
            lon = min_lon + (max_lon - min_lon) * np.random.random()
            refill_points.append((lat, lon))
        
        return refill_points
    
    def update_swarm(self, dt_seconds: float, fire_state: Any) -> Dict[str, Any]:
        """
        Update entire swarm for one time step using parallel processing.
        
        Args:
            dt_seconds: Time step in seconds
            fire_state: Current fire state from simulation
            
        Returns:
            Swarm-level statistics and actions
        """
        # Extract fire map for drone decisions
        fire_map = fire_state.current_flame_front.heat_intensity
        
        # Update active fires and assignments
        self._update_fire_tracking(fire_map)
        
        # Parallel drone updates
        swarm_actions = self._parallel_drone_updates(dt_seconds, fire_map)
        
        # Coordinate swarm behavior
        coordination_actions = self._coordinate_swarm()
        
        # Update communication network
        if len(swarm_actions) % 100 == 0:  # Every 100 steps
            self._update_network_topology()
        
        # Collect performance metrics
        metrics = self._collect_swarm_metrics()
        
        return {
            'swarm_actions': swarm_actions,
            'coordination_actions': coordination_actions,
            'metrics': metrics,
            'active_fires': len(self.active_fires),
            'operational_drones': sum(1 for d in self.drones.values() if d.is_operational())
        }
    
    def _parallel_drone_updates(self, dt_seconds: float, fire_map: np.ndarray) -> List[Dict[str, Any]]:
        """Update all drones in parallel for performance."""
        all_actions = []
        
        # Split drones into chunks for parallel processing
        drone_items = list(self.drones.items())
        chunk_size = max(1, len(drone_items) // self.max_workers)
        drone_chunks = [drone_items[i:i + chunk_size] for i in range(0, len(drone_items), chunk_size)]
        
        def update_drone_chunk(chunk):
            chunk_actions = []
            for drone_id, drone in chunk:
                try:
                    action = drone.update(dt_seconds, fire_map, self.refill_locations)
                    action['drone_id'] = drone_id
                    chunk_actions.append(action)
                    
                    # Apply water drops to fire simulation
                    if action['type'] == 'dropping_water':
                        self._apply_water_drop_to_fire(drone, action, fire_map)
                        
                except Exception as e:
                    self.logger.error(f"Error updating drone {drone_id}: {e}")
                    
            return chunk_actions
        
        # Execute parallel updates
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(update_drone_chunk, chunk) for chunk in drone_chunks]
            
            for future in as_completed(futures):
                try:
                    chunk_actions = future.result()
                    all_actions.extend(chunk_actions)
                except Exception as e:
                    self.logger.error(f"Error in parallel drone update: {e}")
        
        return all_actions
    
    def _update_fire_tracking(self, fire_map: np.ndarray):
        """Update tracking of active fires for assignment."""
        with self.update_lock:
            # Clear old fire data
            self.active_fires.clear()
            
            # Find active fire cells
            threshold = 100.0  # kW/m² threshold for active fire
            fire_coords = np.where(fire_map >= threshold)
            
            for row, col in zip(fire_coords[0], fire_coords[1]):
                self.active_fires[(row, col)] = float(fire_map[row, col])
    
    def _coordinate_swarm(self) -> List[Dict[str, Any]]:
        """Coordinate swarm behavior for optimal fire suppression."""
        coordination_actions = []
        
        # Reassign drones to fires based on strategy
        if self.assignment_strategy == "greedy":
            coordination_actions.extend(self._greedy_fire_assignment())
        elif self.assignment_strategy == "auction":
            coordination_actions.extend(self._auction_fire_assignment())
        elif self.assignment_strategy == "clusters":
            coordination_actions.extend(self._cluster_fire_assignment())
        
        # Manage refill queues
        coordination_actions.extend(self._manage_refill_queues())
        
        # Emergency redeployment
        coordination_actions.extend(self._emergency_redeployment())
        
        return coordination_actions
    
    def _greedy_fire_assignment(self) -> List[Dict[str, Any]]:
        """Simple greedy assignment of drones to nearest fires."""
        actions = []
        
        # Get available drones (idle with water)
        available_drones = [
            (drone_id, drone) for drone_id, drone in self.drones.items()
            if (drone.state == DroneState.IDLE and 
                drone.water_level > 0.5 and 
                drone.battery_level > 0.3 and
                drone_id not in self.fire_assignments)
        ]
        
        # Get unassigned fires (sorted by intensity)
        assigned_fires = set(self.fire_assignments.values())
        unassigned_fires = [
            (location, intensity) for location, intensity in self.active_fires.items()
            if location not in assigned_fires
        ]
        unassigned_fires.sort(key=lambda x: x[1], reverse=True)  # Highest intensity first
        
        # Assign drones to fires greedily
        for fire_location, fire_intensity in unassigned_fires:
            if not available_drones:
                break
            
            # Find closest available drone
            best_drone = None
            best_distance = float('inf')
            
            for drone_id, drone in available_drones:
                distance = self._calculate_distance(drone.position, fire_location)
                if distance < best_distance:
                    best_distance = distance
                    best_drone = (drone_id, drone)
            
            if best_drone:
                drone_id, drone = best_drone
                
                # Assign drone to fire
                self.fire_assignments[drone_id] = fire_location
                available_drones.remove(best_drone)
                
                # Send assignment message
                fire_lat, fire_lon = self._grid_to_latlon(fire_location[0], fire_location[1])
                drone._start_fire_mission((fire_lat, fire_lon))
                
                actions.append({
                    'type': 'fire_assignment',
                    'drone_id': drone_id,
                    'fire_location': fire_location,
                    'fire_intensity': fire_intensity,
                    'distance': best_distance
                })
                
                self.coordination_messages_sent += 1
        
        return actions
    
    def _auction_fire_assignment(self) -> List[Dict[str, Any]]:
        """Market-based auction assignment for better optimization."""
        # Simplified auction algorithm
        actions = []
        
        # This would implement a full auction protocol
        # For now, fall back to greedy
        return self._greedy_fire_assignment()
    
    def _cluster_fire_assignment(self) -> List[Dict[str, Any]]:
        """Cluster-based assignment for handling large fire complexes."""
        actions = []
        
        # Cluster nearby fires and assign drone groups
        fire_clusters = self._cluster_fires()
        
        for cluster_id, cluster_fires in fire_clusters.items():
            cluster_drones = self._get_drones_for_cluster(cluster_fires)
            
            # Assign drones within cluster
            for drone_id in cluster_drones:
                if cluster_fires and drone_id not in self.fire_assignments:
                    fire_location = cluster_fires.pop(0)  # Take first fire
                    self.fire_assignments[drone_id] = fire_location
                    
                    actions.append({
                        'type': 'cluster_assignment',
                        'drone_id': drone_id,
                        'cluster_id': cluster_id,
                        'fire_location': fire_location
                    })
        
        return actions
    
    def _manage_refill_queues(self) -> List[Dict[str, Any]]:
        """Manage queues at refill stations to prevent congestion."""
        actions = []
        
        # Clear old queues
        self.refill_queues.clear()
        
        # Find drones heading to refill
        for drone_id, drone in self.drones.items():
            if (drone.state == DroneState.FLYING_TO_REFILL and 
                drone.assigned_refill_location):
                self.refill_queues[drone.assigned_refill_location].append(drone_id)
        
        # Redistribute if queues are too long
        max_queue_length = 10
        for location, queue in self.refill_queues.items():
            if len(queue) > max_queue_length:
                # Redirect some drones to other locations
                excess_drones = queue[max_queue_length:]
                
                for drone_id in excess_drones:
                    drone = self.drones[drone_id]
                    
                    # Find alternative refill location
                    alternative_locations = [
                        loc for loc in self.refill_locations 
                        if loc != location and len(self.refill_queues[loc]) < max_queue_length
                    ]
                    
                    if alternative_locations:
                        new_location = min(alternative_locations,
                                         key=lambda loc: self._calculate_distance(drone.position, loc))
                        
                        drone._start_refill_mission([new_location])
                        
                        actions.append({
                            'type': 'refill_redirect',
                            'drone_id': drone_id,
                            'from_location': location,
                            'to_location': new_location,
                            'queue_length': len(queue)
                        })
        
        return actions
    
    def _emergency_redeployment(self) -> List[Dict[str, Any]]:
        """Handle emergency situations requiring drone redeployment."""
        actions = []
        
        # Find drones that are out of battery or stuck
        emergency_drones = [
            (drone_id, drone) for drone_id, drone in self.drones.items()
            if drone.state == DroneState.OUT_OF_BATTERY
        ]
        
        # For each emergency drone, send nearby drones to assist
        for drone_id, emergency_drone in emergency_drones:
            # Find nearby operational drones
            nearby_drones = self._find_nearby_drones(
                emergency_drone.position, 
                max_distance=0.1,  # degrees
                filter_operational=True
            )
            
            if nearby_drones:
                # Send one drone to assist
                helper_drone_id = nearby_drones[0]
                actions.append({
                    'type': 'emergency_assistance',
                    'emergency_drone': drone_id,
                    'helper_drone': helper_drone_id,
                    'location': emergency_drone.position
                })
        
        return actions
    
    def _apply_water_drop_to_fire(self, drone: Drone, action: Dict[str, Any], fire_map: np.ndarray):
        """Apply water drop effects to the fire simulation."""
        if action['type'] == 'dropping_water':
            position = action['details']['position']
            water_amount = action['details']['water_dropped']
            
            # Convert position to grid coordinates
            row, col = self._latlon_to_grid(position[0], position[1], fire_map.shape)
            
            # This would call the fire simulator's apply_water_drop method
            # For now, just update our tracking
            if (row, col) in self.active_fires:
                suppression_effect = water_amount * 50  # Simplified suppression
                self.active_fires[(row, col)] = max(0, self.active_fires[(row, col)] - suppression_effect)
                
                if self.active_fires[(row, col)] < 50:  # Fire suppressed
                    del self.active_fires[(row, col)]
                    self.total_fires_suppressed += 1
            
            self.total_water_dropped += water_amount
    
    def _update_network_topology(self):
        """Update communication network topology between drones."""
        self.network_topology.clear()
        
        # For large swarms, use efficient spatial indexing
        drone_positions = {
            drone_id: drone.position 
            for drone_id, drone in self.drones.items()
        }
        
        # Build communication graph
        for drone_id, position in drone_positions.items():
            connected_drones = []
            
            for other_id, other_position in drone_positions.items():
                if drone_id != other_id:
                    distance = self._calculate_distance(position, other_position)
                    distance_m = distance * 111000  # Convert degrees to meters (rough)
                    
                    if distance_m <= self.communication_range_m:
                        connected_drones.append(other_id)
            
            self.network_topology[drone_id] = connected_drones
    
    def _collect_swarm_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive swarm performance metrics."""
        operational_drones = sum(1 for d in self.drones.values() if d.is_operational())
        
        # Battery statistics
        battery_levels = [d.battery_level for d in self.drones.values()]
        water_levels = [d.water_level for d in self.drones.values()]
        
        # State distribution
        state_counts = defaultdict(int)
        for drone in self.drones.values():
            state_counts[drone.state.value] += 1
        
        # Mission statistics
        total_missions = sum(d.missions_completed for d in self.drones.values())
        total_flight_time = sum(d.total_flight_time for d in self.drones.values())
        
        return {
            'swarm_size': self.swarm_size,
            'operational_drones': operational_drones,
            'operational_percentage': operational_drones / self.swarm_size * 100,
            'average_battery': np.mean(battery_levels),
            'average_water': np.mean(water_levels),
            'state_distribution': dict(state_counts),
            'total_missions_completed': total_missions,
            'total_flight_time_hours': total_flight_time / 60,
            'total_water_dropped_liters': self.total_water_dropped,
            'total_fires_suppressed': self.total_fires_suppressed,
            'active_fire_assignments': len(self.fire_assignments),
            'coordination_messages': self.coordination_messages_sent,
            'communication_network_density': len(self.network_topology) / max(1, self.swarm_size),
            'average_connections_per_drone': np.mean([len(connections) for connections in self.network_topology.values()]) if self.network_topology else 0
        }
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate distance between two positions."""
        if isinstance(pos2, tuple) and len(pos2) == 2 and isinstance(pos2[0], (int, np.integer)):
            # Grid coordinates, convert to lat/lon
            pos2 = self._grid_to_latlon(pos2[0], pos2[1])
        
        lat_diff = pos1[0] - pos2[0]
        lon_diff = pos1[1] - pos2[1]
        return np.sqrt(lat_diff**2 + lon_diff**2)
    
    def _grid_to_latlon(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid coordinates to lat/lon."""
        min_lat, min_lon, max_lat, max_lon = self.deployment_area
        
        # Assume 100x100 grid for simplicity
        grid_size = 100
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        
        lat = min_lat + (row / grid_size) * lat_range
        lon = min_lon + (col / grid_size) * lon_range
        
        return lat, lon
    
    def _latlon_to_grid(self, lat: float, lon: float, fire_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Convert lat/lon to grid coordinates."""
        min_lat, min_lon, max_lat, max_lon = self.deployment_area
        
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        
        row = int(((lat - min_lat) / lat_range) * fire_shape[0])
        col = int(((lon - min_lon) / lon_range) * fire_shape[1])
        
        # Clamp to valid range
        row = max(0, min(row, fire_shape[0] - 1))
        col = max(0, min(col, fire_shape[1] - 1))
        
        return row, col
    
    def _cluster_fires(self) -> Dict[int, List[Tuple[int, int]]]:
        """Cluster nearby fires for coordinated response."""
        # Simple clustering based on distance
        clusters = {}
        cluster_id = 0
        
        unassigned_fires = list(self.active_fires.keys())
        
        while unassigned_fires:
            # Start new cluster with first unassigned fire
            cluster_center = unassigned_fires.pop(0)
            clusters[cluster_id] = [cluster_center]
            
            # Find nearby fires within cluster radius
            cluster_radius = 5  # grid cells
            
            i = 0
            while i < len(unassigned_fires):
                fire_location = unassigned_fires[i]
                
                # Check distance to any fire in current cluster
                min_distance = min(
                    abs(fire_location[0] - cf[0]) + abs(fire_location[1] - cf[1])
                    for cf in clusters[cluster_id]
                )
                
                if min_distance <= cluster_radius:
                    clusters[cluster_id].append(fire_location)
                    unassigned_fires.pop(i)
                else:
                    i += 1
            
            cluster_id += 1
        
        return clusters
    
    def _get_drones_for_cluster(self, cluster_fires: List[Tuple[int, int]]) -> List[str]:
        """Get appropriate drones for a fire cluster."""
        if not cluster_fires:
            return []
        
        # Calculate cluster center
        cluster_center_row = np.mean([f[0] for f in cluster_fires])
        cluster_center_col = np.mean([f[1] for f in cluster_fires])
        cluster_center_latlon = self._grid_to_latlon(int(cluster_center_row), int(cluster_center_col))
        
        # Find nearby available drones
        available_drones = [
            (drone_id, drone) for drone_id, drone in self.drones.items()
            if (drone.state == DroneState.IDLE and 
                drone.water_level > 0.5 and 
                drone.battery_level > 0.3)
        ]
        
        # Sort by distance to cluster center
        available_drones.sort(
            key=lambda x: self._calculate_distance(x[1].position, cluster_center_latlon)
        )
        
        # Return drone IDs for cluster (limit to cluster size * 2)
        max_drones = min(len(available_drones), len(cluster_fires) * 2)
        return [drone_id for drone_id, _ in available_drones[:max_drones]]
    
    def _find_nearby_drones(self, position: Tuple[float, float], 
                           max_distance: float, filter_operational: bool = False) -> List[str]:
        """Find drones within specified distance of a position."""
        nearby_drones = []
        
        for drone_id, drone in self.drones.items():
            if filter_operational and not drone.is_operational():
                continue
            
            distance = self._calculate_distance(position, drone.position)
            if distance <= max_distance:
                nearby_drones.append(drone_id)
        
        return nearby_drones
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get comprehensive swarm status for monitoring."""
        return {
            'timestamp': datetime.now(),
            'runtime_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
            'metrics': self._collect_swarm_metrics(),
            'active_fires': len(self.active_fires),
            'fire_assignments': len(self.fire_assignments),
            'refill_locations': len(self.refill_locations),
            'network_connected_components': self._count_network_components()
        }
    
    def _count_network_components(self) -> int:
        """Count connected components in communication network."""
        if not self.network_topology:
            return 0
        
        visited = set()
        components = 0
        
        def dfs(drone_id):
            if drone_id in visited:
                return
            visited.add(drone_id)
            for neighbor in self.network_topology.get(drone_id, []):
                dfs(neighbor)
        
        for drone_id in self.drones.keys():
            if drone_id not in visited:
                dfs(drone_id)
                components += 1
        
        return components 