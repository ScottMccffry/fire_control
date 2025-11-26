"""
Gymnasium environment for drone swarm wildfire control.

This module provides both single-agent and multi-agent reinforcement learning
environments for training drone swarm fire suppression strategies.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.mock_fire import MockFireSimulator
from drones.drone import Drone, DroneState, DroneCapabilities
from drones.swarm import DroneSwarm
from config import Config, load_config


class FireControlEnv(gym.Env):
    """
    Single-agent Gymnasium environment for drone swarm fire control.

    The agent controls the high-level strategy for the entire swarm,
    making decisions about resource allocation and coordination.

    Observation Space:
        - Fire grid (normalized heat intensity)
        - Drone positions and states
        - Resource levels (battery, water)
        - Wind conditions

    Action Space:
        - Discrete: Select coordination strategy and target areas
        - Or Continuous: Resource allocation weights

    Rewards:
        - Positive: Fire cells suppressed, area protected
        - Negative: Drones lost, fire spread, inefficiency
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(
        self,
        config: Optional[Config] = None,
        render_mode: Optional[str] = None,
        swarm_size: int = 50,
        grid_size: Tuple[int, int] = (50, 50),
        max_episode_steps: int = 200,
        discrete_actions: bool = True,
        num_target_zones: int = 9,
        normalize_obs: bool = True,
    ):
        """
        Initialize Fire Control environment.

        Args:
            config: Configuration object (uses defaults if None)
            render_mode: Rendering mode ('human' or 'rgb_array')
            swarm_size: Number of drones in swarm
            grid_size: Fire simulation grid size
            max_episode_steps: Maximum steps per episode
            discrete_actions: Use discrete action space
            num_target_zones: Number of target zones for discrete actions
            normalize_obs: Normalize observations to [0, 1]
        """
        super().__init__()

        self.config = config or load_config()
        self.render_mode = render_mode
        self.swarm_size = swarm_size
        self.grid_size = grid_size
        self.max_episode_steps = max_episode_steps
        self.discrete_actions = discrete_actions
        self.num_target_zones = num_target_zones
        self.normalize_obs = normalize_obs

        self.logger = logging.getLogger(__name__)

        # Fire bounds (California region by default)
        self.bounds = tuple(self.config.fire.bounds)

        # Time step in seconds (5 minutes default)
        self.dt_seconds = 300.0

        # Define observation space
        self._define_observation_space()

        # Define action space
        self._define_action_space()

        # Reward configuration
        self.reward_config = self.config.rewards

        # Initialize components (will be created in reset)
        self.fire_sim: Optional[MockFireSimulator] = None
        self.swarm: Optional[DroneSwarm] = None

        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.prev_burned_area = 0
        self.prev_active_fires = 0

        # Metrics
        self.episode_metrics = defaultdict(float)

    def _define_observation_space(self):
        """Define the observation space."""
        # Observations:
        # 1. Fire grid (downsampled): grid_size flattened
        # 2. Drone density grid: grid_size flattened (how many drones in each zone)
        # 3. Drone resource summary: [avg_battery, avg_water, operational_ratio]
        # 4. Wind: [speed_normalized, direction_sin, direction_cos]
        # 5. Global state: [burned_ratio, active_fire_ratio, step_normalized]

        fire_obs_dim = self.grid_size[0] * self.grid_size[1]
        drone_density_dim = self.num_target_zones  # Simplified drone density per zone
        resource_dim = 3  # avg_battery, avg_water, operational_ratio
        wind_dim = 3  # speed, sin(dir), cos(dir)
        global_dim = 3  # burned_ratio, active_ratio, step_ratio

        total_dim = fire_obs_dim + drone_density_dim + resource_dim + wind_dim + global_dim

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(total_dim,),
            dtype=np.float32
        )

        # Store dimensions for observation construction
        self.obs_dims = {
            'fire': fire_obs_dim,
            'density': drone_density_dim,
            'resource': resource_dim,
            'wind': wind_dim,
            'global': global_dim
        }

    def _define_action_space(self):
        """Define the action space."""
        if self.discrete_actions:
            # Discrete actions:
            # - 0-8: Target zone (3x3 grid) for drone focus
            # - 9-11: Coordination strategy (greedy, balanced, defensive)
            # - 12-14: Resource priority (fire suppression, refill, balanced)
            # Combined: 9 zones * 3 strategies * 3 priorities = 81 actions
            # Simplified: just zone selection + strategy = 9 * 3 = 27
            self.action_space = spaces.Discrete(self.num_target_zones * 3)
        else:
            # Continuous actions:
            # - [0-8]: Attention weights for each zone (softmax applied)
            # - [9]: Aggression (0=defensive, 1=aggressive)
            # - [10]: Coordination mode (0=distributed, 1=concentrated)
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_target_zones + 2,),
                dtype=np.float32
            )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        # Create new fire simulator
        self.fire_sim = MockFireSimulator(
            grid_size=self.grid_size,
            cell_size_meters=self.config.fire.cell_size_meters,
            bounds=self.bounds
        )

        # Set random ignition points (1-3 fires)
        num_ignitions = np.random.randint(1, 4)
        ignition_points = []
        for _ in range(num_ignitions):
            row = np.random.randint(self.grid_size[0] // 4, 3 * self.grid_size[0] // 4)
            col = np.random.randint(self.grid_size[1] // 4, 3 * self.grid_size[1] // 4)
            ignition_points.append((row, col))
        self.fire_sim.set_ignition_points(ignition_points)

        # Set random weather
        wind_speed = np.random.uniform(2.0, 15.0)
        wind_direction = np.random.uniform(0.0, 360.0)
        self.fire_sim.set_weather(
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            temperature=np.random.uniform(20.0, 40.0),
            humidity=np.random.uniform(0.1, 0.5)
        )

        # Create drone swarm
        drone_capabilities = DroneCapabilities(
            max_flight_time_hours=self.config.drone.max_flight_time_hours,
            water_capacity_liters=self.config.drone.water_capacity_liters,
            max_speed_ms=self.config.drone.max_speed_mps,
            communication_range_m=self.config.drone.communication_range_m,
        )

        self.swarm = DroneSwarm(
            swarm_size=self.swarm_size,
            deployment_area=self.bounds,
            drone_capabilities=drone_capabilities
        )

        # Reset tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.prev_burned_area = 0
        self.prev_active_fires = len(self.fire_sim.get_active_fire_cells())
        self.episode_metrics = defaultdict(float)

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        self.current_step += 1

        # Parse action
        target_zone, strategy = self._parse_action(action)

        # Apply action to swarm coordination
        self._apply_action(target_zone, strategy)

        # Step fire simulation
        fire_state = self.fire_sim.step()

        # Update drone swarm
        swarm_result = self.swarm.update_swarm(self.dt_seconds, fire_state)

        # Apply water drops from drones to fire
        self._apply_drone_water_drops()

        # Calculate reward
        reward = self._calculate_reward(swarm_result)
        self.episode_reward += reward

        # Check termination
        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_episode_steps

        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        info['episode_reward'] = self.episode_reward

        return obs, reward, terminated, truncated, info

    def _parse_action(self, action: Union[int, np.ndarray]) -> Tuple[int, str]:
        """Parse action into target zone and strategy."""
        if self.discrete_actions:
            # action = zone * 3 + strategy_idx
            zone = action // 3
            strategy_idx = action % 3
            strategies = ['greedy', 'coordinated', 'defensive']
            return zone, strategies[strategy_idx]
        else:
            # Continuous: zone weights + parameters
            zone_weights = action[:self.num_target_zones]
            zone = int(np.argmax(zone_weights))
            aggression = action[self.num_target_zones] if len(action) > self.num_target_zones else 0.5

            if aggression < 0.33:
                strategy = 'defensive'
            elif aggression < 0.66:
                strategy = 'coordinated'
            else:
                strategy = 'greedy'

            return zone, strategy

    def _apply_action(self, target_zone: int, strategy: str):
        """Apply action to modify swarm behavior."""
        # Set coordination strategy
        self.swarm.assignment_strategy = strategy if strategy != 'defensive' else 'greedy'

        # Calculate zone bounds
        zone_row = target_zone // 3
        zone_col = target_zone % 3

        rows, cols = self.grid_size
        zone_row_start = zone_row * (rows // 3)
        zone_row_end = (zone_row + 1) * (rows // 3)
        zone_col_start = zone_col * (cols // 3)
        zone_col_end = (zone_col + 1) * (cols // 3)

        # Find fires in target zone and prioritize them
        priority_fires = []
        for (r, c), intensity in self.swarm.active_fires.items():
            if zone_row_start <= r < zone_row_end and zone_col_start <= c < zone_col_end:
                priority_fires.append(((r, c), intensity))

        # Update fire assignments to focus on priority zone
        if priority_fires and strategy != 'defensive':
            priority_fires.sort(key=lambda x: x[1], reverse=True)

            # Get available drones
            available_drones = [
                (drone_id, drone) for drone_id, drone in self.swarm.drones.items()
                if (drone.state == DroneState.IDLE and
                    drone.water_level > 0.3 and
                    drone.battery_level > 0.25)
            ]

            # Assign some drones to priority zone
            num_to_assign = min(len(available_drones) // 2, len(priority_fires) * 2)

            for i in range(num_to_assign):
                if i >= len(available_drones) or i >= len(priority_fires):
                    break

                drone_id, drone = available_drones[i]
                fire_loc, _ = priority_fires[i % len(priority_fires)]

                fire_lat, fire_lon = self.swarm._grid_to_latlon(fire_loc[0], fire_loc[1])
                drone._start_fire_mission((fire_lat, fire_lon))
                self.swarm.fire_assignments[drone_id] = fire_loc

    def _apply_drone_water_drops(self):
        """Apply water drops from drones to fire simulation."""
        for drone_id, drone in self.swarm.drones.items():
            if drone.state == DroneState.DROPPING_WATER:
                # Convert drone position to grid
                row, col = self.fire_sim._latlon_to_grid(
                    drone.position[0], drone.position[1]
                )

                # Apply water drop
                water_amount = drone.capabilities.water_drop_rate_lps * (self.dt_seconds / 60)
                self.fire_sim.apply_water_drop(row, col, water_amount)

    def _calculate_reward(self, swarm_result: Dict[str, Any]) -> float:
        """Calculate reward for current step."""
        reward = 0.0

        # Get current fire state
        current_burned = np.sum(self.fire_sim.burned_area)
        current_active = len(self.fire_sim.get_active_fire_cells())

        # Reward for suppressing fires
        fires_suppressed = max(0, self.prev_active_fires - current_active)
        reward += fires_suppressed * self.reward_config.fire_suppressed_cell

        # Penalty for fire spread
        new_burned = current_burned - self.prev_burned_area
        reward += new_burned * self.reward_config.fire_spread_penalty

        # Reward for water dropped effectively
        water_dropped = swarm_result['metrics'].get('total_water_dropped_liters', 0)
        if water_dropped > 0 and fires_suppressed > 0:
            efficiency = fires_suppressed / (water_dropped + 1)
            reward += efficiency * self.reward_config.water_efficiency_bonus

        # Penalty for lost drones
        operational = swarm_result['operational_drones']
        lost_drones = self.swarm_size - operational
        reward += lost_drones * self.reward_config.drone_lost_penalty / self.max_episode_steps

        # Small penalty for idle drones when there are active fires
        if current_active > 0:
            idle_count = sum(1 for d in self.swarm.drones.values()
                            if d.state == DroneState.IDLE)
            reward += idle_count * self.reward_config.idle_drone_penalty

        # Update tracking
        self.prev_burned_area = current_burned
        self.prev_active_fires = current_active

        # Track metrics
        self.episode_metrics['total_reward'] += reward
        self.episode_metrics['fires_suppressed'] += fires_suppressed
        self.episode_metrics['new_burned'] += new_burned

        return reward

    def _check_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if all fire is out
        if len(self.fire_sim.get_active_fire_cells()) == 0 and np.sum(self.fire_sim.burned_area) > 0:
            return True

        # Terminate if fire covers too much area
        total_cells = self.grid_size[0] * self.grid_size[1]
        burned_ratio = np.sum(self.fire_sim.burned_area) / total_cells
        if burned_ratio > 0.8:
            return True

        # Terminate if all drones are lost
        operational = sum(1 for d in self.swarm.drones.values() if d.is_operational())
        if operational == 0:
            return True

        return False

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        obs_parts = []

        # 1. Fire grid (normalized)
        fire_obs = self.fire_sim.heat_intensity.flatten().copy()
        if self.normalize_obs:
            fire_obs = fire_obs / (self.config.fire.max_heat_intensity_kw + 1e-8)
        obs_parts.append(fire_obs.astype(np.float32))

        # 2. Drone density per zone
        density = np.zeros(self.num_target_zones, dtype=np.float32)
        rows, cols = self.grid_size
        zone_h, zone_w = rows // 3, cols // 3

        for drone in self.swarm.drones.values():
            if drone.is_operational():
                r, c = self.fire_sim._latlon_to_grid(drone.position[0], drone.position[1])
                zone_r = min(2, r // zone_h)
                zone_c = min(2, c // zone_w)
                zone_idx = zone_r * 3 + zone_c
                if 0 <= zone_idx < self.num_target_zones:
                    density[zone_idx] += 1

        if self.normalize_obs:
            density = density / (self.swarm_size + 1e-8)
        obs_parts.append(density)

        # 3. Resource summary
        batteries = [d.battery_level for d in self.swarm.drones.values()]
        waters = [d.water_level for d in self.swarm.drones.values()]
        operational = sum(1 for d in self.swarm.drones.values() if d.is_operational())

        resource_obs = np.array([
            np.mean(batteries),
            np.mean(waters),
            operational / self.swarm_size
        ], dtype=np.float32)
        obs_parts.append(resource_obs)

        # 4. Wind
        wind_speed_norm = self.fire_sim.wind_speed / 20.0  # Normalize to ~[0, 1]
        wind_dir_rad = np.radians(self.fire_sim.wind_direction)
        wind_obs = np.array([
            min(1.0, wind_speed_norm),
            (np.sin(wind_dir_rad) + 1) / 2,
            (np.cos(wind_dir_rad) + 1) / 2
        ], dtype=np.float32)
        obs_parts.append(wind_obs)

        # 5. Global state
        total_cells = self.grid_size[0] * self.grid_size[1]
        burned_ratio = np.sum(self.fire_sim.burned_area) / total_cells
        active_ratio = len(self.fire_sim.get_active_fire_cells()) / total_cells
        step_ratio = self.current_step / self.max_episode_steps

        global_obs = np.array([
            burned_ratio,
            min(1.0, active_ratio * 10),  # Scale up small values
            step_ratio
        ], dtype=np.float32)
        obs_parts.append(global_obs)

        # Concatenate all observations
        obs = np.concatenate(obs_parts)

        # Clip to valid range
        obs = np.clip(obs, 0.0, 1.0)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about current state."""
        return {
            'step': self.current_step,
            'burned_cells': int(np.sum(self.fire_sim.burned_area)),
            'active_fires': len(self.fire_sim.get_active_fire_cells()),
            'operational_drones': sum(1 for d in self.swarm.drones.values() if d.is_operational()),
            'avg_battery': np.mean([d.battery_level for d in self.swarm.drones.values()]),
            'avg_water': np.mean([d.water_level for d in self.swarm.drones.values()]),
            'total_water_dropped': self.swarm.total_water_dropped,
            'episode_metrics': dict(self.episode_metrics),
        }

    def render(self):
        """Render the environment."""
        if self.render_mode == 'rgb_array':
            return self._render_frame()
        elif self.render_mode == 'human':
            self._render_human()

    def _render_frame(self) -> np.ndarray:
        """Render frame as RGB array."""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig, ax = plt.subplots(figsize=(8, 8))

        # Draw fire intensity
        im = ax.imshow(
            self.fire_sim.heat_intensity,
            cmap='hot',
            vmin=0,
            vmax=self.config.fire.max_heat_intensity_kw,
            origin='lower'
        )

        # Draw burned area overlay
        burned_mask = self.fire_sim.burned_area.astype(float)
        burned_mask[burned_mask == 0] = np.nan
        ax.imshow(burned_mask, cmap='Greys', alpha=0.3, origin='lower')

        # Draw drones
        for drone in self.swarm.drones.values():
            r, c = self.fire_sim._latlon_to_grid(drone.position[0], drone.position[1])

            if drone.state == DroneState.DROPPING_WATER:
                color = 'blue'
                marker = 'v'
            elif drone.state == DroneState.FLYING_TO_FIRE:
                color = 'green'
                marker = '^'
            elif drone.state == DroneState.REFILLING:
                color = 'cyan'
                marker = 's'
            elif drone.is_operational():
                color = 'white'
                marker = 'o'
            else:
                color = 'red'
                marker = 'x'

            ax.scatter(c, r, c=color, marker=marker, s=20, alpha=0.7)

        ax.set_title(f'Step {self.current_step} | Active: {len(self.fire_sim.get_active_fire_cells())} | Drones: {sum(1 for d in self.swarm.drones.values() if d.is_operational())}')

        # Convert to RGB array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        rgb_array = np.asarray(buf)[:, :, :3]

        plt.close(fig)
        return rgb_array

    def _render_human(self):
        """Render for human viewing."""
        # For human mode, we'd use matplotlib interactive
        pass

    def close(self):
        """Clean up environment resources."""
        pass


class FireControlMultiAgentEnv(gym.Env):
    """
    Multi-agent environment where each drone is controlled independently.

    This is useful for decentralized control learning where each drone
    makes its own decisions based on local observations.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        num_drones: int = 10,
        grid_size: Tuple[int, int] = (50, 50),
        max_episode_steps: int = 200,
        communication_enabled: bool = True,
    ):
        """Initialize multi-agent environment."""
        super().__init__()

        self.config = config or load_config()
        self.num_drones = num_drones
        self.grid_size = grid_size
        self.max_episode_steps = max_episode_steps
        self.communication_enabled = communication_enabled

        self.bounds = tuple(self.config.fire.bounds)
        self.dt_seconds = 300.0

        # Per-agent observation space
        # Local observation: nearby fire, own state, nearby drones
        local_obs_dim = 25 + 5 + 10  # 5x5 fire grid + drone state + nearby drones
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(local_obs_dim,), dtype=np.float32
        )

        # Per-agent action space
        # 0: idle, 1-4: move direction, 5: drop water, 6: go to refill
        self.action_space = spaces.Discrete(7)

        self.fire_sim: Optional[MockFireSimulator] = None
        self.drones: Dict[str, Drone] = {}
        self.current_step = 0

    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Initialize fire
        self.fire_sim = MockFireSimulator(
            grid_size=self.grid_size,
            bounds=self.bounds
        )

        num_ignitions = np.random.randint(1, 3)
        ignitions = [
            (np.random.randint(10, self.grid_size[0]-10),
             np.random.randint(10, self.grid_size[1]-10))
            for _ in range(num_ignitions)
        ]
        self.fire_sim.set_ignition_points(ignitions)

        # Initialize drones
        self.drones = {}
        for i in range(self.num_drones):
            drone_id = f"drone_{i:03d}"
            lat = np.random.uniform(self.bounds[0], self.bounds[2])
            lon = np.random.uniform(self.bounds[1], self.bounds[3])
            self.drones[drone_id] = Drone(drone_id, (lat, lon))

        self.current_step = 0

        # Return observations for all agents
        obs = {drone_id: self._get_drone_obs(drone_id) for drone_id in self.drones}
        info = {'step': 0}

        return obs, info

    def step(self, actions: Dict[str, int]):
        """Execute actions for all agents."""
        self.current_step += 1

        # Apply actions
        for drone_id, action in actions.items():
            self._apply_drone_action(drone_id, action)

        # Step fire simulation
        fire_state = self.fire_sim.step()

        # Calculate rewards
        rewards = {}
        for drone_id in self.drones:
            rewards[drone_id] = self._calculate_drone_reward(drone_id)

        # Get observations
        obs = {drone_id: self._get_drone_obs(drone_id) for drone_id in self.drones}

        # Check termination
        terminated = len(self.fire_sim.get_active_fire_cells()) == 0
        truncated = self.current_step >= self.max_episode_steps

        info = {
            'step': self.current_step,
            'active_fires': len(self.fire_sim.get_active_fire_cells()),
        }

        return obs, rewards, terminated, truncated, info

    def _get_drone_obs(self, drone_id: str) -> np.ndarray:
        """Get local observation for a drone."""
        drone = self.drones[drone_id]

        # Get drone's grid position
        r, c = self.fire_sim._latlon_to_grid(drone.position[0], drone.position[1])

        # Local fire observation (5x5 around drone)
        fire_obs = np.zeros(25)
        idx = 0
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size[0] and 0 <= nc < self.grid_size[1]:
                    fire_obs[idx] = self.fire_sim.heat_intensity[nr, nc] / 1000.0
                idx += 1

        # Drone state
        state_obs = np.array([
            drone.battery_level,
            drone.water_level,
            float(drone.state == DroneState.IDLE),
            float(drone.state == DroneState.DROPPING_WATER),
            float(drone.is_operational()),
        ])

        # Nearby drones (simplified)
        nearby_obs = np.zeros(10)
        nearby_idx = 0
        for other_id, other_drone in self.drones.items():
            if other_id != drone_id and nearby_idx < 5:
                dist = np.sqrt(
                    (drone.position[0] - other_drone.position[0])**2 +
                    (drone.position[1] - other_drone.position[1])**2
                )
                nearby_obs[nearby_idx * 2] = min(1.0, dist / 0.1)
                nearby_obs[nearby_idx * 2 + 1] = other_drone.water_level
                nearby_idx += 1

        obs = np.concatenate([fire_obs, state_obs, nearby_obs]).astype(np.float32)
        return np.clip(obs, 0.0, 1.0)

    def _apply_drone_action(self, drone_id: str, action: int):
        """Apply action to drone."""
        drone = self.drones[drone_id]

        if action == 0:  # Idle
            pass
        elif action in [1, 2, 3, 4]:  # Move
            # Simple movement in cardinal directions
            move_delta = 0.01  # degrees
            if action == 1:  # North
                drone.position = (drone.position[0] + move_delta, drone.position[1])
            elif action == 2:  # South
                drone.position = (drone.position[0] - move_delta, drone.position[1])
            elif action == 3:  # East
                drone.position = (drone.position[0], drone.position[1] + move_delta)
            elif action == 4:  # West
                drone.position = (drone.position[0], drone.position[1] - move_delta)
        elif action == 5:  # Drop water
            if drone.water_level > 0 and drone.state != DroneState.DROPPING_WATER:
                drone.state = DroneState.DROPPING_WATER
                r, c = self.fire_sim._latlon_to_grid(drone.position[0], drone.position[1])
                water = min(drone.water_level * 10, 5)
                self.fire_sim.apply_water_drop(r, c, water)
                drone.water_level = max(0, drone.water_level - 0.5)
        elif action == 6:  # Refill
            drone.water_level = min(1.0, drone.water_level + 0.1)

    def _calculate_drone_reward(self, drone_id: str) -> float:
        """Calculate reward for a drone."""
        drone = self.drones[drone_id]
        reward = 0.0

        # Small reward for being near active fires
        r, c = self.fire_sim._latlon_to_grid(drone.position[0], drone.position[1])
        if self.fire_sim.heat_intensity[r, c] > 100:
            reward += 0.1

        # Reward for water drops on fire
        if drone.state == DroneState.DROPPING_WATER:
            if self.fire_sim.heat_intensity[r, c] > 0:
                reward += 1.0

        # Penalty for low resources
        if drone.battery_level < 0.2:
            reward -= 0.1

        return reward

    def render(self):
        pass

    def close(self):
        pass


# Register environments
def register_envs():
    """Register environments with Gymnasium."""
    from gymnasium.envs.registration import register

    try:
        register(
            id='FireControl-v0',
            entry_point='envs.fire_control_env:FireControlEnv',
            max_episode_steps=200,
        )
    except:
        pass  # Already registered

    try:
        register(
            id='FireControlMultiAgent-v0',
            entry_point='envs.fire_control_env:FireControlMultiAgentEnv',
            max_episode_steps=200,
        )
    except:
        pass
