"""
Configuration management for Fire Control system.
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DroneConfig:
    """Drone physical properties configuration."""
    max_flight_time_hours: float = 2.0
    water_capacity_liters: float = 10.0
    max_speed_mps: float = 15.0
    water_drop_rate_lps: float = 5.0
    refill_time_seconds: float = 30.0
    battery_drain_rate: float = 0.02
    communication_range_m: float = 1000.0
    sensor_range_m: float = 500.0
    acceleration_mps2: float = 3.0
    deceleration_mps2: float = 5.0
    min_altitude_m: float = 5.0
    max_altitude_m: float = 150.0


@dataclass
class FireConfig:
    """Fire simulation configuration."""
    grid_size: tuple = (100, 100)
    cell_size_meters: float = 100.0
    bounds: tuple = (34.0, -119.0, 35.0, -118.0)
    base_spread_rate: float = 0.1
    max_heat_intensity_kw: float = 1000.0
    ignition_threshold_kw: float = 300.0
    heat_decay_rate: float = 0.05
    water_suppression_factor: float = 50.0
    water_decay_rate: float = 0.1


@dataclass
class SwarmConfig:
    """Swarm configuration."""
    size: int = 100
    deployment_area: tuple = (34.0, -119.0, 35.0, -118.0)
    assignment_strategy: str = "coordinated"
    max_workers: int = 8


@dataclass
class TrainingConfig:
    """RL training configuration."""
    algorithm: str = "PPO"
    learning_rate: float = 0.0003
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    total_timesteps: int = 1000000
    eval_freq: int = 10000
    save_freq: int = 50000


@dataclass
class RewardConfig:
    """Reward function configuration."""
    fire_suppressed_cell: float = 10.0
    fire_intensity_reduction: float = 0.1
    area_saved_per_cell: float = 5.0
    water_efficiency_bonus: float = 1.0
    drone_lost_penalty: float = -50.0
    fire_spread_penalty: float = -1.0
    idle_drone_penalty: float = -0.1


@dataclass
class Config:
    """Main configuration container."""
    drone: DroneConfig = field(default_factory=DroneConfig)
    fire: FireConfig = field(default_factory=FireConfig)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        config = cls()

        if 'drone' in data:
            for key, value in data['drone'].items():
                if hasattr(config.drone, key):
                    setattr(config.drone, key, value)

        if 'fire' in data:
            fire_data = data['fire']
            if 'grid_size' in fire_data:
                config.fire.grid_size = tuple(fire_data['grid_size'])
            if 'bounds' in fire_data:
                config.fire.bounds = tuple(fire_data['bounds'])
            for key in ['cell_size_meters', 'base_spread_rate', 'max_heat_intensity_kw',
                        'ignition_threshold_kw', 'heat_decay_rate', 'water_suppression_factor',
                        'water_decay_rate']:
                if key in fire_data:
                    setattr(config.fire, key, fire_data[key])

        if 'swarm' in data:
            swarm_data = data['swarm']
            if 'size' in swarm_data:
                config.swarm.size = swarm_data['size']
            if 'deployment_area' in swarm_data:
                config.swarm.deployment_area = tuple(swarm_data['deployment_area'])
            if 'assignment_strategy' in swarm_data:
                config.swarm.assignment_strategy = swarm_data['assignment_strategy']
            if 'max_workers' in swarm_data:
                config.swarm.max_workers = swarm_data['max_workers']

        if 'training' in data:
            train_data = data['training']
            for key in ['algorithm', 'total_timesteps', 'eval_freq', 'save_freq']:
                if key in train_data:
                    setattr(config.training, key, train_data[key])
            # PPO specific
            if 'ppo' in train_data:
                ppo = train_data['ppo']
                for key in ['learning_rate', 'n_steps', 'batch_size', 'n_epochs',
                            'gamma', 'gae_lambda', 'clip_range']:
                    if key in ppo:
                        setattr(config.training, key, ppo[key])

        if 'rewards' in data:
            for key, value in data['rewards'].items():
                if hasattr(config.rewards, key):
                    setattr(config.rewards, key, value)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'drone': {
                'max_flight_time_hours': self.drone.max_flight_time_hours,
                'water_capacity_liters': self.drone.water_capacity_liters,
                'max_speed_mps': self.drone.max_speed_mps,
                'water_drop_rate_lps': self.drone.water_drop_rate_lps,
                'refill_time_seconds': self.drone.refill_time_seconds,
                'battery_drain_rate': self.drone.battery_drain_rate,
                'communication_range_m': self.drone.communication_range_m,
                'sensor_range_m': self.drone.sensor_range_m,
            },
            'fire': {
                'grid_size': list(self.fire.grid_size),
                'cell_size_meters': self.fire.cell_size_meters,
                'bounds': list(self.fire.bounds),
                'base_spread_rate': self.fire.base_spread_rate,
                'max_heat_intensity_kw': self.fire.max_heat_intensity_kw,
            },
            'swarm': {
                'size': self.swarm.size,
                'assignment_strategy': self.swarm.assignment_strategy,
            },
            'training': {
                'algorithm': self.training.algorithm,
                'learning_rate': self.training.learning_rate,
                'total_timesteps': self.training.total_timesteps,
            },
            'rewards': {
                'fire_suppressed_cell': self.rewards.fire_suppressed_cell,
                'drone_lost_penalty': self.rewards.drone_lost_penalty,
            }
        }


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or use defaults."""
    if config_path is None:
        # Try to find default config
        possible_paths = [
            'config/default.yaml',
            '../config/default.yaml',
            os.path.join(os.path.dirname(__file__), 'default.yaml'),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break

    if config_path and os.path.exists(config_path):
        return Config.from_yaml(config_path)

    return Config()


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config):
    """Set global configuration instance."""
    global _config
    _config = config
