# Wildfire Spread and Drone Swarm Coordination Simulator

## Project Overview

This is a modular software system that simulates wildfire flame front spread using the WRF-SFIRE model via the wrfxpy GitHub repository and coordinates a swarm of autonomous drones (5,000–10,000 units) using reinforcement learning. The drones each have a max flight time of 2 hours and carry 10L water, dropped via sprinklers. They autonomously refill at lakes and aim to minimize fire spread.

## Features

- **Wildfire Simulation**: Run wildfire simulations using WRF-SFIRE via wrfxpy or mock simulator for rapid prototyping
- **Rothermel Fire Physics**: Realistic fire spread model based on Rothermel equations
- **Flame Front Extraction**: Extract burned area, heat intensity, and fire perimeter data over time
- **Autonomous Drone Swarm**: 5,000-10,000 drones with realistic constraints (2h flight time, 10L water capacity)
- **Reinforcement Learning**: Multi-agent DQN/PPO/SAC/A2C algorithms via Stable-Baselines3
- **Gymnasium Environment**: Full RL environment wrapper for training
- **Real-time Visualization**: Fire propagation and drone movement visualization
- **Google Colab Support**: Ready-to-use training notebook for Colab Pro
- **Modular Architecture**: Extensible system with YAML configuration
- **Spatial Indexing**: Efficient O(1) neighbor queries for large swarms

## Tech Stack

- **Python 3.8+** - Core programming language
- **wrfxpy + WRF-SFIRE** - Wildfire simulation (Fortran backend)
- **Gymnasium** - RL environment framework
- **Stable-Baselines3** - Reinforcement learning algorithms (PPO, DQN, SAC, A2C)
- **PyTorch** - Deep learning backend
- **Matplotlib / Plotly** - Visualization
- **NumPy / SciPy** - Scientific computing

## Directory Structure

```
fire_control/
├── agents/                 # RL agents and training infrastructure
│   ├── __init__.py
│   ├── trainer.py         # Main training class
│   └── callbacks.py       # Custom training callbacks
├── config/                 # Configuration system
│   ├── __init__.py        # Config loader and dataclasses
│   └── default.yaml       # Default configuration
├── drones/                 # Drone swarm classes and behavior
│   ├── drone.py           # Individual drone class
│   ├── swarm.py           # Swarm coordination
│   └── behaviors.py       # Behavior policies
├── envs/                   # Gymnasium RL environments
│   ├── __init__.py
│   └── fire_control_env.py # Main RL environment
├── notebooks/              # Jupyter notebooks
│   └── train_colab_pro.ipynb # Google Colab training notebook
├── sim/                    # Fire simulation
│   ├── fire_state.py      # Fire state data structures
│   ├── mock_fire.py       # Cellular automata simulator
│   ├── fire_physics.py    # Rothermel fire model
│   └── wrf_interface.py   # WRF-SFIRE integration
├── utils/                  # Utility modules
│   ├── __init__.py
│   └── spatial.py         # Spatial indexing for scalability
├── scripts/                # Entry points
│   └── run_grid_simulation.py
├── visualization/          # Visualization tools
│   └── realtime_viz.py
├── tests/                  # Unit tests
│   └── test_basic_functionality.py
├── results/                # Simulation outputs
├── Makefile               # Build commands
├── requirements.txt       # Python dependencies
└── readme.md              # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Simple Simulation

```bash
# Quick demo with 100 drones
make demo

# Standard simulation with 1000 drones
make run-simulation
```

### 3. Train RL Agent (Local)

```python
from agents.trainer import FireControlTrainer

# Create trainer
trainer = FireControlTrainer(
    algorithm='PPO',
    n_envs=4,
    seed=42
)

# Create environment
trainer.create_env(swarm_size=50, grid_size=(50, 50))

# Create and train model
trainer.create_model()
trainer.train(total_timesteps=100000)

# Evaluate
results = trainer.evaluate(n_episodes=10)
print(f"Mean reward: {results['mean_reward']:.2f}")
```

### 4. Train on Google Colab

1. Open `notebooks/train_colab_pro.ipynb` in Google Colab
2. Enable GPU: Runtime > Change runtime type > GPU
3. Run all cells
4. Download trained model

## Configuration

All parameters are configurable via YAML:

```yaml
# config/default.yaml

drone:
  max_flight_time_hours: 2.0
  water_capacity_liters: 10.0
  max_speed_mps: 15.0

fire:
  grid_size: [100, 100]
  cell_size_meters: 100.0
  base_spread_rate: 0.1

swarm:
  size: 100
  assignment_strategy: "coordinated"

training:
  algorithm: "PPO"
  total_timesteps: 1000000
  learning_rate: 0.0003
```

Load custom config:

```python
from config import Config

config = Config.from_yaml('config/my_config.yaml')
```

## RL Environment

The `FireControlEnv` provides a Gymnasium-compatible interface:

```python
import gymnasium as gym
from envs.fire_control_env import FireControlEnv

env = FireControlEnv(
    swarm_size=50,
    grid_size=(50, 50),
    max_episode_steps=200,
)

obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

**Observation Space:**
- Fire grid (heat intensity, normalized)
- Drone density per zone
- Resource summary (battery, water, operational ratio)
- Wind conditions
- Global state (burned ratio, step ratio)

**Action Space (Discrete):**
- 27 actions: 9 target zones × 3 strategies (greedy, coordinated, defensive)

**Rewards:**
- `+10` per fire cell suppressed
- `-1` per new cell burned
- `-50` per drone lost
- Efficiency bonuses for water usage

## Fire Physics

Two fire spread models are available:

### Mock Simulator (Fast)
Cellular automata with:
- Wind-driven spread
- Fuel moisture effects
- Terrain slope factors
- Water suppression

### Rothermel Model (Realistic)
Based on Rothermel (1972) fire spread equations:
- 13 standard fuel models
- Physics-based spread rate calculation
- Fireline intensity and flame length

```python
from sim.fire_physics import ImprovedFireSimulator, FuelModel

sim = ImprovedFireSimulator(
    grid_size=(100, 100),
    use_rothermel=True
)
sim.set_ignition_points([(50, 50)])
sim.set_weather(wind_speed=10.0, wind_direction=45.0)

for _ in range(100):
    state = sim.step()
    print(f"Burning cells: {state['burning_cells']}")
```

## Spatial Indexing

For large swarms (5000+ drones), use spatial indexing:

```python
from utils.spatial import GridSpatialIndex

index = GridSpatialIndex(
    bounds=(34.0, -119.0, 35.0, -118.0),
    cell_size=0.01
)

# Insert drones
for drone_id, position in drone_positions.items():
    index.insert(drone_id, position)

# Query neighbors (O(k) instead of O(n))
neighbors = index.query_radius(position, radius=0.05)

# Build communication graph efficiently
graph = index.build_communication_graph(communication_range=0.01)
```

## Training Results

Typical training progress with PPO:

| Timesteps | Mean Reward | Episodes |
|-----------|-------------|----------|
| 100k      | -15.2       | 500      |
| 250k      | 5.3         | 1,250    |
| 500k      | 18.7        | 2,500    |
| 1M        | 28.4        | 5,000    |

## Makefile Commands

```bash
make install          # Install dependencies
make test            # Run tests
make demo            # 100-drone demo
make run-simulation  # 1000-drone simulation
make run-large       # 5000-drone simulation
make compare-policies # Compare strategies
make lint            # Code quality check
make clean           # Clean generated files
```

## Implementation Status

- [x] Modular directory structure
- [x] Mock fire simulator for rapid prototyping
- [x] Rothermel fire physics model
- [x] Drone class with movement, water, refill logic
- [x] Swarm coordination (greedy, coordinated, cluster-based)
- [x] Gymnasium RL environment
- [x] PPO/DQN/A2C training infrastructure
- [x] YAML configuration system
- [x] Spatial indexing for scalability
- [x] Google Colab training notebook
- [x] Real-time visualization
- [ ] Full WRF-SFIRE integration
- [ ] Multi-agent RL (independent drone control)
- [ ] Real terrain/fuel data integration

## Stretch Goals

- Tight WRF-SFIRE integration with drone water drop effects
- Skybrush/Gazebo deployment with PX4 SITL drones
- Real-time fire evolution and drone reaction loops
- Transfer learning from grid world to geospatial maps
- Real-time satellite data integration
- Weather-aware pathfinding

## Contributing

1. Follow the modular architecture principles
2. Add tests for new components
3. Document API changes
4. Use consistent coding standards
5. Update configuration schema for new parameters

## License

MIT License - see LICENSE file for details.
