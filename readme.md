# Wildfire Spread and Drone Swarm Coordination Simulator

## Project Overview

This is a modular software system that simulates wildfire flame front spread using the WRF-SFIRE model via the wrfxpy GitHub repository and coordinates a swarm of autonomous drones (5,000–10,000 units) using reinforcement learning. The drones each have a max flight time of 2 hours and carry 10L water, dropped via sprinklers. They autonomously refill at lakes and aim to minimize fire spread.

## Features

- **Wildfire Simulation**: Run wildfire simulations using WRF-SFIRE via wrfxpy for California and Australia
- **Flame Front Extraction**: Extract burned area, heat intensity, and fire perimeter data over time
- **Autonomous Drone Swarm**: 5,000-10,000 drones with realistic constraints (2h flight time, 10L water capacity)
- **Reinforcement Learning**: Multi-agent DQN/PPO algorithms for coordinated firefighting behavior
- **Real-time Visualization**: Fire propagation and drone movement visualization
- **Modular Architecture**: Extensible system supporting various simulation backends
- **Scalable Physics**: 2D grid world for large swarms with optional high-fidelity integration

## Tech Stack

- **Python** - Core programming language
- **wrfxpy + WRF-SFIRE** - Wildfire simulation (Fortran backend)
- **Gymnasium** - RL environment framework
- **Ray RLlib / Stable-Baselines3** - Multi-agent reinforcement learning
- **Matplotlib / Plotly** - Visualization
- **Skybrush / MAVSDK / PX4-SITL** - Optional advanced drone testing

## Directory Structure

```
fire_control/
├── sim/                    # Core wildfire simulation interface (wrfxpy + WRF-SFIRE)
├── envs/                   # Custom RL environments (DroneFireEnv)
├── agents/                 # RL agents and training code
├── drones/                 # Drone swarm classes and behavior models
├── data/                   # Input data (terrain, fuel maps, ignition points, weather)
│   ├── california/         # California fire domain data
│   └── australia/          # Australia fire domain data
├── scripts/                # Entry points for simulation, training, and evaluation
├── visualization/          # Fire spread and drone movement visualization tools
└── tests/                  # Unit tests and integration tests
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up WRF-SFIRE** (requires separate installation):
   ```bash
   # Follow wrfxpy installation guide
   git clone https://github.com/openwfm/wrfxpy.git
   ```

3. **Run Simple Grid Simulation**:
   ```bash
   python scripts/run_grid_simulation.py
   ```

4. **Train RL Agents**:
   ```bash
   python scripts/train_drone_swarm.py --config configs/dqn_swarm.yaml
   ```

5. **Visualize Results**:
   ```bash
   python scripts/visualize_simulation.py --run_id latest
   ```

## First Tasks Implementation

- [x] Modular directory structure
- [ ] wrfxpy integration with California fire domain
- [ ] 2D grid environment (DroneFireEnv) for RL testing
- [ ] Drone class with movement, water payload, and refill logic
- [ ] Mock fire dynamics for rapid RL prototyping
- [ ] Baseline swarm control policies (random, heuristic)
- [ ] Multi-agent RL training pipeline
- [ ] Real-time visualization system

## Stretch Goals

- Tight WRF-SFIRE integration with drone water drop effects
- Skybrush/Gazebo deployment with PX4 SITL drones
- Real-time fire evolution and drone reaction loops
- Lake refill behavior and resupply scheduling
- Transfer learning from grid world to geospatial maps
- Real-time satellite data integration
- Weather-aware pathfinding

## Contributing

1. Follow the modular architecture principles
2. Add tests for new components
3. Document API changes
4. Use consistent coding standards

## License

MIT License - see LICENSE file for details.
