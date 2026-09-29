# Wildfire Spread and Drone Swarm Coordination Simulator

Simulate wildfire flame-front spread and train autonomous drone swarms to
suppress it with reinforcement learning — from a fast mock fire for RL
prototyping all the way to a **real WRF-SFIRE coupled fire–atmosphere model**.

## What works today

- **Fire simulation**: a fast vectorized cellular-automaton fire (`MockFireSimulator`)
  for RL, plus a real **WRF-SFIRE** integration (compiled from source; restart-based
  coupling) and a `WRFReplaySimulator` bridge from real `wrfout` output to the
  drone-swarm `FireState`.
- **Drone swarm**: drone/swarm classes with battery, 10 L water, refill, and
  baseline heuristic policies (random / greedy / coordinated).
- **RL environments**: `DroneFireEnv` (flat and spatial/CNN observation modes) and
  `PerDroneSwarmVecEnv` (decentralized, parameter-shared per-drone policy).
- **Trained policies that beat the greedy baseline** and transfer across grid scale.
- **Closed-loop coupling**: the trained swarm modifies a live WRF-SFIRE fire and
  reduces burned area.

See **`docs/rl_pipeline.md`** (RL results) and **`docs/wrf_sfire.md`** (WRF-SFIRE
build, bridge, and closed-loop coupling) for the full story and numbers.

## Headline results (paired evaluation on identical fires)

| controller | medium (32×32) burned | large (100×100) burned |
|------------|----------------------:|-----------------------:|
| random | ~60% of grid | ~25% |
| greedy (hand-coded) | ~19% (+64% vs random) | ~9% (+62%) |
| **per-drone RL policy** | **~17% (+72%)** | **~7% (+74%)** |

The decentralized per-drone policy is the first controller to beat greedy, learns
from scratch (no behavior cloning), and is scale-free (train at 32×32, deploy at
100×100). On a **real WRF-SFIRE fire** it tracks the flame front at near-oracle
coverage (open-loop) and, in **closed-loop**, cuts burned area by modifying the
fuel where drones drop water.

## Install

```bash
pip install -r requirements.txt          # core: numpy, scipy, gymnasium, stable-baselines3, torch
```
Running real WRF-SFIRE additionally requires a compiled WRF-SFIRE (gfortran +
NetCDF/HDF5) — see `docs/wrf_sfire.md`.

## Quick start

```bash
# 1. Heuristic swarm vs a mock fire (no RL needed)
python scripts/run_grid_simulation.py --swarm-size 100 --hours 2 --no-viz

# 2. Train the decentralized per-drone RL policy (recommended) + paired eval
python scripts/train_per_drone.py --difficulty medium --timesteps 2000000

# 3. Paired comparison of a saved flat-MLP policy vs random/greedy
python scripts/evaluate_policy.py --difficulty medium --episodes 40

# 4. CNN policy with behavior-cloning warm-start (alternative architecture)
python scripts/train_drone_swarm_cnn.py --difficulty medium

# 5. Real WRF-SFIRE: bridge wrfout frames into the FireState pipeline
python scripts/wrf_sfire_demo.py --wrfout-dir /path/to/wrfout --out fire.png

# 6. Run the trained swarm on real WRF-SFIRE frames (open-loop)
python scripts/evaluate_on_wrf.py --wrfout-dir /path/to/wrfout \
    --policy agents/checkpoints/per_drone_ppo_large_g100_d40

# 7. Closed-loop: swarm suppresses a live WRF-SFIRE fire (restart cycling)
python scripts/wrf_closed_loop.py --src /path/to/wrf_run \
    --policy agents/checkpoints/per_drone_ppo_large_g100_d40
```

## Directory structure

```
fire_control/
├── sim/            # Fire models: MockFireSimulator, WRFFireSimulator, WRFReplaySimulator, FireState
├── envs/           # RL envs: DroneFireEnv (flat/spatial), PerDroneSwarmVecEnv
├── agents/         # Policy nets (FireSwarmExtractor CNN) + saved checkpoints
├── drones/         # Drone, DroneSwarm, baseline policies
├── scripts/        # Simulation, training, evaluation, WRF coupling entry points
├── visualization/  # Real-time fire/drone visualization
├── docs/           # rl_pipeline.md, wrf_sfire.md (+ figures)
└── tests/          # Unit/integration tests
```

## Architecture progression (what was tried)

1. **Flat-MLP RL** over a flattened fire map — did not learn (no spatial structure).
2. **CNN + behavior cloning** from greedy — matched greedy, scaled to 100×100.
3. **Decentralized per-drone policy** — each drone runs one shared policy on an
   egocentric observation (fire bearing, neighbour drones, fleet rank); per-drone
   rewards fix credit assignment. **Beats greedy**, scale-free.
4. **Real WRF-SFIRE** — compiled, run, bridged to `FireState`, then closed-loop
   coupled so the swarm changes the live fire.

## Status

- [x] Modular structure, vectorized mock fire, drone/swarm classes
- [x] Baseline policies (random, greedy, coordinated) + paired evaluation
- [x] RL training: flat-MLP, CNN+BC, and decentralized per-drone (beats greedy)
- [x] Real WRF-SFIRE build + `wrfout` → `FireState` bridge
- [x] Closed-loop drone suppression of a live WRF-SFIRE fire
- [ ] Geolocated WRF-SFIRE domains (California/Australia) — needs geog + GRIB data
- [ ] Calibrated water→suppression physics and finer coupling (in progress)
- [ ] Hardware-in-the-loop (Skybrush/PX4-SITL) deployment

## License

MIT License - see LICENSE file for details.
