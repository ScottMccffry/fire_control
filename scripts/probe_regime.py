#!/usr/bin/env python3
"""Probe candidate regimes: find one where random burns ~40-60% (headroom) and
greedy clearly beats it, on a grid big enough that the fire outruns the fleet."""
import sys
from pathlib import Path
import logging
import numpy as np

root = Path(__file__).parent.parent
sys.path.insert(0, str(root)); sys.path.insert(0, str(root / "scripts"))
from envs.per_drone_env import PerDroneSwarmVecEnv  # noqa: E402
logging.disable(logging.CRITICAL)


def run(kind, kw, seed):
    env = PerDroneSwarmVecEnv(episode_seed=seed, **kw)
    obs = env.reset()
    while True:
        if kind == "random":
            a = np.random.uniform(-1, 1, (env.n_drones, 2)).astype(np.float32)
        else:
            a = env.greedy_actions()
        env.step_async(a)
        obs, _, d, info = env.step_wait()
        if d[0]:
            env.close()
            return info[0]["burned_cells"]


CANDS = [
    dict(grid_size=60, n_drones=24, max_steps=180, n_ignitions=1, wind_speed=14.0,
         fuel_moisture_range=[0.06, 0.18], fuel_load_range=[0.75, 1.0], base_spread_rate=0.18),
    dict(grid_size=60, n_drones=24, max_steps=150, n_ignitions=1, wind_speed=14.0,
         fuel_moisture_range=[0.08, 0.2], fuel_load_range=[0.7, 1.0], base_spread_rate=0.15),
    dict(grid_size=80, n_drones=24, max_steps=160, n_ignitions=1, wind_speed=14.0,
         fuel_moisture_range=[0.06, 0.18], fuel_load_range=[0.75, 1.0], base_spread_rate=0.17),
]
for kw in CANDS:
    kw2 = dict(kw, continuous=True, wind_obs=True, spread_penalty=2.5)
    tot = kw["grid_size"] ** 2
    rb = np.mean([run("random", kw2, 9000 + e) for e in range(6)])
    gb = np.mean([run("greedy", kw2, 9000 + e) for e in range(6)])
    print(f"grid={kw['grid_size']} steps={kw['max_steps']} spread={kw['base_spread_rate']} "
          f"-> random {rb:.0f} ({rb/tot:.0%})  greedy {gb:.0f} ({gb/tot:.0%})  "
          f"greedy gain {(rb-gb)/rb:+.0%}")
