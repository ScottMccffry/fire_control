#!/usr/bin/env python3
"""Test the user's macro-view intuition: does nearest-fire greedy fail when
there are SEVERAL fires of different sizes (it piles onto the near/small one
while a far one runs away)? Compare random vs nearest-greedy vs a
'biggest-front' greedy (target the largest active-fire cluster's centroid)
across 1, 2, 4 ignitions on the macro grid."""
import sys
from pathlib import Path
import logging
import numpy as np
from scipy import ndimage

root = Path(__file__).parent.parent
sys.path.insert(0, str(root)); sys.path.insert(0, str(root / "scripts"))
from envs.per_drone_env import PerDroneSwarmVecEnv  # noqa: E402
logging.disable(logging.CRITICAL)


def biggest_front_actions(env):
    """Each drone steers toward the centroid of the LARGEST active-fire blob."""
    sim = env.sim
    thr = sim.ignition_threshold * 0.3
    active = sim.heat_intensity > thr
    lab, n = ndimage.label(active)
    if n == 0:
        return np.zeros((env.n_drones, 2), np.float32)
    sizes = ndimage.sum(active, lab, range(1, n + 1))
    big = np.argmax(sizes) + 1
    cr, cc = ndimage.center_of_mass(active, lab, big)
    d = np.stack([cr - env.drone_posf[:, 0], cc - env.drone_posf[:, 1]], 1)
    nrm = np.linalg.norm(d, axis=1, keepdims=True)
    return (d / np.maximum(nrm, 1e-6)).astype(np.float32)


def run(kind, kw, seed):
    env = PerDroneSwarmVecEnv(episode_seed=seed, **kw)
    obs = env.reset()
    while True:
        if kind == "random":
            a = np.random.uniform(-1, 1, (env.n_drones, 2)).astype(np.float32)
        elif kind == "greedy":
            a = env.greedy_actions()
        else:
            a = biggest_front_actions(env)
        env.step_async(a)
        obs, _, d, info = env.step_wait()
        if d[0]:
            env.close()
            return info[0]["burned_cells"]


BASE = dict(grid_size=60, n_drones=24, max_steps=180, wind_speed=14.0,
            fuel_moisture_range=[0.06, 0.18], fuel_load_range=[0.75, 1.0],
            base_spread_rate=0.18, continuous=True, wind_obs=True, spread_penalty=2.5)
tot = BASE["grid_size"] ** 2
for nig in (1, 2, 4):
    kw = dict(BASE, n_ignitions=nig)
    res = {}
    for kind in ("random", "greedy", "biggest"):
        res[kind] = np.mean([run(kind, kw, 9000 + e) for e in range(6)])
    print(f"ignitions={nig}: " + "  ".join(
        f"{k} {v:.0f} ({v/tot:.0%})" for k, v in res.items()))
