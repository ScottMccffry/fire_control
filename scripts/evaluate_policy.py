#!/usr/bin/env python3
"""
Paired evaluation of drone-swarm controllers on identical wildfire scenarios.

The fire's spread is stochastic, and episode-to-episode variance in burned area
is large. Comparing two controllers on *different* random fires (as the quick
summary in train_drone_swarm.py does) is therefore very noisy. This script runs
every controller on the *same* set of seeded fires, so differences reflect the
controller, not luck.

It compares three controllers:
  * random  - uniformly random drone movement (lower bound)
  * greedy  - hand-coded: each drone steps toward the nearest active fire cell
  * trained - the saved PPO policy (with its VecNormalize obs statistics)

Example
-------
    python scripts/evaluate_policy.py \
        --model agents/checkpoints/drone_swarm_ppo_medium_g32_d16 \
        --difficulty medium --episodes 40
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from envs import DroneFireEnv  # noqa: E402
from train_drone_swarm import DIFFICULTY_PRESETS  # noqa: E402


def greedy_action(env):
    """Move each drone one Manhattan step toward its nearest active fire cell."""
    thr = env.sim.ignition_threshold * 0.3
    fr, fc = np.where(env.sim.heat_intensity > thr)
    acts = np.zeros(env.n_drones, dtype=np.int64)
    if len(fr) == 0:
        return acts
    for i in range(env.n_drones):
        r, c = env.drone_pos[i]
        d = np.abs(fr - r) + np.abs(fc - c)
        j = int(np.argmin(d))
        tr, tc = fr[j], fc[j]
        if r == tr and c == tc:
            acts[i] = 0
        elif abs(tr - r) >= abs(tc - c):
            acts[i] = 1 if tr < r else 2
        else:
            acts[i] = 3 if tc < c else 4
    return acts


def run_episode(env, seed, policy="random", model=None, normalizer=None):
    obs, _ = env.reset(seed=seed)
    done = False
    info = {}
    while not done:
        if policy == "random":
            action = env.action_space.sample()
        elif policy == "greedy":
            action = greedy_action(env)
        else:  # trained
            pred = normalizer.normalize_obs(obs) if normalizer is not None else obs
            action, _ = model.predict(pred, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
    return info["burned_cells"], info["fire_extinguished"]


def main():
    p = argparse.ArgumentParser(description="Paired controller evaluation on identical fires")
    p.add_argument("--model", default="agents/checkpoints/drone_swarm_ppo_medium_g32_d16",
                   help="Path to saved PPO model (without .zip)")
    p.add_argument("--difficulty", choices=list(DIFFICULTY_PRESETS), default="medium")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--base-seed", type=int, default=10000)
    args = p.parse_args()

    logging.disable(logging.CRITICAL)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    cfg = DIFFICULTY_PRESETS[args.difficulty]
    grid, tot = cfg["grid_size"], cfg["grid_size"] ** 2

    def make_env():
        return DroneFireEnv(**cfg)

    env = make_env()

    # Load the trained policy and its observation-normalization statistics.
    model = PPO.load(args.model)
    vec_path = args.model + "_vecnormalize.pkl"
    normalizer = None
    if Path(vec_path).exists():
        normalizer = VecNormalize.load(vec_path, DummyVecEnv([make_env]))
        normalizer.training = False
        normalizer.norm_reward = False

    seeds = [args.base_seed + i for i in range(args.episodes)]
    results = {k: {"burned": [], "ext": 0} for k in ("random", "greedy", "trained")}

    for s in seeds:
        for name, kw in (
            ("random", {}),
            ("greedy", {}),
            ("trained", {"model": model, "normalizer": normalizer}),
        ):
            burned, ext = run_episode(env, s, policy=name, **kw)
            results[name]["burned"].append(burned)
            results[name]["ext"] += int(ext)

    print("=" * 70)
    print(f"PAIRED EVALUATION  ({args.episodes} identical fires, "
          f"difficulty={args.difficulty}, grid={grid}x{grid}={tot} cells)")
    print("=" * 70)
    rand_mean = float(np.mean(results["random"]["burned"]))
    print(f"{'controller':<10} {'mean burned':>12} {'% of grid':>10} "
          f"{'extinguish':>11} {'vs random':>12}")
    for name in ("random", "greedy", "trained"):
        b = np.array(results[name]["burned"], dtype=float)
        red = rand_mean - b.mean()
        print(f"{name:<10} {b.mean():>12.1f} {b.mean() / tot:>9.1%} "
              f"{results[name]['ext'] / args.episodes:>10.0%} "
              f"{red:>+8.1f} ({red / max(1.0, rand_mean):>+5.1%})")

    # Paired difference (trained vs random on the same fires) with a std error.
    diff = np.array(results["random"]["burned"], dtype=float) - np.array(
        results["trained"]["burned"], dtype=float
    )
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    print("-" * 70)
    print(f"Trained vs random (paired): {diff.mean():+.1f} cells saved "
          f"± {se:.1f} SE  (n={len(diff)})")
    print(f"  -> {'SIGNIFICANT' if abs(diff.mean()) > 2 * se else 'within noise'} "
          f"at ~2 SE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
