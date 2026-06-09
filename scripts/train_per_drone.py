#!/usr/bin/env python3
"""
Train a decentralized PER-DRONE policy (parameter sharing) with PPO.

Every drone runs the same small policy on its own egocentric observation
(nearest-fire bearing, fire centroid, local heat patch, neighbour drone
offsets, fleet-centroid offset and fleet rank). Training uses
PerDroneSwarmVecEnv: each drone is one VecEnv stream, so a 16-drone swarm
yields 16 training samples per simulation step, and PPO learns one shared
policy with per-drone rewards (own drops + spacing + shared fire terms).

Example:
    python scripts/train_per_drone.py --difficulty medium --timesteps 2000000
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from envs.per_drone_env import PerDroneSwarmVecEnv  # noqa: E402
from train_drone_swarm import DIFFICULTY_PRESETS  # noqa: E402

# Per-drone env accepts these preset keys directly.
_ENV_KEYS = ("grid_size", "n_drones", "max_steps", "n_ignitions", "wind_speed",
             "fuel_moisture_range", "fuel_load_range", "base_spread_rate")


def env_kwargs(difficulty):
    cfg = DIFFICULTY_PRESETS[difficulty]
    return {k: cfg[k] for k in _ENV_KEYS if k in cfg}


def rollout(kind, kwargs, seed, model=None):
    """One paired episode on the fire defined by `seed`. Returns final stats."""
    env = PerDroneSwarmVecEnv(episode_seed=seed, **kwargs)
    obs = env.reset()
    while True:
        if kind == "random":
            actions = np.random.randint(0, 5, env.n_drones)
        elif kind == "greedy":
            actions = env.greedy_actions()
        else:
            actions, _ = model.predict(obs, deterministic=True)
        env.step_async(actions)
        obs, _, dones, infos = env.step_wait()
        if dones[0]:
            env.close()
            return infos[0]["burned_cells"], infos[0]["fire_extinguished"]


def paired_eval(kinds, kwargs, episodes, base_seed, model=None):
    tot = kwargs["grid_size"] ** 2
    results = {}
    for kind in kinds:
        burned, ext = [], 0
        for e in range(episodes):
            b, x = rollout(kind, kwargs, base_seed + e, model)
            burned.append(b)
            ext += int(x)
        results[kind] = (float(np.mean(burned)), ext / episodes)
    return results, tot


def main():
    p = argparse.ArgumentParser(description="Per-drone shared-policy PPO")
    p.add_argument("--difficulty", choices=list(DIFFICULTY_PRESETS), default="medium")
    p.add_argument("--timesteps", type=int, default=2_000_000,
                   help="Total per-drone frames (sim steps x n_drones)")
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="./agents/checkpoints")
    args = p.parse_args()

    logging.disable(logging.CRITICAL)
    from stable_baselines3 import PPO

    kwargs = env_kwargs(args.difficulty)
    print("=" * 66)
    print("Per-Drone Shared-Policy PPO (decentralized)")
    print("=" * 66)
    print(f"difficulty={args.difficulty}  grid={kwargs['grid_size']}  "
          f"drones={kwargs['n_drones']}  frames={args.timesteps}")

    print("\n[1/4] Baselines on identical fires ...")
    base, tot = paired_eval(("random", "greedy"), kwargs,
                            args.eval_episodes, 9000)
    for k, (b, x) in base.items():
        print(f"      {k:7s}: {b:7.0f} burned ({b/tot:.0%})  ext={x:.0%}")

    print("\n[2/4] Training ...")
    env = PerDroneSwarmVecEnv(**kwargs)
    env.seed(args.seed)
    model = PPO(
        "MlpPolicy", env, seed=args.seed, verbose=1,
        n_steps=256, batch_size=512, n_epochs=8,
        gamma=0.99, gae_lambda=0.95, ent_coef=0.01, learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[128, 128]),
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"per_drone_ppo_{args.difficulty}_g{kwargs['grid_size']}_d{kwargs['n_drones']}"
    model.save(str(path))
    print(f"      saved {path}.zip")

    print("\n[3/4] Evaluating trained per-drone policy (paired) ...")
    trained, _ = paired_eval(("trained",), kwargs, args.eval_episodes, 9000,
                             model=model)

    print("\n[4/4] RESULT (identical fires, paired)")
    print("=" * 66)
    rb = base["random"][0]
    print(f"{'controller':<12}{'burned':>9}{'% grid':>9}{'extinguish':>12}{'vs random':>11}")
    rows = [("random", *base["random"]), ("greedy", *base["greedy"]),
            ("per-drone", *trained["trained"])]
    for name, b, x in rows:
        print(f"{name:<12}{b:>9.0f}{b/tot:>8.0%}{x:>11.0%}{(rb-b)/rb:>+10.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
