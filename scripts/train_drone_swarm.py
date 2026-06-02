#!/usr/bin/env python3
"""
Train an RL drone-swarm controller on the wildfire suppression environment.

This is the training entry point advertised in the README. It uses
stable-baselines3 PPO over the :class:`DroneFireEnv` Gymnasium environment,
which wraps the repository's own ``MockFireSimulator`` fire physics.

Examples
--------
    # Quick smoke-test run (a few thousand steps, finishes in seconds)
    python scripts/train_drone_swarm.py --timesteps 5000 --grid 16 --drones 4

    # Longer run
    python scripts/train_drone_swarm.py --timesteps 200000

The script also evaluates the trained policy against a do-nothing baseline so
you can see whether the agent actually learned to reduce burned area.
"""

import sys
import argparse
from pathlib import Path

import numpy as np

# Make the project root importable when run as a script.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs import DroneFireEnv  # noqa: E402


# Difficulty presets controlling the fire regime. "easy" reproduces the
# simulator's mild default (the fire mostly self-extinguishes); "medium" and
# "hard" produce drier, denser, faster-spreading fires that burn most of the
# grid if left unchecked, so the drone policy has real headroom to learn.
DIFFICULTY_PRESETS = {
    "easy": dict(
        grid_size=20, n_drones=6, max_steps=120, n_ignitions=2,
        wind_speed=6.0, fuel_moisture_range=(0.1, 0.4),
        fuel_load_range=(0.5, 1.0), base_spread_rate=0.10,
    ),
    "medium": dict(
        grid_size=32, n_drones=16, max_steps=200, n_ignitions=3,
        wind_speed=9.0, fuel_moisture_range=(0.06, 0.22),
        fuel_load_range=(0.65, 1.0), base_spread_rate=0.17,
    ),
    "hard": dict(
        grid_size=40, n_drones=24, max_steps=220, n_ignitions=4,
        wind_speed=12.0, fuel_moisture_range=(0.05, 0.18),
        fuel_load_range=(0.75, 1.0), base_spread_rate=0.20,
    ),
}


def build_env_kwargs(args):
    """Merge difficulty preset with any explicit CLI overrides."""
    kwargs = dict(DIFFICULTY_PRESETS[args.difficulty])
    overrides = {
        "grid_size": args.grid,
        "n_drones": args.drones,
        "max_steps": args.max_steps,
        "wind_speed": args.wind_speed,
        "n_ignitions": args.ignitions,
    }
    for key, val in overrides.items():
        if val is not None:
            kwargs[key] = val
    return kwargs


def make_env(args):
    return DroneFireEnv(**build_env_kwargs(args))


def evaluate(model, env, episodes: int = 5, deterministic: bool = True, normalizer=None):
    """Roll out a policy (or None for a random/no-op baseline) and average stats.

    If ``normalizer`` (a fitted VecNormalize) is given, raw observations are
    normalized with its running statistics before being passed to the policy,
    matching the way the policy was trained.
    """
    burned, rewards, extinguished = [], [], 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        info = {}
        while not done:
            if model is None:
                action = env.action_space.sample()
            else:
                pred_obs = normalizer.normalize_obs(obs) if normalizer is not None else obs
                action, _ = model.predict(pred_obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        burned.append(info.get("burned_cells", 0))
        rewards.append(ep_reward)
        extinguished += int(info.get("fire_extinguished", False))
    return {
        "mean_burned_cells": float(np.mean(burned)),
        "mean_reward": float(np.mean(rewards)),
        "extinguish_rate": extinguished / episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Train RL drone swarm for wildfire suppression")
    parser.add_argument("--timesteps", type=int, default=20000, help="Total training timesteps")
    parser.add_argument("--difficulty", choices=list(DIFFICULTY_PRESETS), default="medium",
                        help="Fire-regime difficulty preset")
    parser.add_argument("--grid", type=int, default=None, help="Override grid size (NxN)")
    parser.add_argument("--drones", type=int, default=None, help="Override number of drones")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps per episode")
    parser.add_argument("--wind-speed", type=float, default=None, help="Override wind speed (m/s)")
    parser.add_argument("--ignitions", type=int, default=None, help="Override number of ignition points")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel environments for training")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes for evaluation")
    parser.add_argument("--output", default="./agents/checkpoints", help="Where to save the model")
    parser.add_argument("--algo", choices=["ppo", "dqn"], default="ppo", help="RL algorithm")
    args = parser.parse_args()

    # Imported here so `--help` works even before sb3 is installed.
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.env_util import make_vec_env

    cfg = build_env_kwargs(args)
    print("=" * 64)
    print("RL Drone Swarm Wildfire Suppression — Training")
    print("=" * 64)
    print(f"difficulty={args.difficulty}  grid={cfg['grid_size']}x{cfg['grid_size']}  "
          f"drones={cfg['n_drones']}  wind={cfg['wind_speed']}m/s")
    print(f"algo={args.algo.upper()}  timesteps={args.timesteps}  n_envs={args.n_envs}")

    # Validate the environment conforms to the Gymnasium API.
    print("\n[1/4] Validating environment against Gymnasium API ...")
    check_env(make_env(args), warn=True)
    print("      OK")

    # Vectorized training envs to use multiple CPU cores. Use the "fork" start
    # method so the env factory (a closure) works without needing to be pickled.
    vec_cls = SubprocVecEnv if args.n_envs > 1 else None
    vec_kwargs = {"start_method": "fork"} if vec_cls is SubprocVecEnv else {}
    env = make_vec_env(
        lambda: make_env(args),
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=vec_cls,
        vec_env_kwargs=vec_kwargs,
    )
    # Normalize observations and rewards: PPO is sensitive to input/return
    # scale, and the raw reward here spans large negative values dominated by
    # natural fire spread. Normalization is what lets the policy learn.
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Baseline (random actions) before training, for comparison.
    print("\n[2/4] Evaluating random baseline ...")
    baseline = evaluate(None, make_env(args), episodes=args.eval_episodes)
    print(f"      baseline: burned={baseline['mean_burned_cells']:.1f} cells, "
          f"reward={baseline['mean_reward']:.2f}, "
          f"extinguish_rate={baseline['extinguish_rate']:.0%}")

    if args.algo == "dqn":
        # DQN needs a Discrete action space; MultiDiscrete isn't supported.
        raise SystemExit(
            "DQN does not support the MultiDiscrete action space used here; "
            "use --algo ppo (the default)."
        )

    print("\n[3/4] Training PPO ...")
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        ent_coef=0.01,
        learning_rate=3e-4,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"drone_swarm_{args.algo}_{args.difficulty}_g{cfg['grid_size']}_d{cfg['n_drones']}"
    model.save(str(model_path))
    env.save(str(model_path) + "_vecnormalize.pkl")
    print(f"      model saved to {model_path}.zip")

    print("\n[4/4] Evaluating trained policy ...")
    trained = evaluate(model, make_env(args), episodes=args.eval_episodes, normalizer=env)
    print(f"      trained:  burned={trained['mean_burned_cells']:.1f} cells, "
          f"reward={trained['mean_reward']:.2f}, "
          f"extinguish_rate={trained['extinguish_rate']:.0%}")

    print("\n" + "=" * 64)
    print("RESULT")
    print("=" * 64)
    improvement = baseline["mean_burned_cells"] - trained["mean_burned_cells"]
    print(f"Burned-area reduction vs random baseline: {improvement:+.1f} cells "
          f"({improvement / max(1.0, baseline['mean_burned_cells']):+.1%})")
    print(f"Mean reward: {baseline['mean_reward']:.2f} (baseline) -> "
          f"{trained['mean_reward']:.2f} (trained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
