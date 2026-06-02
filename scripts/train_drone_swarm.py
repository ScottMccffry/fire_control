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


def make_env(args):
    return DroneFireEnv(
        grid_size=args.grid,
        n_drones=args.drones,
        max_steps=args.max_steps,
        wind_speed=args.wind_speed,
    )


def evaluate(model, env, episodes: int = 5, deterministic: bool = True):
    """Roll out a policy (or None for a random/no-op baseline) and average stats."""
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
                action, _ = model.predict(obs, deterministic=deterministic)
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
    parser.add_argument("--grid", type=int, default=20, help="Fire grid size (NxN)")
    parser.add_argument("--drones", type=int, default=6, help="Number of drones to control")
    parser.add_argument("--max-steps", type=int, default=120, help="Max steps per episode")
    parser.add_argument("--wind-speed", type=float, default=6.0, help="Wind speed (m/s)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes for evaluation")
    parser.add_argument("--output", default="./agents/checkpoints", help="Where to save the model")
    parser.add_argument("--algo", choices=["ppo", "dqn"], default="ppo", help="RL algorithm")
    args = parser.parse_args()

    # Imported here so `--help` works even before sb3 is installed.
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor

    print("=" * 64)
    print("RL Drone Swarm Wildfire Suppression — Training")
    print("=" * 64)
    print(f"grid={args.grid}x{args.grid}  drones={args.drones}  "
          f"algo={args.algo.upper()}  timesteps={args.timesteps}")

    # Validate the environment conforms to the Gymnasium API.
    print("\n[1/4] Validating environment against Gymnasium API ...")
    check_env(make_env(args), warn=True)
    print("      OK")

    env = Monitor(make_env(args))

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
    model_path = out_dir / f"drone_swarm_{args.algo}_g{args.grid}_d{args.drones}"
    model.save(str(model_path))
    print(f"      model saved to {model_path}.zip")

    print("\n[4/4] Evaluating trained policy ...")
    trained = evaluate(model, make_env(args), episodes=args.eval_episodes)
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
