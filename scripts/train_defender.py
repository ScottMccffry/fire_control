#!/usr/bin/env python3
"""
Train a decentralized PER-DRONE DEFENDER policy (parameter sharing) with PPO.

Each drone runs the same small policy on its egocentric view of the live fire
front and decides where to move and lay retardant, so the fleet learns to build
a containment line that minimises burned area -- the learned, continuously-
adaptive alternative to the hand-coded perimeter/grid in scripts/realistic_sim.py.

Example:
    python scripts/train_defender.py --timesteps 1500000 --grid 48 --drones 30
"""
import sys
import argparse
import logging
from pathlib import Path

import numpy as np

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "scripts"))
from envs.defender_env import DefenderSwarmVecEnv  # noqa: E402


def rollout(kind, kwargs, seed, model=None):
    env = DefenderSwarmVecEnv(episode_seed=seed, **kwargs)
    obs = env.reset()
    while True:
        if kind == "random":
            a = np.random.uniform(-1, 1, (env.n_drones, 2)).astype(np.float32)
        elif kind == "greedy":
            a = env.greedy_actions()
        else:
            a, _ = model.predict(obs, deterministic=True)
        env.step_async(a)
        obs, _, dones, infos = env.step_wait()
        if dones[0]:
            env.close()
            return infos[0]["burned_cells"], infos[0]["fire_extinguished"], infos[0]["retardant_laid"]


def paired_eval(kinds, kwargs, episodes, base_seed, model=None):
    tot = kwargs["grid_size"] ** 2
    res = {}
    for k in kinds:
        b, x, rl = [], 0, []
        for e in range(episodes):
            bb, xx, rr = rollout(k, kwargs, base_seed + e, model)
            b.append(bb); x += int(xx); rl.append(rr)
        res[k] = (float(np.mean(b)), x / episodes, float(np.mean(rl)))
    return res, tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=1_500_000)
    p.add_argument("--grid", type=int, default=48)
    p.add_argument("--drones", type=int, default=30)
    p.add_argument("--ignitions", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=160)
    p.add_argument("--front-band", type=float, default=5.0)
    p.add_argument("--spread-penalty", type=float, default=1.0)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--eval-episodes", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--output", default="./agents/checkpoints")
    args = p.parse_args()

    logging.disable(logging.CRITICAL)
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    kwargs = dict(grid_size=args.grid, n_drones=args.drones, max_steps=args.max_steps,
                  n_ignitions=args.ignitions, front_band=args.front_band,
                  spread_penalty=args.spread_penalty)
    tot = args.grid ** 2
    print("=" * 64)
    print("Per-Drone DEFENDER Shared-Policy PPO")
    print("=" * 64)
    print(f"grid={args.grid} drones={args.drones} ignitions={args.ignitions} "
          f"front_band={args.front_band} frames={args.timesteps}")

    print("\n[1/4] Baselines on identical fires ...")
    base, _ = paired_eval(("random", "greedy"), kwargs, args.eval_episodes, 9000)
    for k, (b, x, rl) in base.items():
        print(f"      {k:7s}: {b:7.0f} burned ({b/tot:.0%})  contained={x:.0%}  retardant={rl:.0f}")

    print("\n[2/4] Training ...")
    env = DefenderSwarmVecEnv(**kwargs)
    env.seed(args.seed)
    model = PPO("MlpPolicy", env, seed=args.seed, verbose=1,
                n_steps=256, batch_size=512, n_epochs=8, gamma=0.99,
                gae_lambda=0.95, ent_coef=args.ent_coef, learning_rate=3e-4,
                policy_kwargs=dict(net_arch=[128, 128]))
    ckpt = CheckpointCallback(save_freq=25000, save_path=str(Path(args.output) / "ckpt"),
                              name_prefix=f"defender_{args.tag or 'run'}")
    model.learn(total_timesteps=args.timesteps, progress_bar=False, callback=ckpt)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    path = out / f"defender_ppo_g{args.grid}_d{args.drones}{suffix}"
    model.save(str(path))
    print(f"      saved {path}.zip")

    print("\n[3/4] Evaluating trained defender (paired) ...")
    trained, _ = paired_eval(("trained",), kwargs, args.eval_episodes, 9000, model=model)

    print("\n[4/4] RESULT (identical fires, paired)")
    print("=" * 64)
    rb = base["random"][0]
    print(f"{'controller':<12}{'burned':>9}{'%grid':>7}{'contained':>11}{'retardant':>11}{'vs random':>11}")
    for name, (b, x, rl) in [("random", base["random"]), ("greedy", base["greedy"]),
                             ("defender", trained["trained"])]:
        print(f"{name:<12}{b:>9.0f}{b/tot:>6.0%}{x:>10.0%}{rl:>11.0f}{(rb-b)/rb:>+10.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
