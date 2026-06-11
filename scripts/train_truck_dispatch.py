#!/usr/bin/env python3
"""
Train the decentralized dispatch policy on the truck-logistics env.

Drones learn to fight the fire AND manage refuelling at mobile trucks (observing
each truck's remaining water). Train small (scale-free), deploy large.

    python scripts/train_truck_dispatch.py --timesteps 3000000
"""
import sys, argparse, logging
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.truck_dispatch_env import TruckDispatchVecEnv  # noqa: E402


def rollout(kind, kw, seed, model=None):
    env = TruckDispatchVecEnv(episode_seed=seed, **kw)
    obs = env.reset()
    while True:
        if kind == "random":
            a = np.random.uniform(-1, 1, (env.n_drones, 2)).astype(np.float32)
        elif kind == "greedy":
            a = env.greedy_actions()
        else:
            a, _ = model.predict(obs, deterministic=True)
        env.step_async(a); obs, _, d, info = env.step_wait()
        if d[0]:
            env.close(); return info[0]["burned_cells"], info[0]["fire_extinguished"]


def paired(kinds, kw, eps, base, model=None):
    tot = kw["grid_size"] ** 2; out = {}
    for k in kinds:
        b, e = [], 0
        for i in range(eps):
            bb, xx = rollout(k, kw, base + i, model); b.append(bb); e += int(xx)
        out[k] = (float(np.mean(b)), e / eps)
    return out, tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--grid", type=int, default=120)
    p.add_argument("--trucks", type=int, default=3)
    p.add_argument("--dpt", type=int, default=60)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--base-spread", type=float, default=0.16)
    p.add_argument("--water-cost", type=float, default=0.2)
    p.add_argument("--eval-episodes", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--output", default="./agents/checkpoints")
    args = p.parse_args()
    logging.disable(logging.CRITICAL)
    from stable_baselines3 import PPO

    kw = dict(grid_size=args.grid, n_trucks=args.trucks, drones_per_truck=args.dpt,
              max_steps=args.max_steps, water_cost=args.water_cost,
              base_spread_rate=args.base_spread)
    print("=" * 64); print("Truck-Dispatch Per-Drone PPO"); print("=" * 64)
    print(f"grid={args.grid} trucks={args.trucks} dpt={args.dpt} "
          f"drones={args.trucks*args.dpt} frames={args.timesteps}")

    print("\n[1/4] Baselines ...")
    base, tot = paired(("random", "greedy"), kw, args.eval_episodes, 9000)
    for k, (b, e) in base.items():
        print(f"  {k:7s}: {b:7.0f} burned ({b/tot:.0%}) ext={e:.0%}")

    print("\n[2/4] Training ...")
    env = TruckDispatchVecEnv(**kw); env.seed(args.seed)
    model = PPO("MlpPolicy", env, seed=args.seed, verbose=1, n_steps=256,
                batch_size=512, n_epochs=8, gamma=0.99, gae_lambda=0.95,
                ent_coef=0.01, learning_rate=3e-4, policy_kwargs=dict(net_arch=[128, 128]))
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    suf = f"_{args.tag}" if args.tag else ""
    path = out / f"truck_dispatch_g{args.grid}_t{args.trucks}_d{args.dpt}{suf}"
    model.save(str(path)); print(f"  saved {path}.zip")

    print("\n[3/4] Eval trained ...")
    tr, _ = paired(("trained",), kw, args.eval_episodes, 9000, model)
    print("\n[4/4] RESULT (paired, identical fires)"); print("=" * 64)
    rb = base["random"][0]
    rows = [("random", *base["random"]), ("greedy", *base["greedy"]), ("dispatch", *tr["trained"])]
    print(f"{'controller':<11}{'burned':>9}{'%grid':>8}{'ext':>7}{'vs random':>11}")
    for n, b, e in rows:
        print(f"{n:<11}{b:>9.0f}{b/tot:>7.0%}{e:>6.0%}{(rb-b)/rb:>+10.0%}")


if __name__ == "__main__":
    main()
