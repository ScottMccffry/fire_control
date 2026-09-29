#!/usr/bin/env python3
"""
Behavior-clone the hand-coded ring EXPERT into the per-drone defender policy,
then optionally RL-fine-tune. PPO-from-scratch never beat random on this hard
global line-coordination task, but the expert (each drone -> containment ring at
its own bearing) contains well; cloning it gives a working NN policy that fits
the realistic sim, and RL can refine from that good initialization.

Saves a standard PPO model (loadable by realistic_sim --pattern learned).
"""
import sys
import argparse
import logging
from pathlib import Path

import numpy as np

root = Path(__file__).parent.parent
sys.path.insert(0, str(root)); sys.path.insert(0, str(root / "scripts"))
from envs.defender_env import DefenderSwarmVecEnv  # noqa: E402


def env_kwargs(args):
    return dict(grid_size=args.grid, n_drones=args.drones, max_steps=args.max_steps,
                n_ignitions=args.ignitions, front_band=args.front_band,
                base_spread_rate=args.base_spread, wind_speed=args.wind,
                peak_kw=args.peak_kw, move_frac=args.move_frac, line_kw=args.line_kw,
                fire_headstart=args.fire_headstart, ret_capacity=args.ret_capacity,
                ret_regen_per_step=args.ret_regen, spread_penalty=args.spread_penalty)


def rollout(kind, kw, seed, model=None):
    env = DefenderSwarmVecEnv(episode_seed=seed, **kw)
    obs = env.reset()
    while True:
        if kind == "random":
            a = np.random.uniform(-1, 1, (env.n_drones, 2)).astype(np.float32)
        elif kind == "expert":
            a = env.expert_actions()
        else:
            a, _ = model.predict(obs, deterministic=True)
        env.step_async(a); obs, _, d, info = env.step_wait()
        if d[0]:
            env.close()
            return info[0]["burned_cells"]


def evaluate(kinds, kw, eps, model=None):
    tot = kw["grid_size"] ** 2
    return {k: float(np.mean([rollout(k, kw, 9000 + e, model) for e in range(eps)])) / tot
            for k in kinds}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=140)
    p.add_argument("--drones", type=int, default=750)
    p.add_argument("--ignitions", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--base-spread", type=float, default=0.45)
    p.add_argument("--wind", type=float, default=12.0)
    p.add_argument("--peak-kw", type=float, default=1400.0)
    p.add_argument("--move-frac", type=float, default=0.03)
    p.add_argument("--line-kw", type=float, default=800.0)
    p.add_argument("--fire-headstart", type=int, default=18)
    p.add_argument("--ret-capacity", type=float, default=60.0)
    p.add_argument("--ret-regen", type=float, default=1.5)
    p.add_argument("--spread-penalty", type=float, default=4.0)
    p.add_argument("--front-band", type=float, default=5.0)
    p.add_argument("--bc-steps", type=int, default=250, help="env steps of expert data to collect")
    p.add_argument("--bc-epochs", type=int, default=15)
    p.add_argument("--finetune", type=int, default=0, help="RL fine-tune frames after BC")
    p.add_argument("--eval-episodes", type=int, default=8)
    p.add_argument("--tag", default="bc")
    p.add_argument("--output", default="./agents/checkpoints")
    args = p.parse_args()
    logging.disable(logging.CRITICAL)
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import obs_as_tensor

    kw = env_kwargs(args)
    tot = args.grid ** 2
    print("=" * 64)
    print("Behavior-clone ring expert -> per-drone defender policy")
    print("=" * 64)
    base = evaluate(("random", "expert"), kw, args.eval_episodes)
    print(f"baselines: random {base['random']:.0%}  expert {base['expert']:.0%}")

    print(f"\n[1/3] Collecting expert data ({args.bc_steps} steps x {args.drones} drones) ...")
    env = DefenderSwarmVecEnv(**kw); obs = env.reset()
    O, A = [], []
    for _ in range(args.bc_steps):
        a = env.expert_actions()
        O.append(obs.copy()); A.append(a.copy())
        env.step_async(a); obs, _, _, _ = env.step_wait()
    O = np.concatenate(O); A = np.concatenate(A)
    print(f"      collected {len(O):,} samples")

    print(f"\n[2/3] Behavior cloning ({args.bc_epochs} epochs) ...")
    model = PPO("MlpPolicy", env, verbose=0, policy_kwargs=dict(net_arch=[256, 256]),
                n_steps=256, batch_size=512, learning_rate=3e-4)
    opt = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    Ot = torch.as_tensor(O, device=model.device)
    At = torch.as_tensor(A, device=model.device)
    n = len(O); bs = 4096
    for ep in range(args.bc_epochs):
        perm = torch.randperm(n)
        tot_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            dist = model.policy.get_distribution(Ot[idx])
            mean = dist.distribution.mean
            loss = ((mean - At[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += float(loss) * len(idx)
        print(f"      epoch {ep+1:2d}  mse={tot_loss/n:.4f}")

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    path = out / f"defender_ppo_g{args.grid}_d{args.drones}_{args.tag}"
    model.save(str(path))
    bc = evaluate(("trained",), kw, args.eval_episodes, model=model)
    print(f"\n[3/3] BC policy burned: {bc['trained']:.0%}   (random {base['random']:.0%}, "
          f"expert {base['expert']:.0%})")
    print(f"      saved {path}.zip")

    if args.finetune > 0:
        print(f"\n[+] RL fine-tuning {args.finetune} frames from the BC policy ...")
        env.seed(0)
        model.set_env(env)
        model.learn(total_timesteps=args.finetune, progress_bar=False)
        model.save(str(path) + "_ft")
        ft = evaluate(("trained",), kw, args.eval_episodes, model=model)
        print(f"    fine-tuned burned: {ft['trained']:.0%}   saved {path}_ft.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
