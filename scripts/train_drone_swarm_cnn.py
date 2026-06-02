#!/usr/bin/env python3
"""
Train a CNN drone-swarm controller with a behavior-cloning warm-start.

This is the scalable RL path: a convolutional policy over spatial fire/drone
maps (see agents/cnn_policy.py), warm-started by imitating the greedy controller
and then fine-tuned with PPO. The motivation (see the project history): a flat
MLP policy did not learn on large grids, while a hand-coded greedy controller
saves ~60-70% of burned area. Behavior cloning starts PPO from that strong
behavior instead of from scratch.

Pipeline
--------
  1. Validate the spatial env.
  2. Evaluate random + greedy baselines on identical fires.
  3. Collect a greedy demonstration dataset.
  4. Behavior-clone the CNN policy on it (supervised).
  5. Evaluate the cloned policy.
  6. Fine-tune with PPO.
  7. Evaluate the final policy and save it.

Example
-------
    python scripts/train_drone_swarm_cnn.py --difficulty medium \
        --bc-steps 40000 --bc-epochs 6 --timesteps 150000
"""

import sys
import argparse
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from envs import DroneFireEnv  # noqa: E402
from agents import FireSwarmExtractor  # noqa: E402
from train_drone_swarm import DIFFICULTY_PRESETS  # noqa: E402
from evaluate_policy import greedy_action  # noqa: E402


def make_spatial_env(cfg):
    return DroneFireEnv(obs_mode="spatial", **cfg)


def paired_eval(cfg, episodes, base_seed, policy_fn):
    """Mean burned cells + extinguish rate for a controller over seeded fires."""
    env = make_spatial_env(cfg)
    burned, ext = [], 0
    for i in range(episodes):
        obs, _ = env.reset(seed=base_seed + i)
        done = False
        info = {}
        while not done:
            action = policy_fn(env, obs)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        burned.append(info["burned_cells"])
        ext += int(info["fire_extinguished"])
    return float(np.mean(burned)), ext / episodes


def collect_greedy_dataset(cfg, n_steps, base_seed):
    """Roll out greedy and record (map, drones, action) transitions."""
    env = make_spatial_env(cfg)
    maps, drones, actions = [], [], []
    obs, _ = env.reset(seed=base_seed)
    ep = 0
    while len(actions) < n_steps:
        a = greedy_action(env)
        maps.append(obs["map"])
        drones.append(obs["drones"])
        actions.append(a)
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            ep += 1
            obs, _ = env.reset(seed=base_seed + 1000 + ep)
    return (
        np.asarray(maps, dtype=np.float32),
        np.asarray(drones, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
    )


def warmup_value(model, cfg, n_steps, gamma=0.99, lr=1e-3, epochs=4):
    """Fit the value network to the current (behavior-cloned) policy's returns.

    BC only trains the actor; the critic starts random, which makes the first
    PPO updates produce noisy advantages that push the good actor off the BC
    solution. Pre-fitting the critic (on frozen features) avoids that.
    """
    import torch

    env = make_spatial_env(cfg)
    maps, drones, rewards, dones = [], [], [], []
    obs, _ = env.reset(seed=12345)
    for t in range(n_steps):
        a = model.predict(obs, deterministic=False)[0]
        maps.append(obs["map"])
        drones.append(obs["drones"])
        obs, r, term, trunc, _ = env.step(a)
        rewards.append(r)
        dones.append(term or trunc)
        if term or trunc:
            obs, _ = env.reset(seed=12345 + t + 1)

    # Monte-Carlo discounted returns (reset at episode boundaries).
    returns = np.zeros(len(rewards), dtype=np.float32)
    g = 0.0
    for t in reversed(range(len(rewards))):
        if dones[t]:
            g = 0.0
        g = rewards[t] + gamma * g
        returns[t] = g

    device = model.device
    maps_t = torch.as_tensor(np.asarray(maps, dtype=np.float32), device=device)
    drones_t = torch.as_tensor(np.asarray(drones, dtype=np.float32), device=device)
    ret_t = torch.as_tensor(returns, device=device)
    # Train only the critic head; leave the shared extractor and actor untouched.
    params = (list(model.policy.value_net.parameters())
              + list(model.policy.mlp_extractor.value_net.parameters()))
    opt = torch.optim.Adam(params, lr=lr)
    n = len(returns)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for s in range(0, n, 256):
            idx = perm[s:s + 256]
            obs = {"map": maps_t[idx], "drones": drones_t[idx]}
            values = model.policy.predict_values(obs).squeeze(-1)
            loss = ((values - ret_t[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"      value warmup epoch {epoch + 1}/{epochs}  mse={total / n:.2f}")


def behavior_clone(model, dataset, epochs, batch_size, lr):
    """Supervised imitation: maximize log-prob of greedy actions under policy."""
    import torch

    maps, drones, actions = dataset
    device = model.device
    policy = model.policy
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    maps_t = torch.as_tensor(maps, device=device)
    drones_t = torch.as_tensor(drones, device=device)
    actions_t = torch.as_tensor(actions, device=device)
    n = len(actions)

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total, nb = 0.0, 0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            obs = {"map": maps_t[idx], "drones": drones_t[idx]}
            _, log_prob, entropy = policy.evaluate_actions(obs, actions_t[idx])
            loss = -log_prob.mean() - 0.001 * entropy.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            nb += len(idx)
        print(f"      BC epoch {epoch + 1}/{epochs}  loss={total / nb:.4f}")


def main():
    p = argparse.ArgumentParser(description="CNN policy + behavior-cloning warm-start")
    p.add_argument("--difficulty", choices=list(DIFFICULTY_PRESETS), default="medium")
    p.add_argument("--bc-steps", type=int, default=40000, help="Greedy transitions to collect")
    p.add_argument("--bc-epochs", type=int, default=6, help="Behavior-cloning epochs")
    p.add_argument("--bc-batch", type=int, default=256)
    p.add_argument("--bc-lr", type=float, default=3e-4)
    p.add_argument("--timesteps", type=int, default=150000, help="PPO fine-tune timesteps")
    p.add_argument("--ppo-lr", type=float, default=1e-4, help="PPO fine-tune learning rate")
    p.add_argument("--ent-coef", type=float, default=0.0, help="PPO entropy coefficient")
    p.add_argument("--target-kl", type=float, default=0.03, help="PPO KL drift limit")
    p.add_argument("--value-warmup", type=int, default=20000,
                   help="Steps to warm up the critic on the BC policy before PPO")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--eval-episodes", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-bc", action="store_true", help="Ablation: PPO from scratch")
    p.add_argument("--skip-ppo", action="store_true", help="Only behavior-clone")
    p.add_argument("--output", default="./agents/checkpoints")
    args = p.parse_args()

    import logging
    logging.disable(logging.CRITICAL)
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.env_util import make_vec_env

    cfg = DIFFICULTY_PRESETS[args.difficulty]
    grid, tot = cfg["grid_size"], cfg["grid_size"] ** 2

    print("=" * 66)
    print("CNN Drone Swarm + Behavior-Cloning Warm-Start")
    print("=" * 66)
    print(f"difficulty={args.difficulty}  grid={grid}x{grid}  drones={cfg['n_drones']}")
    print(f"bc_steps={args.bc_steps}  bc_epochs={args.bc_epochs}  "
          f"ppo_timesteps={args.timesteps}  n_envs={args.n_envs}")

    print("\n[1/7] Validating spatial environment ...")
    check_env(make_spatial_env(cfg), warn=True)
    print("      OK")

    print("\n[2/7] Baselines on identical fires ...")
    rand_burned, rand_ext = paired_eval(
        cfg, args.eval_episodes, 9000, lambda e, o: e.action_space.sample())
    greedy_burned, greedy_ext = paired_eval(
        cfg, args.eval_episodes, 9000, lambda e, o: greedy_action(e))
    print(f"      random: {rand_burned:.0f} ({rand_burned/tot:.0%})  ext={rand_ext:.0%}")
    print(f"      greedy: {greedy_burned:.0f} ({greedy_burned/tot:.0%})  ext={greedy_ext:.0%}  "
          f"saves {(rand_burned-greedy_burned)/rand_burned:.0%}")

    # Build the PPO model with the CNN policy over the Dict observation.
    vec_kwargs = {"start_method": "fork"} if args.n_envs > 1 else {}
    env = make_vec_env(
        lambda: make_spatial_env(cfg),
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv if args.n_envs > 1 else None,
        vec_env_kwargs=vec_kwargs,
    )
    # No VecNormalize: observations are already in [0,1], and PPO normalizes
    # advantages internally. Keeping raw rewards means the behavior-cloned
    # critic warmup and PPO's value targets share the same return scale.

    policy_kwargs = dict(
        features_extractor_class=FireSwarmExtractor,
        features_extractor_kwargs=dict(cnn_out=128, drone_out=64),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )
    # Fine-tune-friendly PPO: no entropy bonus (don't push the sharp BC policy
    # back toward randomness), small LR, and a KL limit so it stays near BC.
    model = PPO(
        "MultiInputPolicy", env, seed=args.seed, verbose=1,
        n_steps=512, batch_size=256, n_epochs=5, gamma=0.99, gae_lambda=0.95,
        ent_coef=args.ent_coef, learning_rate=args.ppo_lr, target_kl=args.target_kl,
        clip_range=0.2, policy_kwargs=policy_kwargs,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"cnn_{'bc_' if not args.skip_bc else ''}{args.difficulty}_g{grid}_d{cfg['n_drones']}"
    model_path = out_dir / f"drone_swarm_{tag}"

    if not args.skip_bc:
        print(f"\n[3/7] Collecting greedy dataset ({args.bc_steps} transitions) ...")
        dataset = collect_greedy_dataset(cfg, args.bc_steps, args.seed)
        print(f"      collected {len(dataset[2])} transitions")

        print(f"\n[4/7] Behavior cloning ({args.bc_epochs} epochs) ...")
        behavior_clone(model, dataset, args.bc_epochs, args.bc_batch, args.bc_lr)

        print("\n[5/7] Evaluating cloned policy ...")
        bc_burned, bc_ext = paired_eval(
            cfg, args.eval_episodes, 9000,
            lambda e, o: model.predict(o, deterministic=True)[0])
        print(f"      cloned: {bc_burned:.0f} ({bc_burned/tot:.0%})  ext={bc_ext:.0%}  "
              f"saves {(rand_burned-bc_burned)/rand_burned:+.0%} vs random")
        # Persist the behavior-cloned policy separately: it is often the best
        # policy (PPO fine-tuning can mildly degrade it when greedy is already
        # near-optimal under the current reward).
        model.save(str(model_path) + "_bc")
        print(f"      saved {model_path}_bc.zip")
    else:
        print("\n[3-5/7] Skipping behavior cloning (ablation: PPO from scratch)")

    if not args.skip_ppo and args.timesteps > 0:
        if not args.skip_bc and args.value_warmup > 0:
            print(f"\n[5b/7] Warming up critic on BC policy ({args.value_warmup} steps) ...")
            warmup_value(model, cfg, args.value_warmup)
        print(f"\n[6/7] PPO fine-tuning ({args.timesteps} timesteps) ...")
        model.learn(total_timesteps=args.timesteps, progress_bar=False)

    model.save(str(model_path))
    print(f"      saved {model_path}.zip")

    print("\n[7/7] Final evaluation ...")
    final_burned, final_ext = paired_eval(
        cfg, args.eval_episodes, 9000,
        lambda e, o: model.predict(o, deterministic=True)[0])

    print("\n" + "=" * 66)
    print("RESULT (identical fires, paired)")
    print("=" * 66)
    print(f"{'controller':<16}{'burned':>10}{'% grid':>9}{'extinguish':>12}{'vs random':>12}")
    for name, b, e in (
        ("random", rand_burned, rand_ext),
        ("greedy", greedy_burned, greedy_ext),
        ("trained (final)", final_burned, final_ext),
    ):
        print(f"{name:<16}{b:>10.0f}{b/tot:>8.0%}{e:>11.0%}"
              f"{(rand_burned-b)/rand_burned:>+11.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
