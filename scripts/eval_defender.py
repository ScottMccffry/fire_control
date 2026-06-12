#!/usr/bin/env python3
"""
Evaluate the trained per-drone DEFENDER policy: out-of-distribution
generalization table + a rendered episode showing the emergent containment
(no hand-coded perimeter/grid/web -- the line is entirely learned).

Usage:
  python scripts/eval_defender.py --model agents/checkpoints/defender_ppo_g48_d30_v1.zip \
      --out docs/figures/defender_learned.mp4
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
from train_defender import rollout                  # noqa: E402


def gen_table(model):
    def ev(kw, kind, eps=10, seed=9000, m=None):
        b, x = [], 0
        for e in range(eps):
            bb, xx, _ = rollout(kind, kw, seed + e, m)
            b.append(bb); x += int(xx)
        return np.mean(b) / kw["grid_size"] ** 2, x / eps
    base = dict(max_steps=160, front_band=5.0, spread_penalty=1.0)
    regimes = [
        ("train: g48 d30 1ign", dict(grid_size=48, n_drones=30, n_ignitions=1, **base)),
        ("bigger grid: g64 d30", dict(grid_size=64, n_drones=30, n_ignitions=1, max_steps=200, front_band=5.0, spread_penalty=1.0)),
        ("multi-fire: g48 3ign", dict(grid_size=48, n_drones=30, n_ignitions=3, **base)),
        ("fewer drones: g48 d18", dict(grid_size=48, n_drones=18, n_ignitions=1, **base)),
        ("big+multi: g80 d40 4ign", dict(grid_size=80, n_drones=40, n_ignitions=4, max_steps=220, front_band=5.0, spread_penalty=1.0)),
    ]
    print(f"{'regime':<28}{'rnd':>7}{'learned':>9}{'contained':>11}")
    for name, kw in regimes:
        rb, _ = ev(kw, "random")
        lb, lc = ev(kw, "trained", m=model)
        print(f"{name:<28}{rb:>6.0%}{lb:>8.0%}{lc:>10.0%}")


def render_episode(model, kw, seed, out, fps=18):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    env = DefenderSwarmVecEnv(episode_seed=seed, **kw)
    obs = env.reset()
    g = env.grid
    frames = []
    while True:
        a, _ = model.predict(obs, deterministic=True)
        env.step_async(a)
        obs, _, dones, _ = env.step_wait()
        if env.steps % 2 == 0:
            frames.append((env.sim.heat_intensity.copy(), env.sim.retardant.copy(),
                           env.drone_pos.copy(), int(env.sim.burned_area.sum())))
        if dones[0]:
            break
    tot = g * g
    w = imageio.get_writer(out, fps=fps, codec="libx264", quality=7, macro_block_size=None)
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    for t, (heat, ret, pos, burned) in enumerate(frames):
        ax.clear()
        ax.imshow(heat, origin="lower", cmap="inferno", vmin=0, vmax=1000, extent=[0, g, 0, g])
        ry, rx = np.where(ret > 0.5)
        if len(ry):
            ax.scatter(rx, ry, s=3, c="deepskyblue", alpha=0.55, marker="s")
        ax.scatter(pos[:, 1], pos[:, 0], s=8, c="white", alpha=0.85)
        ax.set_title(f"Learned per-drone defender (no hand-coded geometry)   step {t*2}\n"
                     f"grid {g}x{g}, {env.n_drones} drones, {env.n_ignitions} ignitions  |  "
                     f"burned {burned/tot:.0%}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        fig.canvas.draw()
        w.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
    w.close(); plt.close(fig)
    print(f"  wrote {out}  (final burned {frames[-1][3]/tot:.0%})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="agents/checkpoints/defender_ppo_g48_d30_v1.zip")
    p.add_argument("--out", default="docs/figures/defender_learned.mp4")
    p.add_argument("--render-grid", type=int, default=80)
    p.add_argument("--render-drones", type=int, default=40)
    p.add_argument("--render-ignitions", type=int, default=4)
    p.add_argument("--seed", type=int, default=9003)
    args = p.parse_args()
    logging.disable(logging.CRITICAL)
    from stable_baselines3 import PPO
    model = PPO.load(args.model)
    print("Out-of-distribution generalization (trained only on g48/1ign/d30):")
    gen_table(model)
    print("\nRendering an episode ...")
    kw = dict(grid_size=args.render_grid, n_drones=args.render_drones,
              n_ignitions=args.render_ignitions, max_steps=220,
              front_band=5.0, spread_penalty=1.0)
    render_episode(model, kw, args.seed, args.out)


if __name__ == "__main__":
    main()
